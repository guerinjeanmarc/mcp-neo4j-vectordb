import json
import logging
from typing import Literal, Optional

import litellm
from fastmcp.exceptions import ToolError
from fastmcp.server import FastMCP
from fastmcp.tools.tool import TextContent, ToolResult
from mcp.types import ToolAnnotations
from neo4j import AsyncDriver, AsyncGraphDatabase, RoutingControl
from neo4j.exceptions import ClientError, Neo4jError
from pydantic import Field

from .utils import (
    _value_sanitize, 
    _truncate_results_to_token_limit,
    MAX_LIST_SIZE,
    MAX_STRING_SIZE,
    RESPONSE_TOKEN_LIMIT,
)

logger = logging.getLogger("mcp_neo4j_vectordb")


def _format_namespace(namespace: str) -> str:
    """Format namespace with trailing dash if needed."""
    if namespace:
        return namespace if namespace.endswith("-") else namespace + "-"
    return ""


def create_mcp_server(
    neo4j_driver: AsyncDriver,
    embedding_model: str,
    database: str = "neo4j",
    namespace: str = "",
    read_timeout: int = 30,
    config_sample_size: int = 1000,
) -> FastMCP:
    """Create the Neo4j Vector Database MCP server - mimics a pure vector DB (no graph traversal)."""
    
    mcp: FastMCP = FastMCP("mcp-neo4j-vectordb")
    namespace_prefix = _format_namespace(namespace)

    # ========================================
    # DISCOVERY TOOL
    # ========================================

    @mcp.tool(
        name=namespace_prefix + "get_searchable_content",
        annotations=ToolAnnotations(
            title="Get Searchable Content",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def get_searchable_content(
        sample_size: int = Field(
            default=config_sample_size,
            description="The sample size used to infer property sizes. Larger samples are slower but more accurate."
        )
    ) -> list[ToolResult]:
        """
        Returns vector and fulltext indexes with searchable node properties and size warnings.
        
        **IMPORTANT: Call this tool BEFORE using any search tools (vector_search, fulltext_search).**
        
        This tool provides:
        - Vector & fulltext indexes available for search
        - Node properties ONLY for labels that have indexes (no relationships, no non-indexed nodes)
        - Property types and size warnings to help you choose efficient return_properties
        
        Property size warnings help you avoid token limits when using search tools.
        For example, if a property has warning "avg ~100-200KB", avoid returning it unless necessary.
        
        You should only provide a `sample_size` value if requested by the user, or tuning performance.
        """
        effective_sample_size = sample_size if sample_size else config_sample_size
        logger.info(f"Running `get_searchable_content` with sample size {effective_sample_size}")

        # Step 1: Get search indexes
        vector_index_query = """
        SHOW INDEXES
        YIELD name, type, entityType, labelsOrTypes, properties, options
        WHERE type = 'VECTOR'
        RETURN name, entityType, labelsOrTypes, properties, options
        """

        fulltext_index_query = """
        SHOW INDEXES
        YIELD name, type, entityType, labelsOrTypes, properties, options
        WHERE type = 'FULLTEXT'
        RETURN name, entityType, labelsOrTypes, properties, options
        """

        # Step 2: Get schema using APOC
        get_schema_query = f"CALL apoc.meta.schema({{sample: {effective_sample_size}}}) YIELD value RETURN value"

        try:
            # Fetch indexes
            vector_indexes = await neo4j_driver.execute_query(
                vector_index_query,
                routing_control=RoutingControl.READ,
                database_=database,
                result_transformer_=lambda r: r.data(),
            )

            fulltext_indexes = await neo4j_driver.execute_query(
                fulltext_index_query,
                routing_control=RoutingControl.READ,
                database_=database,
                result_transformer_=lambda r: r.data(),
            )

            # Fetch schema
            schema_results = await neo4j_driver.execute_query(
                get_schema_query,
                routing_control=RoutingControl.READ,
                database_=database,
                result_transformer_=lambda r: r.data(),
            )

            # Step 3: Sample property sizes for indexed labels
            indexed_labels = set()
            for idx in vector_indexes:
                indexed_labels.update(idx.get("labelsOrTypes", []))
            for idx in fulltext_indexes:
                indexed_labels.update(idx.get("labelsOrTypes", []))

            property_size_warnings = {}
            for label in indexed_labels:
                # Sample property sizes for this label
                size_query = f"""
                MATCH (n:{label})
                WITH n LIMIT {min(effective_sample_size, 100)}
                WITH n, properties(n) as props
                UNWIND keys(props) as propName
                WITH propName, props[propName] as propValue
                WHERE propValue IS NOT NULL
                WITH propName,
                     valueType(propValue) as propType,
                     CASE
                         WHEN valueType(propValue) STARTS WITH 'LIST' THEN size(propValue)
                         WHEN valueType(propValue) STARTS WITH 'STRING' THEN size(propValue)
                         ELSE 0
                     END as propSize
                RETURN propName, propType, avg(propSize) as avgSize, max(propSize) as maxSize
                ORDER BY avgSize DESC
                """
                
                try:
                    size_results = await neo4j_driver.execute_query(
                        size_query,
                        routing_control=RoutingControl.READ,
                        database_=database,
                        result_transformer_=lambda r: r.data(),
                    )
                    
                    if label not in property_size_warnings:
                        property_size_warnings[label] = {}
                    
                    logger.debug(f"Sampled {len(size_results)} properties for label {label}")
                    
                    for row in size_results:
                        prop_name = row["propName"]
                        prop_type = row["propType"]
                        avg_size = row["avgSize"]
                        max_size = row["maxSize"]
                        
                        logger.debug(f"  {prop_name} ({prop_type}): avg={avg_size}, max={max_size}")
                        
                        # Generate warning for large properties based on type
                        warning = None
                        if prop_type.startswith("STRING"):
                            # For strings, warn if >= 100KB
                            if avg_size >= 100000:
                                warning = f"very large (avg ~{int(avg_size/1000)}KB, max ~{int(max_size/1000)}KB)"
                        elif prop_type.startswith("LIST"):
                            # For lists, warn if >= 1000 items (likely embeddings)
                            if avg_size >= 1000:
                                warning = f"large list (avg ~{int(avg_size)} items)"
                        
                        if warning:
                            property_size_warnings[label][prop_name] = warning
                
                except Exception as e:
                    logger.warning(f"Could not sample property sizes for label {label}: {e}")
                    continue

            # Step 4: Clean and enrich schema with warnings (only indexed nodes, no relationships)
            def clean_and_enrich_schema(schema: dict, indexed_labels: set) -> dict:
                cleaned = {}
                for key, entry in schema.items():
                    # ONLY include nodes that have vector or fulltext indexes
                    if key not in indexed_labels:
                        continue
                    
                    new_entry = {"type": entry["type"]}
                    if "count" in entry:
                        new_entry["count"] = entry["count"]

                    labels = entry.get("labels", [])
                    if labels:
                        new_entry["labels"] = labels

                    props = entry.get("properties", {})
                    clean_props = {}
                    for pname, pinfo in props.items():
                        cp = {}
                        if "indexed" in pinfo:
                            cp["indexed"] = pinfo["indexed"]
                        if "type" in pinfo:
                            cp["type"] = pinfo["type"]
                        
                        # Add size warning if available
                        if key in property_size_warnings and pname in property_size_warnings[key]:
                            cp["warning"] = property_size_warnings[key][pname]
                        
                        if cp:
                            clean_props[pname] = cp
                    if clean_props:
                        new_entry["properties"] = clean_props

                    # NO relationships in vector DB mimic mode
                    # Relationships are intentionally excluded to mimic pure vector database behavior

                    cleaned[key] = new_entry
                return cleaned

            schema_clean = clean_and_enrich_schema(schema_results[0].get("value"), indexed_labels)

            # Step 5: Combine everything into compact JSON
            result = {
                "indexes": {
                    "vector": vector_indexes,
                    "fulltext": fulltext_indexes
                },
                "schema": schema_clean
            }

            result_json = json.dumps(result, default=str)
            logger.debug(f"Found {len(vector_indexes)} vector indexes, {len(fulltext_indexes)} fulltext indexes")

            return ToolResult(content=[TextContent(type="text", text=result_json)])

        except ClientError as e:
            if "Neo.ClientError.Procedure.ProcedureNotFound" in str(e):
                raise ToolError(
                    "Neo4j Client Error: This instance of Neo4j does not have the APOC plugin installed. Please install and enable APOC."
                )
            else:
                raise ToolError(f"Neo4j Client Error: {e}")

        except Neo4jError as e:
            raise ToolError(f"Neo4j Error: {e}")

        except Exception as e:
            logger.error(f"Error retrieving Neo4j schema and indexes: {e}")
            raise ToolError(f"Unexpected Error: {e}")

    # ========================================
    # SIMPLE SEARCH TOOLS
    # ========================================

    @mcp.tool(
        name=namespace_prefix + "vector_search",
        annotations=ToolAnnotations(
            title="Vector Similarity Search",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def vector_search(
        text_query: str = Field(..., description="The text query to search for. This will be embedded and used for similarity search."),
        vector_index: str = Field(..., description="The name of the vector index to search in. Use get_searchable_content to see available indexes."),
        top_k: int = Field(default=5, description="The number of most similar results to return."),
        return_properties: Optional[str] = Field(
            None,
            description='Optional: Comma-separated list of properties to return (e.g., "pageNumber,id"). If not specified, returns all properties with automatic sanitization (large values are truncated).'
        ),
    ) -> list[ToolResult]:
        """
        Performs vector similarity search on a Neo4j vector index.
        
        This tool embeds your text query using OpenAI and searches the specified vector index.
        Returns node IDs, labels, node properties (automatically sanitized), and similarity scores.
        
        **Automatic Sanitization (always applied):**
        - Embedding property used by the vector index → automatically excluded (vector_search only)
        - Large lists (≥128 items) → replaced with placeholders
        - Large strings (≥10K chars) → truncated with suffix
        - Total response limited to 8000 tokens (results dropped if needed)
        
        **Property Selection:**
        - Default (no return_properties): Returns ALL properties (sanitized)
        - With return_properties: Returns ONLY specified properties
        - Example: return_properties="pageNumber,id" → returns only these two
        - Check get_searchable_content for property warnings to avoid large fields
        
        **Performance Optimization:**
        Internally fetches max(top_k × 2, 100) results to avoid local maximum problems in kANN algorithms.
        """
        logger.info(f"Running `vector_search` with query='{text_query}', index='{vector_index}', top_k={top_k}, return_properties={return_properties}")

        try:
            # Get the embedding property name from the vector index
            index_info_query = """
            SHOW INDEXES
            YIELD name, type, properties
            WHERE name = $index_name AND type = 'VECTOR'
            RETURN properties
            """
            
            index_info = await neo4j_driver.execute_query(
                index_info_query,
                parameters_={"index_name": vector_index},
                routing_control=RoutingControl.READ,
                database_=database,
                result_transformer_=lambda r: r.data(),
            )
            
            if not index_info:
                raise ToolError(f"Vector index '{vector_index}' not found. Use get_searchable_content to see available indexes.")
            
            embedding_property = index_info[0]["properties"][0] if index_info[0]["properties"] else None
            logger.debug(f"Embedding property for index '{vector_index}': {embedding_property}")

            # Generate embedding using LiteLLM (supports multiple providers)
            logger.debug(f"Generating embedding with model: {embedding_model}")
            embedding_response = litellm.embedding(
                model=embedding_model,
                input=[text_query]
            )
            query_embedding = embedding_response.data[0]["embedding"]

            # Fetch more results to avoid local maximum
            fetch_k = max(top_k * 2, 100)
            logger.debug(f"Fetching {fetch_k} results from vector index")

            # Parse return_properties if provided (comma-separated string)
            property_list = None
            if return_properties:
                property_list = [prop.strip() for prop in return_properties.split(",")]
                logger.debug(f"Parsed return_properties: {property_list}")

            # Build RETURN clause based on return_properties
            if property_list:
                props_return = ", ".join([f"node.{prop} as {prop}" for prop in property_list])
                search_query = f"""
                CALL db.index.vector.queryNodes($index_name, $fetch_k, $query_vector)
                YIELD node, score
                RETURN elementId(node) as nodeId, labels(node) as labels, {props_return}, score
                ORDER BY score DESC
                LIMIT $top_k
                """
            else:
                search_query = """
                CALL db.index.vector.queryNodes($index_name, $fetch_k, $query_vector)
                YIELD node, score
                RETURN elementId(node) as nodeId, labels(node) as labels, properties(node) as properties, score
                ORDER BY score DESC
                LIMIT $top_k
                """

            results = await neo4j_driver.execute_query(
                search_query,
                parameters_={
                    "index_name": vector_index,
                    "fetch_k": fetch_k,
                    "top_k": top_k,
                    "query_vector": query_embedding
                },
                routing_control=RoutingControl.READ,
                database_=database,
                result_transformer_=lambda r: r.data(),
            )

            logger.debug(f"Vector search returned {len(results)} results")

            # Auto-exclude the embedding property (vector_search only)
            if embedding_property:
                for result in results:
                    if 'properties' in result and embedding_property in result['properties']:
                        del result['properties'][embedding_property]
                        logger.debug(f"Auto-excluded embedding property: {embedding_property}")

            # Layer 3: Sanitize large lists and strings (images, text, etc.)
            for result in results:
                if 'properties' in result:
                    result['properties'] = _value_sanitize(result['properties'], MAX_LIST_SIZE, MAX_STRING_SIZE)

            # Layer 4: Truncate results to stay under token limit
            original_count = len(results)
            results, was_truncated = _truncate_results_to_token_limit(results, RESPONSE_TOKEN_LIMIT)

            formatted_results = {
                "query": text_query,
                "index": vector_index,
                "top_k": top_k,
                "results": results
            }

            if was_truncated:
                formatted_results["warning"] = f"Results truncated from {original_count} to {len(results)} items (token limit: {RESPONSE_TOKEN_LIMIT})"

            result_json = json.dumps(formatted_results, default=str, indent=2)
            return ToolResult(content=[TextContent(type="text", text=result_json)])

        except Neo4jError as e:
            logger.error(f"Neo4j Error during vector search: {e}")
            raise ToolError(f"Neo4j Error: {e}")

        except Exception as e:
            logger.error(f"Error during vector search: {e}")
            raise ToolError(f"Error: {e}")

    @mcp.tool(
        name=namespace_prefix + "fulltext_search",
        annotations=ToolAnnotations(
            title="Fulltext Search",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def fulltext_search(
        text_query: str = Field(..., description="The text query to search for. Supports Lucene query syntax (AND, OR, wildcards, fuzzy, etc.)."),
        fulltext_index: str = Field(..., description="The name of the fulltext index to search. Use get_searchable_content to see available indexes."),
        top_k: int = Field(default=5, description="The number of most relevant results to return."),
        return_properties: Optional[str] = Field(
            None,
            description='Optional: Comma-separated list of properties to return (e.g., "pageNumber,id"). If not specified, returns all properties with automatic sanitization (large values are truncated).'
        ),
    ) -> list[ToolResult]:
        """
        Performs fulltext search on a Neo4j fulltext index using Lucene query syntax.
        
        **Lucene Syntax Supported:**
        - Boolean: "legal AND compliance", "privacy OR security"
        - Wildcards: "compli*", "te?t"
        - Fuzzy: "complience~"
        - Phrases: "\"exact phrase\""
        
        **Automatic Sanitization (always applied):**
        - Large lists (≥128 items) → replaced with placeholders
        - Large strings (≥10K chars) → truncated with suffix
        - Total response limited to 8000 tokens (results dropped if needed)
        
        **Property Selection:**
        - Default (no return_properties): Returns ALL properties (sanitized)
        - With return_properties: Returns ONLY specified properties
        - Example: return_properties="pageNumber,id" → returns only these two
        - Check get_searchable_content for property warnings to avoid large fields
        
        Returns node/relationship IDs, labels/types, properties (sanitized), and relevance scores.
        """
        logger.info(f"Running `fulltext_search` with query='{text_query}', index='{fulltext_index}', top_k={top_k}, return_properties={return_properties}")

        try:
            # Get index entity type
            index_info_query = """
            SHOW INDEXES
            YIELD name, type, entityType
            WHERE name = $index_name AND type = 'FULLTEXT'
            RETURN entityType
            """
            
            index_info = await neo4j_driver.execute_query(
                index_info_query,
                parameters_={"index_name": fulltext_index},
                routing_control=RoutingControl.READ,
                database_=database,
                result_transformer_=lambda r: r.data(),
            )

            if not index_info:
                raise ToolError(f"Fulltext index '{fulltext_index}' not found. Use get_searchable_content to see available indexes.")

            entity_type = index_info[0]["entityType"]

            # Parse return_properties if provided (comma-separated string)
            property_list = None
            if return_properties:
                property_list = [prop.strip() for prop in return_properties.split(",")]
                logger.debug(f"Parsed return_properties: {property_list}")

            # Build RETURN clause based on return_properties and entity_type
            if entity_type == "NODE":
                if property_list:
                    props_return = ", ".join([f"node.{prop} as {prop}" for prop in property_list])
                    search_query = f"""
                    CALL db.index.fulltext.queryNodes($index_name, $query)
                    YIELD node, score
                    RETURN elementId(node) as nodeId, labels(node) as labels, {props_return}, score
                    ORDER BY score DESC
                    LIMIT $top_k
                    """
                else:
                    search_query = """
                    CALL db.index.fulltext.queryNodes($index_name, $query)
                    YIELD node, score
                    RETURN elementId(node) as nodeId, labels(node) as labels, properties(node) as properties, score
                    ORDER BY score DESC
                    LIMIT $top_k
                    """
            else:
                if property_list:
                    props_return = ", ".join([f"relationship.{prop} as {prop}" for prop in property_list])
                    search_query = f"""
                    CALL db.index.fulltext.queryRelationships($index_name, $query)
                    YIELD relationship, score
                    RETURN elementId(relationship) as relationshipId, type(relationship) as type, {props_return}, score
                    ORDER BY score DESC
                    LIMIT $top_k
                    """
                else:
                    search_query = """
                    CALL db.index.fulltext.queryRelationships($index_name, $query)
                    YIELD relationship, score
                    RETURN elementId(relationship) as relationshipId, type(relationship) as type, properties(relationship) as properties, score
                    ORDER BY score DESC
                    LIMIT $top_k
                    """

            results = await neo4j_driver.execute_query(
                search_query,
                parameters_={
                    "index_name": fulltext_index,
                    "query": text_query,
                    "top_k": top_k
                },
                routing_control=RoutingControl.READ,
                database_=database,
                result_transformer_=lambda r: r.data(),
            )

            logger.debug(f"Fulltext search returned {len(results)} results")

            # Layer 3: Sanitize large lists and strings (embeddings, images, text, etc.)
            for result in results:
                if 'properties' in result:
                    result['properties'] = _value_sanitize(result['properties'], MAX_LIST_SIZE, MAX_STRING_SIZE)

            # Layer 4: Truncate results to stay under token limit
            original_count = len(results)
            results, was_truncated = _truncate_results_to_token_limit(results, RESPONSE_TOKEN_LIMIT)

            formatted_results = {
                "query": text_query,
                "index": fulltext_index,
                "entity_type": entity_type,
                "top_k": top_k,
                "results": results
            }

            if was_truncated:
                formatted_results["warning"] = f"Results truncated from {original_count} to {len(results)} items (token limit: {RESPONSE_TOKEN_LIMIT})"

            result_json = json.dumps(formatted_results, default=str, indent=2)
            return ToolResult(content=[TextContent(type="text", text=result_json)])

        except Neo4jError as e:
            logger.error(f"Neo4j Error during fulltext search: {e}")
            raise ToolError(f"Neo4j Error: {e}")

        except Exception as e:
            logger.error(f"Error during fulltext search: {e}")
            raise ToolError(f"Error: {e}")

    # ========================================
    # NODE RETRIEVAL TOOL
    # ========================================

    @mcp.tool(
        name=namespace_prefix + "get_node_by_id",
        annotations=ToolAnnotations(
            title="Get Node By ID",
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        ),
    )
    async def get_node_by_id(
        node_id: str = Field(..., description="The Neo4j node element ID to retrieve (from search results)."),
        return_properties: Optional[str] = Field(
            None,
            description='Optional: Comma-separated list of properties to return (e.g., "pageNumber,id"). If not specified, returns all properties with automatic sanitization (large values are truncated).'
        ),
    ) -> list[ToolResult]:
        """
        Retrieves a specific node by its Neo4j element ID.
        
        This tool fetches a single node's properties (no relationships - pure vector DB style).
        Useful for getting full details of nodes found via search.
        
        **Automatic Sanitization (always applied):**
        - Large lists (≥128 items) → replaced with placeholders
        - Large strings (≥10K chars) → truncated with suffix
        - Total response limited to 8000 tokens
        
        **Property Selection:**
        - Default (no return_properties): Returns ALL properties (sanitized)
        - With return_properties: Returns ONLY specified properties
        - Example: return_properties="pageNumber,id" → returns only these two
        - Check get_searchable_content for property warnings to avoid large fields
        
        Returns node ID, labels, and properties (no relationships).
        """
        logger.info(f"Running `get_node_by_id` with node_id='{node_id}', return_properties={return_properties}")

        try:
            # Parse return_properties if provided (comma-separated string)
            property_list = None
            if return_properties:
                property_list = [prop.strip() for prop in return_properties.split(",")]
                logger.debug(f"Parsed return_properties: {property_list}")

            # Build RETURN clause based on return_properties
            if property_list:
                # Return specific properties only
                return_clause = ", ".join([f"n.{prop} AS {prop}" for prop in property_list])
                query = f"""
                MATCH (n)
                WHERE elementId(n) = $node_id
                RETURN elementId(n) AS elementId, labels(n) AS labels, {return_clause}
                """
            else:
                # Return all properties (will be sanitized)
                query = """
                MATCH (n)
                WHERE elementId(n) = $node_id
                RETURN elementId(n) AS elementId, labels(n) AS labels, properties(n) AS properties
                """

            results = await neo4j_driver.execute_query(
                query,
                parameters_={"node_id": node_id},
                routing_control=RoutingControl.READ,
                database_=database,
                result_transformer_=lambda r: r.data(),
            )

            if not results:
                raise ToolError(f"Node with ID '{node_id}' not found")

            logger.debug(f"Found node: {results[0].get('labels')}")

            # Layer 3: Sanitize large lists and strings
            sanitized_results = [_value_sanitize(el, MAX_LIST_SIZE, MAX_STRING_SIZE) for el in results]

            # Layer 4: Truncate results to stay under token limit
            original_count = len(sanitized_results)
            sanitized_results, was_truncated = _truncate_results_to_token_limit(
                sanitized_results, RESPONSE_TOKEN_LIMIT
            )

            if was_truncated:
                logger.warning(
                    f"Node results truncated from {original_count} to {len(sanitized_results)} rows"
                )

            results_json_str = json.dumps(sanitized_results, default=str)
            return ToolResult(content=[TextContent(type="text", text=results_json_str)])

        except Neo4jError as e:
            logger.error(f"Neo4j Error getting node by ID: {e}")
            raise ToolError(f"Neo4j Error: {e}")

        except Exception as e:
            logger.error(f"Error getting node by ID: {e}")
            raise ToolError(f"Error: {e}")

    return mcp


async def main(
    db_url: str,
    username: str,
    password: str,
    database: str,
    embedding_model: str = "text-embedding-3-small",
    transport: Literal["stdio", "sse", "http"] = "stdio",
    namespace: str = "",
    host: str = "127.0.0.1",
    port: int = 8000,
    path: str = "/mcp/",
    read_timeout: int = 30,
    schema_sample_size: int = 1000,
) -> None:
    """Main entry point for the Neo4j Vector Database MCP Server."""
    logger.info("Starting Neo4j Vector Database MCP Server")
    logger.info(f"Using embedding model: {embedding_model}")

    neo4j_driver = AsyncGraphDatabase.driver(db_url, auth=(username, password))

    mcp = create_mcp_server(
        neo4j_driver=neo4j_driver,
        embedding_model=embedding_model,
        database=database,
        namespace=namespace,
        read_timeout=read_timeout,
        config_sample_size=schema_sample_size,
    )

    match transport:
        case "http":
            logger.info(f"Running Neo4j Vector Database MCP Server with HTTP transport on {host}:{port}...")
            await mcp.run_http_async(host=host, port=port, path=path)
        case "stdio":
            logger.info("Running Neo4j Vector Database MCP Server with stdio transport...")
            await mcp.run_stdio_async()
        case "sse":
            logger.info(f"Running Neo4j Vector Database MCP Server with SSE transport on {host}:{port}...")
            await mcp.run_http_async(host=host, port=port, path=path, transport="sse")
        case _:
            logger.error(f"Invalid transport: {transport}")
            raise ValueError(f"Invalid transport: {transport} | Must be 'stdio', 'sse', or 'http'")


if __name__ == "__main__":
    # This file should not be run directly. Use the CLI: mcp-neo4j-vectordb
    # Or import and call: from mcp_neo4j_vectordb import main
    raise RuntimeError(
        "This module should not be run directly. "
        "Use the CLI command: mcp-neo4j-vectordb --help"
    )

