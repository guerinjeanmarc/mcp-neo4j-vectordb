# Changelog

All notable changes to the Neo4j Vector Database MCP Server will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2025-12-01

### Added
- Initial release of Neo4j Vector Database MCP Server
- **Pure vector database mimic** - no graph traversal or relationships exposed
- Forked from `mcp-neo4j-graphrag` v0.3.0 to create comparison baseline

### Tools
- `get_searchable_content`: Lists vector/fulltext indexes with ONLY indexed node schemas (no relationships, no non-indexed nodes)
- `vector_search`: Vector similarity search with OpenAI embeddings
- `fulltext_search`: Keyword search with Lucene syntax support
- `get_node_by_id`: Retrieve specific node properties (no relationships)

### Features
- Production-ready output control:
  - **Layer 3**: Size-based filtering (lists ≥128 items, strings ≥10K chars)
  - **Layer 4**: Token-aware truncation (8000 token limit using `tiktoken`)
- Automatic embedding property exclusion in `vector_search`
- Property size warnings in `get_searchable_content`
- Flexible property return: all properties (sanitized) by default, or comma-separated list
- Performance optimization: Fetches `max(top_k × 2, 100)` results for vector search (avoids local maximum)

### Removed (compared to mcp-neo4j-graphrag)
- ❌ `read_neo4j_cypher` - No Cypher query access
- ❌ `search_cypher_query` - No search-augmented Cypher
- ❌ Relationships - Hidden from all outputs
- ❌ Non-indexed nodes - Not shown in `get_searchable_content`

### Purpose
- Research/comparison tool to test **Vector RAG vs GraphRAG**
- Isolates the value-add of graph structure (relationships, traversal)
- Provides fair comparison baseline using the same database and search indexes

### Technical Details
- Python 3.10+
- Dependencies: `fastmcp`, `neo4j`, `openai`, `tiktoken`, `pydantic`
- Supports `stdio`, `SSE`, and `HTTP` transports
- Same sanitization and safety features as `mcp-neo4j-graphrag`

---

## Comparison with mcp-neo4j-graphrag

| Aspect | mcp-neo4j-vectordb (v0.1.0) | mcp-neo4j-graphrag (v0.3.0) |
|--------|----------------------------|----------------------------|
| Vector Search | ✅ | ✅ |
| Fulltext Search | ✅ | ✅ |
| Relationships | ❌ Hidden | ✅ Exposed |
| Cypher Queries | ❌ No | ✅ Yes |
| Graph Traversal | ❌ No | ✅ Yes |
| Production Features | ✅ Same (Layers 3 & 4) | ✅ Same (Layers 3 & 4) |
| Use Case | Pure Vector RAG | Graph-augmented RAG |

---

**For full GraphRAG capabilities, see [mcp-neo4j-graphrag](https://github.com/guerinjeanmarc/mcp-neo4j-graphrag)**

