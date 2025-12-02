# Neo4j Vector Database MCP Server

A Model Context Protocol (MCP) server that **mimics a pure vector database** using Neo4j as the backend. 

This server provides **vector and fulltext search WITHOUT graph traversal or relationships**, designed specifically to compare the effectiveness of pure Vector RAG vs Graph-augmented RAG ([GraphRAG](https://github.com/guerinjeanmarc/mcp-neo4j-graphrag)).

---

## 🚀 **Quick Start**

### **1. Install Dependencies**
```bash
cd mcp-neo4j-vectordb
uv sync
```

### **2. Configure in Claude Desktop or Cursor**
Add to your `mcp.json` or `.cursor/mcp.json`:
```json
{
  "mcpServers": {
    "neo4j-vectordb": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/mcp-neo4j-vectordb",
        "run",
        "mcp-neo4j-vectordb"
      ],
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "your-password",
        "NEO4J_DATABASE": "neo4j",
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

### **3. Test the Server**
Restart your IDE/Claude Desktop, then:
- "List available search indexes using get_searchable_content"
- "Search for 'cancer treatment' using vector_search with top_k=5"

### **4. Compare with GraphRAG**
Enable [mcp-neo4j-graphrag](https://github.com/guerinjeanmarc/mcp-neo4j-graphrag) alongside this server and ask the same questions to see the difference!

---

## 🎯 **Purpose**

This server is a **research/comparison tool** that allows you to:

1. **Test pure Vector RAG**: Use Neo4j strictly as a vector database (no graph features)
2. **Compare with GraphRAG**: Run the same queries with both servers to see when graph structure adds value
3. **Isolate search quality**: Understand whether answers improve due to better search or graph context

**Key Design Decision:** This server intentionally **excludes** graph features (relationships, traversal, Cypher queries) to provide a fair comparison baseline.

---

## 🆚 **Vector DB vs GraphRAG Comparison**

| Feature | **mcp-neo4j-vectordb** (this server) | **mcp-neo4j-graphrag** |
|---------|--------------------------------------|------------------------|
| **Vector Search** | ✅ Yes | ✅ Yes |
| **Fulltext Search** | ✅ Yes | ✅ Yes |
| **Relationships** | ❌ No (hidden) | ✅ Yes (full access) |
| **Cypher Queries** | ❌ No | ✅ Yes (`read_neo4j_cypher`, `search_cypher_query`) |
| **Graph Traversal** | ❌ No | ✅ Yes |
| **Post-Filtering** | ⚠️ Limited (property-based only) | ✅ Full (Cypher WHERE clauses) |
| **Graph Aggregation** | ❌ No | ✅ Yes (COUNT, GROUP BY in Cypher) |
| **Use Case** | Pure vector RAG | Graph-augmented RAG |

---

## 📊 **How to Compare Vector RAG vs GraphRAG**

### **Example Database:**

For testing and comparison, you can use the **pharmaceutical pipeline knowledge graph** from:
- **Repository**: [pharma-pipeline-KG-creation](https://github.com/neo4j-field/pharma-pipeline-KG-creation)
- **Database dump**: [Download here](https://drive.google.com/file/d/1Nk1SNk5Rsq6l9J2wiXf43i1OkRMobeAz/view?usp=sharing) (560MB)

This graph contains molecules, targets, pathways, companies, and disease relationships - perfect for comparing vector search vs graph traversal.

### **Methodology:**

1. **Same Database**: Both servers connect to the same Neo4j database
2. **Same Search**: Both use identical vector/fulltext indexes
3. **Same Sanitization**: Both apply the same output control (token limits, size filtering)
4. **Isolated Variable**: The ONLY difference is graph context (relationships, traversal)
5. **Recommended LLM**: Claude 3.5 Haiku in Claude Desktop for consistent comparison

### **Testing Approach:**

**Option 1: Sequential Testing (Recommended)**
```bash
# Test 1: Vector DB only
# - Enable mcp-neo4j-vectordb in Cursor/Claude Desktop
# - Ask a complex question
# - Note the answer quality and tool usage

# Test 2: GraphRAG
# - Disable mcp-neo4j-vectordb
# - Enable mcp-neo4j-graphrag
# - Ask the SAME question
# - Compare answer quality and tool usage

# Test 3: Agent Self-Assessment
# - Enable BOTH servers (different namespaces)
# - Ask the agent: "Which approach gave you better context to answer the question - vector search or graph search? Explain why."
```

**Option 2: A/B Testing**
```bash
# For multiple questions, alternate which server is used first
# Question 1: vectordb first, then graphrag
# Question 2: graphrag first, then vectordb
# This controls for recency bias
```

### **Example Question for Comparison:**

Using the pharmaceutical pipeline database above, try this complex question with **both servers**:

> **"What are all the molecules targeting the EGFR pathway, what are their mechanisms of action, which companies are developing them, and what diseases are they being developed for?"**

**Why this question works well:**
- **Vector DB**: Can find molecules semantically related to "EGFR pathway" but struggles to connect mechanisms, companies, and diseases
- **GraphRAG**: Can traverse relationships (Molecule→Target→Pathway, Molecule→Company, Molecule→Disease) to provide comprehensive answers

**More Example Questions:**

**Good for Vector DB** (semantic search sufficient):
- "What is apoptosis?"
- "Find research about CRISPR gene editing"

**Good for GraphRAG** (requires context/relationships):
- "Which drugs interact with medications for diabetes?"
- "What are the downstream effects of inhibiting protein X?"
- "Find all authors who co-authored with [specific researcher]"

---

## 🛠️ **Available Tools**

### **1. `get_searchable_content`**

Returns **ONLY indexed nodes** (vector or fulltext indexes) with property schemas and size warnings.

**Key Differences from GraphRAG version:**
- ❌ NO relationships shown
- ❌ NO non-indexed node types
- ✅ ONLY nodes with vector/fulltext indexes

**Example Output:**
```json
{
  "indexes": {
    "vector": [{
      "name": "page_text_embeddings",
      "entityType": "NODE",
      "labelsOrTypes": ["Page"],
      "properties": ["embedding"]
    }],
    "fulltext": []
  },
  "schema": {
    "Page": {
      "type": "node",
      "count": 102,
      "properties": {
        "id": {"type": "STRING"},
        "pageNumber": {"type": "INTEGER"},
        "extractedText": {"type": "STRING"},
        "extractedImage": {
          "type": "STRING", 
          "warning": "very large (avg ~308KB)"
        },
        "embedding": {
          "type": "LIST",
          "warning": "large list (avg ~1536 items)"
        }
      }
    }
  }
}
```

---

### **2. `vector_search`**

Performs vector similarity search using OpenAI embeddings.

**Parameters:**
- `text_query` (required): Text to search for (auto-embedded)
- `vector_index` (required): Name of Neo4j vector index
- `top_k` (default: 5): Number of results
- `return_properties` (optional): Comma-separated properties (e.g., `"pageNumber,id"`)

**Returns:** Node IDs, labels, properties (sanitized), and similarity scores.

**Example:**
```python
vector_search(
    text_query="cancer treatment options",
    vector_index="page_text_embeddings",
    top_k=10,
    return_properties="pageNumber,extractedText"
)
```

**Automatic Features:**
- Embedding property auto-excluded (saves tokens)
- Large lists (≥128 items) → placeholder
- Large strings (≥10K chars) → truncated
- Total response limited to 8000 tokens

---

### **3. `fulltext_search`**

Performs keyword search using Neo4j fulltext indexes (Lucene syntax).

**Parameters:**
- `text_query` (required): Search query (supports Lucene operators)
- `fulltext_index` (required): Name of Neo4j fulltext index
- `top_k` (default: 5): Number of results
- `return_properties` (optional): Comma-separated properties

**Lucene Syntax Supported:**
- Boolean: `"cancer AND treatment"`, `"therapy OR medication"`
- Wildcards: `"treat*"`, `"canc?r"`
- Fuzzy: `"treatmnt~"` (finds "treatment")
- Phrases: `"\"exact phrase\""`

**Returns:** Node/relationship IDs, labels/types, properties (sanitized), and relevance scores.

---

### **4. `get_node_by_id`**

Retrieves a specific node by its Neo4j element ID (from search results).

**Parameters:**
- `node_id` (required): Neo4j element ID (e.g., from `vector_search`)
- `return_properties` (optional): Comma-separated properties

**Returns:** Node properties **ONLY** (no relationships - pure vector DB style).

**Example:**
```python
# Step 1: Search
results = vector_search(
    text_query="machine learning",
    vector_index="page_text_embeddings",
    top_k=5
)

# Step 2: Get full node details
node = get_node_by_id(
    node_id=results[0]['elementId'],
    return_properties="id,extractedText,pageNumber"
)
```

---

## ⚙️ **Installation**

### **Prerequisites:**
- Python 3.10+
- Neo4j 5.0+ with vector indexes
- OpenAI API key

### **Install:**
```bash
cd mcp-neo4j-vectordb
uv pip install -e .
```

---

## 🚀 **Usage**

### **Cursor IDE Configuration:**

```json
{
  "mcpServers": {
    "neo4j-vectordb": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/mcp-neo4j-vectordb",
        "run",
        "mcp-neo4j-vectordb"
      ],
      "env": {
        "NEO4J_URI": "neo4j+s://your-instance.databases.neo4j.io",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "your-password",
        "NEO4J_DATABASE": "neo4j",
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

### **Claude Desktop Configuration:**

```json
{
  "mcpServers": {
    "neo4j-vectordb": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/mcp-neo4j-vectordb",
        "run",
        "mcp-neo4j-vectordb",
        "--transport", "stdio"
      ],
      "env": {
        "NEO4J_URI": "neo4j+s://your-instance.databases.neo4j.io",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "your-password",
        "NEO4J_DATABASE": "neo4j",
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

### **Testing with Both Servers:**

To compare vector DB vs GraphRAG, configure **both** servers with different namespaces:

```json
{
  "mcpServers": {
    "neo4j-vectordb": {
      "command": "uv",
      "args": [
        "--directory", "/path/to/mcp-neo4j-vectordb",
        "run", "mcp-neo4j-vectordb",
        "--namespace", "vectordb"
      ],
      "env": { "..." }
    },
    "neo4j-graphrag": {
      "command": "uv",
      "args": [
        "--directory", "/path/to/mcp-neo4j-graphrag",
        "run", "mcp-neo4j-graphrag",
        "--namespace", "graphrag"
      ],
      "env": { "..." }
    }
  }
}
```

Then ask: *"Try answering my question using vectordb tools, then graphrag tools. Which gave better context?"*

---

## 🔒 **Production Features**

This server inherits production-proofing from `mcp-neo4j-graphrag`:

### **Layer 3: Size-Based Filtering**
- Lists with ≥128 items → replaced with `"<list with X items (truncated)>"`
- Strings with ≥10K chars → truncated with `"...<truncated at 10000 chars>"`

### **Layer 4: Token-Aware Truncation**
- Total response limited to 8000 tokens (using `tiktoken`)
- Results dropped from the end if limit exceeded
- Warning logged when truncation occurs

**Hardcoded Values:**
```python
MAX_LIST_SIZE = 128
MAX_STRING_SIZE = 10000
RESPONSE_TOKEN_LIMIT = 8000
```

---

## 🔗 **Related Projects**

- **[mcp-neo4j-graphrag](https://github.com/guerinjeanmarc/mcp-neo4j-graphrag)** - Full GraphRAG server with graph traversal
- **[pharma-pipeline-KG-creation](https://github.com/neo4j-field/pharma-pipeline-KG-creation)** - Example pharmaceutical knowledge graph for testing
- **[mcp-neo4j-cypher](https://github.com/neo4j-contrib/mcp-neo4j/tree/main/servers/mcp-neo4j-cypher)** - Official Neo4j Cypher server
- **[neo4j-graphrag-python](https://github.com/neo4j/neo4j-graphrag-python)** - Neo4j GraphRAG library

---

## 📝 **Configuration Options**

```bash
mcp-neo4j-vectordb --help
```

**Key Options:**
- `--db-url`: Neo4j connection URL (default: from env `NEO4J_URI`)
- `--username`: Neo4j username (default: from env `NEO4J_USERNAME`)
- `--password`: Neo4j password (default: from env `NEO4J_PASSWORD`)
- `--database`: Neo4j database name (default: `neo4j`)
- `--openai-api-key`: OpenAI API key (default: from env `OPENAI_API_KEY`)
- `--embedding-model`: OpenAI embedding model (default: `text-embedding-3-small`)
- `--namespace`: Tool namespace prefix (default: none)
- `--transport`: Transport type: `stdio`, `sse`, `http` (default: `stdio`)
- `--read-timeout`: Query timeout in seconds (default: 30)
- `--schema-sample-size`: Property sampling size (default: 1000)

---

## 🐛 **Troubleshooting**

### **"Node not found" errors**
- Check that the node ID is valid (use `vector_search` or `fulltext_search` to get IDs)
- Verify the node hasn't been deleted since the search

### **"Index not found" errors**
- Run `get_searchable_content` to see available indexes
- Check index names match exactly (case-sensitive)

### **Large properties causing truncation**
- Use `return_properties` to exclude large fields
- Check `get_searchable_content` for property size warnings
- Example: Avoid `extractedImage` (300KB) if you only need `pageNumber`

### **Slow queries**
- Reduce `top_k` for vector/fulltext search
- Use `return_properties` to fetch fewer fields
- Increase `--read-timeout` if needed

---

## 📄 **License**

MIT License - see [LICENSE](LICENSE)

---

## 🤝 **Contributing**

This is a research/comparison tool. If you find bugs or have suggestions:

1. Compare behavior with `mcp-neo4j-graphrag` (they should be identical except for graph features)
2. Check if the issue affects search accuracy or just output formatting
3. Open an issue with example queries and expected vs actual behavior

---

## 🙏 **Acknowledgments**

- Built on top of the [FastMCP](https://github.com/jlowin/fastmcp) framework
- Inspired by the [mcp-neo4j](https://github.com/neo4j-contrib/mcp-neo4j) project
- Sanitization approach based on [Production-Proofing Your Neo4j Cypher MCP Server](https://neo4j.com/blog/developer/production-proofing-cypher-mcp-server/)

---

**For full GraphRAG capabilities (relationships, traversal, Cypher), see [mcp-neo4j-graphrag](https://github.com/guerinjeanmarc/mcp-neo4j-graphrag).**
