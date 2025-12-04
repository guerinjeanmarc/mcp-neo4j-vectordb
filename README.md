# Neo4j VectorDB MCP Server

An MCP server that uses Neo4j as a **pure vector database** — no graph traversal, no relationships.

## Purpose

This server is designed to **compare Vector RAG vs Graph RAG**. It provides the same search capabilities as `mcp-neo4j-graphrag` but intentionally hides graph features, allowing you to measure the value that graph context adds to LLM responses.

| Feature | `mcp-neo4j-vectordb` (This) | `mcp-neo4j-graphrag` |
|---------|----------------------------|----------------------|
| Vector search | ✅ | ✅ |
| Fulltext search | ✅ | ✅ |
| Graph traversal | ❌ | ✅ |
| Cypher queries | ❌ | ✅ |
| Relationships | ❌ Hidden | ✅ Visible |

## Installation

### Step 1: Download the Repository

```bash
git clone https://github.com/guerinjeanmarc/mcp-neo4j-vectordb.git
cd mcp-neo4j-vectordb
```

### Step 2: Configure Claude Desktop

Edit the configuration file:
- **macOS/Linux:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

Add this server configuration (update the path to where you cloned the repo):

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
        "NEO4J_URI": "neo4j+s://demo.neo4jlabs.com",
        "NEO4J_USERNAME": "recommendations",
        "NEO4J_PASSWORD": "recommendations",
        "NEO4J_DATABASE": "recommendations",
        "OPENAI_API_KEY": "sk-...",
        "EMBEDDING_MODEL": "text-embedding-ada-002"
      }
    }
  }
}
```

### Step 3: Reload Configuration

Quit and restart Claude Desktop to load the new configuration.

## Tools

### `get_searchable_content`

Discover available indexes and searchable node properties (no relationships shown).

💡 The agent should automatically call this tool first to understand what can be searched.

**Example prompt:**
> "What can I search in this database?"

### `vector_search`

Semantic similarity search.

**Example prompt:**
> "Find movies similar to 'a hero's journey in space' using moviePlotsEmbedding"

### `fulltext_search`

Keyword search with Lucene syntax.

**Example prompt:**
> "Search for 'Tom Hanks' in the personFulltext index"

### `get_node_by_id`

Retrieve a node's properties by ID (no relationships).

**Example prompt:**
> "Get the full details of the first movie from my search results"

---

## Comparison Workflow: Vector RAG vs Graph RAG

Use both servers to compare how graph context improves LLM responses.

### Prerequisites

Download both repositories:

```bash
git clone https://github.com/guerinjeanmarc/mcp-neo4j-vectordb.git
git clone https://github.com/guerinjeanmarc/mcp-neo4j-graphrag.git
```

### Step 1: Configure Both Servers in Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

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
        "NEO4J_URI": "neo4j+s://demo.neo4jlabs.com",
        "NEO4J_USERNAME": "recommendations",
        "NEO4J_PASSWORD": "recommendations",
        "NEO4J_DATABASE": "recommendations",
        "OPENAI_API_KEY": "sk-...",
        "EMBEDDING_MODEL": "text-embedding-ada-002"
      }
    },
    "neo4j-graphrag": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/mcp-neo4j-graphrag",
        "run",
        "mcp-neo4j-graphrag"
      ],
      "env": {
        "NEO4J_URI": "neo4j+s://demo.neo4jlabs.com",
        "NEO4J_USERNAME": "recommendations",
        "NEO4J_PASSWORD": "recommendations",
        "NEO4J_DATABASE": "recommendations",
        "OPENAI_API_KEY": "sk-...",
        "EMBEDDING_MODEL": "text-embedding-ada-002"
      }
    }
  }
}
```

### Step 2: Test with VectorDB Only

1. In Claude Desktop, click the **"Search and Tools"** button in the conversation interface
2. **Disable** `neo4j-graphrag` tools (keep only `neo4j-vectordb` active)
3. Ask the test question:

> "Find movies about artificial intelligence, and tell me which directors have made multiple AI films and what other genres they typically work in."

4. Note Claude's answer and what tools it used

### Step 3: Test with GraphRAG Only

1. Click **"Search and Tools"** again
2. **Disable** `neo4j-vectordb` tools and **enable** `neo4j-graphrag` tools
3. Ask:

> "You have now access to the Neo4j tools, please answer the same question."

4. Note Claude's answer and what tools it used

### Step 4: Compare Results

Ask Claude to compare:

> "Compare your two previous answers. What approach gave you better context to answer the question? Please explain what technology is better to answer the question: VectorDB or Neo4j?"

### Example Questions for Comparison

| Question | Vector RAG | Graph RAG |
|----------|------------|-----------|
| "What is The Matrix about?" | ✅ Good | ✅ Good |
| "Find sci-fi movies" | ✅ Good | ✅ Good |
| "Which actors worked with both Spielberg and Nolan?" | ⚠️ Limited | ✅ Excellent |
| "What genres does Tom Hanks typically act in?" | ⚠️ Limited | ✅ Excellent |
| "Find movies similar to Inception and show their directors' other work" | ⚠️ Limited | ✅ Excellent |

---

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NEO4J_URI` | Yes | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USERNAME` | Yes | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | Yes | `password` | Neo4j password |
| `NEO4J_DATABASE` | No | `neo4j` | Database name |
| `EMBEDDING_MODEL` | No | `text-embedding-3-small` | Embedding model |

### Embedding Providers

Supports all [LiteLLM embedding providers](https://docs.litellm.ai/docs/embedding/supported_embedding):
- OpenAI: `text-embedding-ada-002`, `text-embedding-3-small`
- Azure: `azure/deployment-name`
- Bedrock: `bedrock/amazon.titan-embed-text-v1`
- Cohere: `cohere/embed-english-v3.0`
- Ollama: `ollama/nomic-embed-text`

---

## Related

- [mcp-neo4j-graphrag](https://github.com/guerinjeanmarc/mcp-neo4j-graphrag) — Full GraphRAG with graph traversal
- [Neo4j MCP Documentation](https://neo4j.com/developer/genai-ecosystem/model-context-protocol-mcp/)
- [Official Neo4j MCP Server](https://github.com/neo4j/mcp)

## License

MIT License
