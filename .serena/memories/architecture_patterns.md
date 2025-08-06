# Architecture and Design Patterns

## Overall Architecture

### MCP Server Design
- **FastMCP Framework**: Uses FastMCP for MCP protocol implementation
- **Single File Architecture**: All server logic in `server.py` for simplicity
- **Subprocess Management**: MkDocs dev server runs as managed subprocess
- **Thread-based Server Management**: Separate thread for MkDocs server lifecycle

### Key Design Patterns

#### 1. Graceful Degradation Pattern
- Vector search falls back to keyword-only if dependencies unavailable
- MkDocs server failures don't crash MCP server
- Missing configurations handled with sensible defaults

#### 2. Resource Management Pattern
- `atexit` handlers for cleanup
- Signal handlers (SIGINT, SIGTERM) for graceful shutdown
- Automatic cleanup of temporary index directories
- Subprocess termination on exit

#### 3. Singleton Pattern
- Global variables for shared state:
  - `_mkdocs_process`: Subprocess instance
  - `_mkdocs_thread`: Management thread
  - `_mkdocs_config`: Configuration cache
  - `_searcher`: DocsSearcher instance

#### 4. Decorator Pattern
- `@mcp.tool` decorators for MCP tool registration
- Separation of implementation (_impl functions) from decorated functions

## Core Components

### DocsSearcher Class
**Responsibilities:**
- Document indexing with Whoosh
- Vector embeddings with Sentence Transformers
- Milvus vector database management
- Hybrid search coordination

**Key Methods:**
- `_init_milvus()`: Initialize vector database
- `_extract_text_from_markdown()`: Convert markdown to plain text
- `build_index()`: Create Whoosh search index
- `_build_vector_index()`: Generate and store embeddings
- `keyword_search()`: Whoosh-based text search
- `vector_search()`: Semantic similarity search
- `hybrid_search()`: Combined search with score normalization

### MkDocs Integration
**Functions:**
- `find_mkdocs_config()`: Auto-detect mkdocs.yml/yaml
- `load_mkdocs_config()`: Parse YAML configuration
- `start_mkdocs_serve()`: Launch subprocess with thread management
- `stop_mkdocs_serve()`: Clean subprocess termination

### Search Architecture
**Three-layer Search Strategy:**
1. **Keyword Layer**: Whoosh full-text indexing
2. **Vector Layer**: Sentence transformer embeddings + Milvus
3. **Hybrid Layer**: Score normalization and result merging

**Index Schema (Whoosh):**
- path: Document file path (ID field)
- title: Document title (TEXT field)
- content: Full text content (TEXT field)
- headings: All headings for structural search (TEXT field)

## Error Handling Strategy

### Layered Error Handling
1. **Tool Level**: Each MCP tool has try-except with specific error messages
2. **Search Level**: Fallback from vector to keyword on failure
3. **System Level**: Signal handlers and cleanup functions
4. **Process Level**: Subprocess monitoring and restart capability

### Error Recovery
- Failed vector search → Use keyword-only
- MkDocs crash → Restart capability via tool
- Missing dependencies → Graceful feature degradation
- Index corruption → Rebuild index tool available

## Performance Considerations

### Caching Strategy
- Vector embeddings cached in memory (`self.embeddings`)
- Document metadata cached (`self.metadata`)
- MkDocs configuration cached (`_mkdocs_config`)

### Resource Optimization
- Lazy loading of vector search components
- Single model instance shared across searches
- Temporary directories cleaned automatically
- Subprocess resource limits

## Concurrency Model
- **Multi-agent Support**: Server handles concurrent MCP connections
- **Thread Safety**: Global state protected by subprocess management thread
- **Async-ready**: FastMCP supports async operations

## Extension Points
- Additional search methods can be added to DocsSearcher
- New MCP tools can be registered with @mcp.tool
- Custom indexing strategies via DocsSearcher subclassing
- Alternative vector models via configuration