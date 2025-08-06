# MkDocs MCP Plugin 🔍

A comprehensive MCP (Model Context Protocol) server for MkDocs documentation that provides intelligent search, retrieval, and integration capabilities for AI agents. This plugin automatically detects MkDocs projects, launches the development server, and provides powerful tools for querying documentation.

## Features

### 🚀 Auto-Detection & Integration
- Automatically detects `mkdocs.yml` or `mkdocs.yaml` in your project
- Launches MkDocs development server alongside the MCP server
- Seamless integration with existing MkDocs workflows

### 🔎 Advanced Search Capabilities
- **Keyword Search**: Fast, accurate text-based search using Whoosh indexing
- **Vector Search**: Semantic search using sentence transformers (`all-MiniLM-L6-v2`)
- **Hybrid Search**: Combines both keyword and semantic search for optimal results
- **Real-time Indexing**: Automatically indexes markdown files with full-text search

### 📄 Document Operations
- Read individual markdown files with metadata extraction
- List all available documentation with titles and paths
- Extract headings, titles, and content structure
- Support for nested directory structures

### 🤖 MCP Protocol Compliance
- Full MCP server implementation using FastMCP
- Tools, resources, and prompts for agent interaction
- Structured responses with comprehensive error handling
- Support for concurrent agent connections

## Installation

### Using UV/UVX (Recommended)

Install and run directly with uvx:

```bash
# Install and run in one command
uvx mkdocs-mcp-plugin

# Or install globally
uv tool install mkdocs-mcp-plugin

# Then run from any MkDocs project
mkdocs-mcp
```

### Using pip

```bash
# Install from source
pip install git+https://github.com/douinc/mkdocs-mcp-plugin.git

# Or clone and install locally
git clone https://github.com/douinc/mkdocs-mcp-plugin.git
cd mkdocs-mcp-plugin
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/douinc/mkdocs-mcp-plugin.git
cd mkdocs-mcp-plugin

# Install with UV (recommended)
uv sync --all-extras

# Or with pip
pip install -e ".[dev]"
```

## Usage

### Basic Usage

Navigate to any directory containing a `mkdocs.yml` file and run:

```bash
mkdocs-mcp
```

The server will:
1. Detect your MkDocs configuration
2. Start the MkDocs development server (default: http://localhost:8000)
3. Launch the MCP server for agent interaction
4. Index your documentation for search

### Configuration

The server automatically adapts to your MkDocs configuration:

```yaml
# mkdocs.yml
site_name: My Documentation
docs_dir: docs  # Custom docs directory
site_url: https://mydocs.example.com
theme:
  name: material
plugins:
  - search
```

### Environment Variables

- `MKDOCS_PORT`: Port for the MkDocs server (default: 8000)
- `MCP_PORT`: Port for the MCP server (auto-selected)

## MCP Tools

### Document Operations

#### `read_document`
Read a specific markdown file with metadata:

```python
{
  "file_path": "getting-started.md",
  "docs_dir": "docs"  # Optional, auto-detected
}
```

#### `list_documents`
Get a list of all available documentation:

```python
{
  "docs_dir": "docs"  # Optional, auto-detected
}
```

### Search Operations

#### `search` (Hybrid Search)
Combines keyword and semantic search:

```python
{
  "query": "authentication setup",
  "search_type": "hybrid",  # "keyword", "vector", or "hybrid"
  "max_results": 10
}
```

#### `keyword_search`
Fast text-based search:

```python
{
  "query": "configuration options",
  "max_results": 10
}
```

#### `vector_search`
Semantic similarity search:

```python
{
  "query": "how to deploy",
  "max_results": 10
}
```

### Utility Tools

#### `get_mkdocs_info`
Get information about the current MkDocs project:

```python
{}  # No parameters required
```

#### `restart_mkdocs_server`
Restart the MkDocs development server:

```python
{
  "port": 8001  # Optional, defaults to 8000
}
```

#### `rebuild_search_index`
Rebuild the search index:

```python
{
  "docs_dir": "docs"  # Optional, auto-detected
}
```

## MCP Resources

### `mkdocs://documents`
Access to document metadata and structure:

```json
{
  "document_count": 25,
  "docs_dir": "/path/to/docs",
  "documents": [
    {
      "path": "index.md",
      "title": "Welcome",
      "size": 1024
    }
  ]
}
```

## MCP Prompts

### `mkdocs-rag-search`
Generate intelligent search queries for documentation:

```python
{
  "topic": "authentication"  # Search topic
}
```

## Advanced Features

### Vector Search Dependencies

For semantic search capabilities, ensure these packages are installed:

```bash
# Included in default installation
pip install sentence-transformers scikit-learn numpy
```

If these packages are not available, the server will fall back to keyword-only search.

### Custom Index Configuration

The server uses Whoosh for indexing with the following schema:
- `path`: Document file path
- `title`: Document title (from first H1 or filename)
- `content`: Full text content (markdown converted to plain text)
- `headings`: All heading text for structural search

### Search Result Structure

All search operations return results in this format:

```json
{
  "success": true,
  "query": "your search query",
  "result_count": 5,
  "results": [
    {
      "path": "docs/api/authentication.md",
      "title": "Authentication Guide",
      "score": 0.95,
      "snippet": "...highlighted excerpt...",
      "search_methods": ["keyword", "vector"]
    }
  ]
}
```

## Integration Examples

### Claude Code Configuration

Add to your Claude Code config:

```json
{
  "mcpServers": {
    "mkdocs-mcp": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/douinc/mkdocs-mcp-plugin",
        "--with",
        "mkdocs-material",
        "--with",
        "mkdocs-git-revision-date-localized-plugin",
        "--with",
        "mkdocs-minify-plugin",
        "--with",
        "mkdocs-mermaid2-plugin",
        "--with",
        "mkdocs-print-site-plugin",
        "mkdocs-mcp"
      ],
      "env": {
        "MKDOCS_PORT": "8000"
      }
    }
  }
}
```

## Error Handling

The server provides comprehensive error handling:

- **Missing MkDocs**: Graceful fallback to MCP-only mode
- **Invalid configurations**: Clear error messages with suggestions
- **Search failures**: Automatic fallback between search methods
- **File access errors**: Detailed error reporting with context

## Troubleshooting

### Common Issues

1. **MkDocs server not starting**:
   ```bash
   # Check if MkDocs is installed
   mkdocs --version
   
   # Install if missing
   pip install mkdocs
   ```

2. **Vector search not working**:
   ```bash
   # Install optional dependencies
   pip install sentence-transformers
   ```

3. **Permission errors**:
   ```bash
   # Check file permissions
   ls -la mkdocs.yml
   ```

### Debug Mode

Run with verbose output:

```bash
# Set environment variable for debug output
MKDOCS_DEBUG=1 mkdocs-mcp
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Run tests: `uv run pytest`
5. Format code: `uv run black . && uv run ruff check --fix .`
6. Submit a pull request

### Development Setup

```bash
git clone https://github.com/douinc/mkdocs-mcp-plugin.git
cd mkdocs-mcp-plugin

# Install with all dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Run linting
uv run ruff check
uv run black --check .
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Changelog

### v0.1.0
- Initial release
- MkDocs auto-detection and server integration
- Hybrid search with keyword and vector capabilities
- Full MCP protocol compliance
- UV/UVX support

## Support

- 📚 [Documentation](https://github.com/douinc/mkdocs-mcp-plugin#readme)
- 🐛 [Issue Tracker](https://github.com/douinc/mkdocs-mcp-plugin/issues)
- 💡 [Feature Requests](https://github.com/douinc/mkdocs-mcp-plugin/discussions)

---

Built with ❤️ by [dou inc.](https://github.com/douinc)