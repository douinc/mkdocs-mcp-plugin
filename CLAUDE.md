# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an MCP (Model Context Protocol) server plugin for MkDocs that provides intelligent search, retrieval, and integration capabilities for AI agents. The plugin automatically detects MkDocs projects, launches the development server, and provides powerful tools for querying documentation.

## Build and Development Commands

```bash
# Install dependencies using UV (recommended)
uv sync --all-extras

# Run the MCP server from any MkDocs project
uv run python server.py

# Run tests
uv run pytest

# Run integration tests
uv run python test_integration.py

# Format code
uv run black .
uv run ruff check --fix .

# Type checking
uv run mypy server.py

# Build package for distribution
uv build

# Publish to PyPI
uv publish
```

## Architecture

### Core Components

1. **MCP Server (`server.py`)**: Main FastMCP server implementation that:
   - Auto-detects MkDocs projects by looking for `mkdocs.yml` or `mkdocs.yaml`
   - Manages MkDocs development server lifecycle (starts/stops/restarts)
   - Provides MCP tools for document operations and search
   - Handles concurrent agent connections

2. **DocsSearcher Class**: Handles document indexing and searching with three search methods:
   - **Keyword Search**: Using Whoosh full-text indexing
   - **Vector Search**: Using sentence transformers (`all-MiniLM-L6-v2`) for semantic search
   - **Hybrid Search**: Combines both methods with score normalization

3. **Search Index Schema**: Whoosh index with fields:
   - `path`: Document file path
   - `title`: Document title (from first H1 or filename)
   - `content`: Full text content (markdown converted to plain text)
   - `headings`: All heading text for structural search

### MCP Tools Available

- `read_document`: Read a specific markdown file with metadata
- `list_documents`: Get list of all available documentation
- `search`: Hybrid search combining keyword and semantic methods
- `keyword_search`: Fast text-based search using Whoosh
- `vector_search`: Semantic similarity search using sentence transformers
- `get_mkdocs_info`: Get current MkDocs project configuration
- `restart_mkdocs_server`: Restart the MkDocs development server
- `rebuild_search_index`: Rebuild the search index

### Key Design Patterns

1. **Graceful Degradation**: Falls back to keyword-only search if vector search dependencies are unavailable
2. **Automatic Resource Management**: Uses `atexit` and signal handlers to ensure clean shutdown of MkDocs server
3. **Concurrent Access**: Supports multiple agent connections to the same MCP server
4. **Error Recovery**: Comprehensive error handling with detailed error messages

## Important Implementation Details

- The server runs MkDocs development server as a subprocess managed in a separate thread
- Search index is built in temporary directory and cleaned up on exit
- Vector embeddings are cached in memory for performance
- All markdown files are converted to plain text for better search accuracy
- Search results include score normalization and deduplication for hybrid search

## Testing

Integration tests are in `test_integration.py` which validates the full workflow of starting the server, performing searches, and cleaning up resources.

Test MkDocs project is provided in `test_mkdocs_project/` directory for development and testing.

## Dependencies Management

Core dependencies are managed in `pyproject.toml`:
- Required: `fastmcp`, `markdown`, `pyyaml`, `whoosh`
- Optional (for vector search): `numpy`, `scikit-learn`, `sentence-transformers`
- Development: `pytest`, `black`, `ruff`, `mypy`, `pre-commit`

Python 3.12+ is required.