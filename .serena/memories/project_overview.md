# MkDocs MCP Plugin - Project Overview

## Project Purpose
MkDocs MCP Plugin is a comprehensive Model Context Protocol (MCP) server for MkDocs documentation that provides intelligent search, retrieval, and integration capabilities for AI agents. It automatically detects MkDocs projects, launches the development server, and provides powerful tools for querying documentation.

## Tech Stack
- **Language**: Python 3.12+
- **Framework**: FastMCP for MCP server implementation
- **Documentation**: MkDocs with Material theme support
- **Search Engine**: 
  - Whoosh for keyword indexing
  - Sentence Transformers (all-MiniLM-L6-v2) for semantic search
  - Milvus for vector database
- **Build System**: Hatchling with UV package manager
- **Dependencies Management**: UV (uv.lock) and pyproject.toml

## Key Features
- Auto-detection of MkDocs projects
- Hybrid search (keyword + vector search)
- Real-time document indexing
- Full MCP protocol compliance
- Support for concurrent agent connections

## Repository Structure
- `server.py` - Main MCP server implementation
- `pyproject.toml` - Project configuration and dependencies
- `test_integration.py` - Integration tests
- `test_mkdocs_project/` - Test MkDocs project for development
- `.github/workflows/` - CI/CD workflows
- `CLAUDE.md` - Documentation for Claude Code AI assistant

## MCP Tools Available
- read_document - Read markdown files with metadata
- list_documents - Get list of all documentation
- search - Hybrid search (keyword + semantic)
- keyword_search - Fast text-based search
- vector_search - Semantic similarity search
- get_mkdocs_info - Get MkDocs project configuration
- restart_mkdocs_server - Restart MkDocs dev server
- rebuild_search_index - Rebuild search index

## Package Information
- **Name**: mkdocs-mcp-plugin
- **Version**: 0.2.0
- **Author**: Dongook Son (d@dou.so)
- **License**: MIT
- **Repository**: https://github.com/dongookson/mkdocs-mcp-plugin