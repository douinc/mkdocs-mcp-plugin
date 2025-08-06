# Code Style and Conventions

## Python Version
- Python 3.12+ required
- Target version specified in pyproject.toml

## Code Style
- **Formatter**: Black with line-length=88
- **Linter**: Ruff with specific rules:
  - Selected: E, W, F, I, B, C4, ARG, SIM
  - Ignored: E501 (line too long), W503, E203
- **Type Checking**: MyPy with strict settings:
  - warn_return_any = true
  - warn_unused_configs = true
  - disallow_untyped_defs = true

## Naming Conventions
- Classes: PascalCase (e.g., `DocsSearcher`)
- Functions: snake_case (e.g., `read_document_impl`)
- Private module variables: Leading underscore (e.g., `_mkdocs_process`)
- Constants: UPPER_SNAKE_CASE (e.g., `VECTOR_SEARCH_AVAILABLE`)

## Documentation Style
- Functions use docstrings with Args and Returns sections
- Type hints are mandatory for all functions
- Example:
```python
@mcp.tool
def read_document(file_path: str, docs_dir: str = "docs") -> dict[str, Any]:
    """
    Read a specific documentation file.

    Args:
        file_path: Path to the documentation file relative to docs_dir
        docs_dir: The documentation directory

    Returns:
        The document content and metadata
    """
```

## Import Organization
- Standard library imports first
- Third-party imports second
- Local imports last
- Each group separated by blank line

## Error Handling
- Comprehensive try-except blocks with specific error messages
- Graceful degradation (e.g., fallback to keyword search if vector search fails)
- Detailed error reporting with context

## Testing Conventions
- Test files: `test_*.py` or `*_test.py`
- Test directory: `tests/` (though integration tests are in root)
- Use pytest for testing framework

## Code Organization Patterns
- Main functionality in single `server.py` file
- Helper classes (like `DocsSearcher`) defined in same file
- MCP decorators (@mcp.tool) for tool functions
- Implementation functions separate from decorated functions (e.g., `read_document` vs `read_document_impl`)