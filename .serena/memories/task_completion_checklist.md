# Task Completion Checklist

When completing any development task in this project, follow these steps:

## 1. Code Quality Checks

### Format Code
```bash
uv run black .
```

### Lint Code  
```bash
uv run ruff check --fix .
```

### Type Checking
```bash
uv run mypy server.py
```

## 2. Testing

### Run Integration Tests
```bash
uv run python test_integration.py
```

### Run Unit Tests (if available)
```bash
uv run pytest
```

## 3. Verification

### Test the Server Manually
```bash
# Navigate to test project
cd test_mkdocs_project

# Run the server
uv run python ../server.py

# Verify MkDocs server starts on port 8000
# Verify MCP server responds to tools
```

### Check for Breaking Changes
- Ensure all MCP tools still work correctly
- Verify backward compatibility with existing MkDocs projects
- Test with both keyword and vector search

## 4. Documentation Updates

If changes affect functionality:
- Update README.md if needed
- Update CLAUDE.md for AI assistant guidance
- Update docstrings in code

## 5. Git Workflow

### Before Committing
```bash
# Check what changed
git status
git diff

# Run all quality checks
uv run black . && uv run ruff check --fix . && uv run mypy server.py

# Run tests
uv run python test_integration.py
```

### Commit Message Format
- Use clear, descriptive commit messages
- Follow conventional commits if applicable:
  - `feat:` for new features
  - `fix:` for bug fixes
  - `docs:` for documentation
  - `refactor:` for code refactoring
  - `test:` for test additions/changes
  - `chore:` for maintenance tasks

## 6. Special Considerations

### Vector Search Dependencies
- If modifying vector search, ensure graceful fallback to keyword-only search
- Test with and without sentence-transformers installed

### MkDocs Integration
- Test with different MkDocs configurations
- Ensure server cleanup on exit (subprocess termination)
- Verify index directory cleanup

### Performance
- Check search performance with larger documentation sets
- Monitor memory usage with vector embeddings
- Ensure proper resource cleanup

## Quick Validation Command
Run all checks in sequence:
```bash
uv run black . && \
uv run ruff check --fix . && \
uv run mypy server.py && \
uv run python test_integration.py
```

If all checks pass ✅, the task is ready for review/merge!