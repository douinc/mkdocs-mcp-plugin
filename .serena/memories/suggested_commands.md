# Suggested Commands for Development

## Build and Development Commands

### Install Dependencies (using UV - recommended)
```bash
# Install all dependencies including dev extras
uv sync --all-extras

# Install specific dependency
uv add package_name

# Update dependencies
uv lock --upgrade
```

### Run the MCP Server
```bash
# From any MkDocs project directory
uv run python server.py

# Or using the installed script
mkdocs-mcp
```

### Testing
```bash
# Run integration tests
uv run python test_integration.py

# Run pytest (when tests are available)
uv run pytest

# Run with coverage
uv run pytest --cov
```

### Code Quality Commands
```bash
# Format code with Black
uv run black .

# Check and fix linting issues with Ruff
uv run ruff check --fix .

# Type checking with MyPy
uv run mypy server.py

# Run all formatters and linters
uv run black . && uv run ruff check --fix .
```

### Build and Distribution
```bash
# Build package for distribution
uv build

# Publish to PyPI
uv publish
```

## System Commands (Darwin/macOS)

### Git Commands
```bash
git status              # Check repository status
git diff               # View uncommitted changes
git add .              # Stage all changes
git commit -m "msg"    # Commit changes
git push              # Push to remote
```

### File System Commands  
```bash
ls -la                # List files with details (colorized on macOS)
cd path/to/dir        # Change directory
find . -name "*.py"   # Find Python files
grep -r "pattern" .   # Search for pattern recursively (colorized)
```

### Python/UV Environment
```bash
which python          # Show Python path
which uv             # Show UV path
uv --version         # Check UV version
python --version     # Check Python version
```

## Project-Specific Commands

### Run MkDocs Development Server
```bash
# From test project
cd test_mkdocs_project
mkdocs serve         # Start MkDocs server on port 8000
```

### Environment Variables
```bash
export MKDOCS_PORT=8000       # Set MkDocs port
export MKDOCS_DEBUG=1         # Enable debug output
```

## Quick Development Workflow
```bash
# 1. Make changes to server.py
# 2. Format and lint
uv run black . && uv run ruff check --fix .

# 3. Type check
uv run mypy server.py

# 4. Test changes
uv run python test_integration.py

# 5. Run server to test manually
cd test_mkdocs_project && uv run python ../server.py
```