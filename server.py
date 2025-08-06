#!/usr/bin/env python

import atexit
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import markdown
import yaml
from fastmcp import FastMCP
from whoosh import index
from whoosh.fields import ID, TEXT, Schema
from whoosh.highlight import UppercaseFormatter
from whoosh.qparser import MultifieldParser

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    VECTOR_SEARCH_AVAILABLE = False

mcp = FastMCP("MkDocs RAG Server 🔍")

# Global variable to track the MkDocs serve process
_mkdocs_process = None
_mkdocs_thread = None
_mkdocs_config = None
_project_root = None


# Search functionality
class DocsSearcher:
    """Handles document indexing and searching for retrieval augmented generation."""

    def __init__(self, docs_dir: str = "docs"):
        self.docs_dir = Path(docs_dir)
        self.index_dir = None
        self.ix = None
        self.model = None
        self.embeddings = {}

        if VECTOR_SEARCH_AVAILABLE:
            try:
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
            except:
                self.model = None

    def _extract_text_from_markdown(
        self, file_path: Path
    ) -> tuple[str, str, list[str]]:
        """Extract title, content, and headings from markdown file."""
        try:
            content = file_path.read_text(encoding="utf-8")

            # Extract title (first H1 or filename)
            title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
            title = title_match.group(1) if title_match else file_path.stem

            # Extract all headings for context
            headings = re.findall(r"^#{1,6}\s+(.+)$", content, re.MULTILINE)

            # Convert markdown to plain text for better searching
            md = markdown.Markdown()
            plain_text = md.convert(content)
            # Remove HTML tags
            plain_text = re.sub(r"<[^>]+>", "", plain_text)

            return title, plain_text, headings
        except Exception:
            return file_path.stem, "", []

    def build_index(self) -> dict[str, Any]:
        """Build the Whoosh search index."""
        try:
            # Create temporary index directory
            self.index_dir = tempfile.mkdtemp()

            # Define schema
            schema = Schema(
                path=ID(stored=True),
                title=TEXT(stored=True),
                content=TEXT(stored=True),
                headings=TEXT(stored=True),
            )

            # Create index
            self.ix = index.create_in(self.index_dir, schema)
            writer = self.ix.writer()

            # Index all markdown files
            indexed_count = 0
            for file_path in self.docs_dir.rglob("*.md"):
                relative_path = file_path.relative_to(self.docs_dir)
                title, content, headings = self._extract_text_from_markdown(file_path)

                writer.add_document(
                    path=str(relative_path),
                    title=title,
                    content=content,
                    headings=" ".join(headings),
                )

                # Build embeddings if vector search is available
                if VECTOR_SEARCH_AVAILABLE and self.model:
                    full_text = f"{title} {' '.join(headings)} {content}"
                    embedding = self.model.encode(full_text)
                    self.embeddings[str(relative_path)] = embedding

                indexed_count += 1

            writer.commit()

            return {
                "success": True,
                "indexed_files": indexed_count,
                "index_location": self.index_dir,
                "vector_search_available": VECTOR_SEARCH_AVAILABLE
                and self.model is not None,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def keyword_search(self, query: str, max_results: int = 10) -> list[dict[str, Any]]:
        """Perform keyword-based search using Whoosh."""
        if not self.ix:
            self.build_index()

        results = []
        with self.ix.searcher() as searcher:
            # Search in multiple fields
            parser = MultifieldParser(["title", "content", "headings"], self.ix.schema)
            q = parser.parse(query)

            search_results = searcher.search(q, limit=max_results)

            # Highlight formatter
            formatter = UppercaseFormatter()

            for hit in search_results:
                # Get highlighted content
                highlighted = hit.highlights("content", top=3)
                if not highlighted:
                    # If no highlights, get a snippet
                    content = hit["content"]
                    snippet = content[:300] + "..." if len(content) > 300 else content
                else:
                    snippet = highlighted

                results.append({
                    "path": hit["path"],
                    "title": hit["title"],
                    "score": hit.score,
                    "snippet": snippet,
                })

        return results

    def vector_search(self, query: str, max_results: int = 10) -> list[dict[str, Any]]:
        """Perform semantic vector search."""
        if not VECTOR_SEARCH_AVAILABLE or not self.model:
            return []

        if not self.embeddings:
            self.build_index()

        # Encode query
        query_embedding = self.model.encode(query)

        # Calculate similarities
        similarities = []
        for path, doc_embedding in self.embeddings.items():
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1), doc_embedding.reshape(1, -1)
            )[0][0]
            similarities.append((path, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Get top results
        results = []
        for path, score in similarities[:max_results]:
            file_path = self.docs_dir / path
            if file_path.exists():
                title, content, _ = self._extract_text_from_markdown(file_path)
                snippet = content[:300] + "..." if len(content) > 300 else content

                results.append({
                    "path": path,
                    "title": title,
                    "score": float(score),
                    "snippet": snippet,
                })

        return results

    def cleanup(self):
        """Clean up temporary index directory."""
        if self.index_dir and Path(self.index_dir).exists():
            shutil.rmtree(self.index_dir)


# Global searcher instance
_searcher = None


def get_searcher(docs_dir: str = "docs") -> DocsSearcher:
    """Get or create the global searcher instance."""
    global _searcher
    if _searcher is None:
        _searcher = DocsSearcher(docs_dir)
    return _searcher


def read_document_impl(file_path: str, docs_dir: str = "docs") -> dict[str, Any]:
    """
    Implementation for reading a specific documentation file.

    Args:
        file_path: Path to the documentation file relative to docs_dir
        docs_dir: The documentation directory

    Returns:
        The document content and metadata
    """
    try:
        full_path = Path(docs_dir) / file_path

        if not full_path.exists():
            return {"success": False, "error": f"File not found: {file_path}"}

        if not full_path.suffix == ".md":
            return {"success": False, "error": "Only markdown files are supported"}

        content = full_path.read_text(encoding="utf-8")

        # Extract metadata
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        title = title_match.group(1) if title_match else full_path.stem

        headings = re.findall(r"^#{1,6}\s+(.+)$", content, re.MULTILINE)

        return {
            "success": True,
            "path": file_path,
            "title": title,
            "content": content,
            "headings": headings,
            "size": len(content),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


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
    return read_document_impl(file_path, docs_dir)


def list_documents_impl(docs_dir: str = "docs") -> dict[str, Any]:
    """
    Implementation for listing all documentation files available for retrieval.

    Args:
        docs_dir: The documentation directory to scan

    Returns:
        A list of all markdown files and their metadata
    """
    try:
        docs_path = Path(docs_dir)
        if not docs_path.exists():
            return {
                "success": False,
                "error": f"Documentation directory '{docs_dir}' not found",
            }

        files = []
        for file_path in docs_path.rglob("*.md"):
            relative_path = file_path.relative_to(docs_path)

            # Extract title
            try:
                content = file_path.read_text(encoding="utf-8")
                title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
                title = title_match.group(1) if title_match else file_path.stem
            except:
                title = file_path.stem

            files.append({
                "path": str(relative_path),
                "title": title,
                "size": file_path.stat().st_size,
            })

        return {
            "success": True,
            "docs_dir": str(docs_path.absolute()),
            "document_count": len(files),
            "documents": sorted(files, key=lambda x: x["path"]),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool
def list_documents(docs_dir: str = "docs") -> dict[str, Any]:
    """
    List all documentation files available for retrieval.

    Args:
        docs_dir: The documentation directory to scan

    Returns:
        A list of all markdown files and their metadata
    """
    return list_documents_impl(docs_dir)


def keyword_search_impl(
    query: str, max_results: int = 10, docs_dir: str = "docs"
) -> dict[str, Any]:
    """
    Implementation for searching documentation using keyword-based search.

    Args:
        query: The search query
        max_results: Maximum number of results to return
        docs_dir: The documentation directory to search

    Returns:
        Search results with snippets and relevance scores
    """
    try:
        searcher = get_searcher(docs_dir)

        # Ensure index is built
        if not searcher.ix:
            index_result = searcher.build_index()
            if not index_result["success"]:
                return index_result

        results = searcher.keyword_search(query, max_results)

        return {
            "success": True,
            "query": query,
            "result_count": len(results),
            "results": results,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool
def keyword_search(
    query: str, max_results: int = 10, docs_dir: str = "docs"
) -> dict[str, Any]:
    """
    Search documentation using keyword-based search.

    Args:
        query: The search query
        max_results: Maximum number of results to return
        docs_dir: The documentation directory to search

    Returns:
        Search results with snippets and relevance scores
    """
    return keyword_search_impl(query, max_results, docs_dir)


def vector_search_impl(
    query: str, max_results: int = 10, docs_dir: str = "docs"
) -> dict[str, Any]:
    """
    Implementation for searching documentation using semantic vector search.

    Args:
        query: The search query
        max_results: Maximum number of results to return
        docs_dir: The documentation directory to search

    Returns:
        Search results with snippets and similarity scores
    """
    try:
        if not VECTOR_SEARCH_AVAILABLE:
            return {
                "success": False,
                "error": "Vector search is not available. Install sentence-transformers: pip install sentence-transformers",
            }

        searcher = get_searcher(docs_dir)

        if not searcher.model:
            return {
                "success": False,
                "error": "Failed to load the sentence transformer model",
            }

        # Ensure embeddings are built
        if not searcher.embeddings:
            index_result = searcher.build_index()
            if not index_result["success"]:
                return index_result

        results = searcher.vector_search(query, max_results)

        return {
            "success": True,
            "query": query,
            "result_count": len(results),
            "results": results,
            "model": "all-MiniLM-L6-v2",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool
def vector_search(
    query: str, max_results: int = 10, docs_dir: str = "docs"
) -> dict[str, Any]:
    """
    Search documentation using semantic vector search.

    Args:
        query: The search query
        max_results: Maximum number of results to return
        docs_dir: The documentation directory to search

    Returns:
        Search results with snippets and similarity scores
    """
    return vector_search_impl(query, max_results, docs_dir)


@mcp.tool
def search(
    query: str,
    search_type: str = "hybrid",
    max_results: int = 10,
    docs_dir: str = "docs",
) -> dict[str, Any]:
    """
    Search documentation using keyword, vector, or hybrid search.

    Args:
        query: The search query
        search_type: Type of search - "keyword", "vector", or "hybrid"
        max_results: Maximum number of results to return
        docs_dir: The documentation directory to search

    Returns:
        Search results with snippets and relevance scores
    """
    try:
        if search_type == "keyword":
            return keyword_search_impl(query, max_results, docs_dir)
        elif search_type == "vector":
            return vector_search_impl(query, max_results, docs_dir)
        elif search_type == "hybrid":
            # Perform both searches
            keyword_results = keyword_search_impl(query, max_results, docs_dir)
            vector_results = vector_search_impl(query, max_results, docs_dir)

            # Combine results
            if not keyword_results["success"]:
                return keyword_results

            # If vector search is not available, return keyword results
            if not vector_results.get("success", False):
                return keyword_results

            # Merge results, prioritizing by combined score
            path_scores = {}
            path_data = {}

            # Add keyword results
            for result in keyword_results.get("results", []):
                path = result["path"]
                path_scores[path] = result["score"]
                path_data[path] = result

            # Add vector results (normalize scores to similar range)
            for result in vector_results.get("results", []):
                path = result["path"]
                # Vector scores are 0-1, keyword scores can be higher
                normalized_score = result["score"] * 10

                if path in path_scores:
                    # Average the scores if path appears in both
                    path_scores[path] = (path_scores[path] + normalized_score) / 2
                else:
                    path_scores[path] = normalized_score
                    path_data[path] = result

            # Sort by combined score
            sorted_paths = sorted(path_scores.items(), key=lambda x: x[1], reverse=True)

            # Build final results
            final_results = []
            for path, score in sorted_paths[:max_results]:
                result = path_data[path].copy()
                result["score"] = score
                result["search_methods"] = []

                # Mark which search methods found this result
                if any(r["path"] == path for r in keyword_results.get("results", [])):
                    result["search_methods"].append("keyword")
                if any(r["path"] == path for r in vector_results.get("results", [])):
                    result["search_methods"].append("vector")

                final_results.append(result)

            return {
                "success": True,
                "query": query,
                "search_type": "hybrid",
                "result_count": len(final_results),
                "results": final_results,
            }
        else:
            return {
                "success": False,
                "error": f"Invalid search_type: {search_type}. Must be 'keyword', 'vector', or 'hybrid'",
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool
def rebuild_search_index(docs_dir: str = "docs") -> dict[str, Any]:
    """
    Rebuild the search index for better performance.

    Args:
        docs_dir: The documentation directory to index

    Returns:
        Information about the rebuilt index
    """
    try:
        searcher = get_searcher(docs_dir)
        result = searcher.build_index()
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.resource("mkdocs://documents")
def get_documents_info() -> dict[str, Any]:
    """Get information about available documents for retrieval."""
    result = list_documents_impl()
    if result["success"]:
        return {
            "document_count": result["document_count"],
            "docs_dir": result["docs_dir"],
            "documents": result["documents"],
        }
    return {"error": result.get("error", "Failed to list documents")}


@mcp.prompt(
    name="mkdocs-rag-search",
    description="Generate a search query for MkDocs documentation",
)
def mkdocs_rag_search(topic: str = "configuration") -> list[dict[str, str]]:
    """Generate a search query for finding relevant documentation."""
    return [
        {
            "role": "user",
            "content": f"I need to find documentation about {topic} in my MkDocs project. Can you help me search for relevant information?",
        },
        {
            "role": "assistant",
            "content": f"""I'll help you search for documentation about {topic}. Let me search through your MkDocs documentation using different search methods.

## Search Strategy

I'll use a hybrid search approach combining:
1. **Keyword search** - Find exact matches and relevant terms
2. **Semantic search** - Find conceptually related content
3. **Combined results** - Merge and rank by relevance

## Performing Search

Let me search for "{topic}" in your documentation:

```python
# Hybrid search for comprehensive results
results = search(
    query="{topic}",
    search_type="hybrid",
    max_results=10
)

# For more specific keyword matches
keyword_results = keyword_search(
    query="{topic}",
    max_results=5
)

# For semantic understanding
vector_results = vector_search(
    query="{topic}",
    max_results=5
)
```

## Understanding Results

The search results will include:
- **Path**: Location of the document
- **Title**: Document title
- **Score**: Relevance score
- **Snippet**: Preview of matching content
- **Search methods**: Which search type found it

## Reading Full Documents

Once you find relevant documents, you can read them in full:

```python
# Read a specific document
doc = read_document(
    file_path="path/to/document.md"
)
```

## Tips for Better Search

1. **Use specific terms**: More specific queries yield better results
2. **Try variations**: Different phrasings might find different documents
3. **Check related topics**: Semantic search finds conceptually related content
4. **Rebuild index**: Run `rebuild_search_index()` if documents were recently added

Would you like me to search for "{topic}" now, or would you like to refine your search query?""",
        },
    ]


def find_mkdocs_config(start_path: Path = None) -> Path | None:
    """Find mkdocs.yml file by traversing up from start_path."""
    if start_path is None:
        start_path = Path.cwd()

    current = start_path.resolve()

    # Check current directory and parent directories
    for _ in range(10):  # Limit search depth
        for config_name in ["mkdocs.yml", "mkdocs.yaml"]:
            config_path = current / config_name
            if config_path.exists():
                return config_path

        parent = current.parent
        if parent == current:  # Reached root
            break
        current = parent

    return None


def load_mkdocs_config(config_path: Path) -> dict[str, Any]:
    """Load and parse MkDocs configuration."""
    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading MkDocs config: {e}", file=sys.stderr)
        return {}


def start_mkdocs_serve(project_root: Path, port: int = 8000):
    """Start MkDocs development server in background."""
    global _mkdocs_process

    def run_mkdocs():
        try:
            print(f"Starting MkDocs server at http://localhost:{port}", file=sys.stderr)
            _mkdocs_process = subprocess.Popen(
                ["mkdocs", "serve", "--dev-addr", f"localhost:{port}"],
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            # Stream output
            for line in _mkdocs_process.stdout:
                if line.strip():
                    print(f"[MkDocs] {line.strip()}", file=sys.stderr)

        except Exception as e:
            print(f"Error starting MkDocs server: {e}", file=sys.stderr)

    thread = threading.Thread(target=run_mkdocs, daemon=True)
    thread.start()
    return thread


def stop_mkdocs_serve():
    """Stop the MkDocs development server."""
    global _mkdocs_process

    if _mkdocs_process:
        try:
            _mkdocs_process.terminate()
            _mkdocs_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _mkdocs_process.kill()
            _mkdocs_process.wait()
        except Exception as e:
            print(f"Error stopping MkDocs server: {e}", file=sys.stderr)
        finally:
            _mkdocs_process = None


def initialize_mkdocs_integration():
    """Initialize MkDocs integration by finding config and starting server."""
    global _mkdocs_config, _project_root, _mkdocs_thread

    # Find MkDocs config
    config_path = find_mkdocs_config()

    if not config_path:
        print(
            "Warning: No mkdocs.yml found. MkDocs serve will not be started.",
            file=sys.stderr,
        )
        print("The MCP server will run in standalone mode.", file=sys.stderr)
        return False

    print(f"Found MkDocs config at: {config_path}", file=sys.stderr)
    _project_root = config_path.parent

    # Load config
    _mkdocs_config = load_mkdocs_config(config_path)
    site_name = _mkdocs_config.get("site_name", "MkDocs")
    print(f"Loaded MkDocs project: {site_name}", file=sys.stderr)

    # Check if MkDocs is installed
    try:
        result = subprocess.run(
            ["mkdocs", "--version"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print(f"MkDocs version: {result.stdout.strip()}", file=sys.stderr)
        else:
            print(
                "Warning: MkDocs command failed. Server will not be started.",
                file=sys.stderr,
            )
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"Warning: MkDocs not found or not responding: {e}", file=sys.stderr)
        print("Install MkDocs with: pip install mkdocs", file=sys.stderr)
        return False

    # Start MkDocs serve
    port = int(os.getenv("MKDOCS_PORT", "8000"))
    _mkdocs_thread = start_mkdocs_serve(_project_root, port)

    # Wait a moment for server to start
    time.sleep(2)

    print("\n" + "=" * 60, file=sys.stderr)
    print("MkDocs RAG Server initialized successfully!", file=sys.stderr)
    print(f"  - MkDocs site: http://localhost:{port}", file=sys.stderr)
    print(f"  - Project root: {_project_root}", file=sys.stderr)
    print(f"  - Docs directory: {_project_root / 'docs'}", file=sys.stderr)
    print("=" * 60 + "\n", file=sys.stderr)

    return True


@mcp.tool
def get_mkdocs_info() -> dict[str, Any]:
    """
    Get information about the current MkDocs project.

    Returns:
        Information about the MkDocs configuration and server status
    """
    global _mkdocs_config, _project_root, _mkdocs_process

    if not _mkdocs_config:
        return {"success": False, "error": "No MkDocs project loaded"}

    server_running = _mkdocs_process is not None and _mkdocs_process.poll() is None
    port = int(os.getenv("MKDOCS_PORT", "8000"))

    return {
        "success": True,
        "project_root": str(_project_root),
        "config_path": str(_project_root / "mkdocs.yml"),
        "docs_dir": str(_project_root / _mkdocs_config.get("docs_dir", "docs")),
        "site_name": _mkdocs_config.get("site_name", "MkDocs"),
        "site_url": _mkdocs_config.get("site_url", ""),
        "theme": _mkdocs_config.get("theme", {}),
        "plugins": list(_mkdocs_config.get("plugins", [])),
        "server_running": server_running,
        "server_url": f"http://localhost:{port}" if server_running else None,
    }


@mcp.tool
def restart_mkdocs_server(port: int | None = None) -> dict[str, Any]:
    """
    Restart the MkDocs development server.

    Args:
        port: Port to run the server on (default: 8000 or MKDOCS_PORT env var)

    Returns:
        Status of the restart operation
    """
    global _mkdocs_thread, _project_root

    if not _project_root:
        return {"success": False, "error": "No MkDocs project loaded"}

    # Stop existing server
    stop_mkdocs_serve()

    # Start new server
    if port is None:
        port = int(os.getenv("MKDOCS_PORT", "8000"))

    _mkdocs_thread = start_mkdocs_serve(_project_root, port)
    time.sleep(2)  # Wait for server to start

    return {
        "success": True,
        "message": f"MkDocs server restarted on port {port}",
        "server_url": f"http://localhost:{port}",
    }


# Cleanup function
def cleanup():
    """Clean up resources on exit."""
    global _searcher

    print("\nShutting down MkDocs RAG Server...", file=sys.stderr)

    # Stop MkDocs server
    stop_mkdocs_serve()

    # Clean up search index
    if _searcher:
        _searcher.cleanup()

    print("Cleanup complete.", file=sys.stderr)


# Register cleanup
atexit.register(cleanup)


# Handle signals for graceful shutdown
def signal_handler(signum, frame):
    cleanup()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Main entry point for the MCP server."""
    # Initialize MkDocs integration
    mkdocs_initialized = initialize_mkdocs_integration()

    # Adjust default docs_dir based on project
    if _project_root and _mkdocs_config:
        docs_dir = _mkdocs_config.get("docs_dir", "docs")
        # Update the default docs_dir for all tools
        default_docs_dir = str(_project_root / docs_dir)

        # Monkey-patch the default parameter for tools
        # This ensures tools use the correct docs directory
        import inspect

        for name, func in inspect.getmembers(sys.modules[__name__]):
            if hasattr(func, "__mcp_tool__"):
                sig = inspect.signature(func)
                params = sig.parameters
                if "docs_dir" in params:
                    # Update default value
                    new_params = []
                    for param_name, param in params.items():
                        if param_name == "docs_dir":
                            new_param = param.replace(default=default_docs_dir)
                            new_params.append(new_param)
                        else:
                            new_params.append(param)
                    func.__signature__ = sig.replace(parameters=new_params)

    # Run the MCP server
    mcp.run()


if __name__ == "__main__":
    main()
