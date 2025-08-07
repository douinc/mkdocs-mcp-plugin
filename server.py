#!/usr/bin/env python

import atexit
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import markdown
import numpy as np
import yaml
from fastmcp import FastMCP
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    connections,
    utility,
)
from pymilvus.model.dense import SentenceTransformerEmbeddingFunction
from pymilvus.model.sparse.bm25 import BM25EmbeddingFunction
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
from sklearn.metrics.pairwise import cosine_similarity

VECTOR_SEARCH_AVAILABLE = True
SPARSE_SEARCH_AVAILABLE = True

mcp = FastMCP("MkDocs RAG Server 🔍")

# Global variable to track the MkDocs serve process
_mkdocs_process = None
_mkdocs_thread = None
_mkdocs_config = None
_project_root = None


# Search functionality
class DocsSearcher:
    """Handles document indexing and searching for retrieval augmented generation."""

    def __init__(self, docs_dir: str = "docs", project_root: Optional[Path] = None):
        self.docs_dir = Path(docs_dir)
        self.project_root = project_root or Path.cwd()
        self.index_dir = self.project_root / ".mkdocs_vector"
        self.collection = None
        self.collection_name = "mkdocs_documents"
        self.milvus_connected = False
        self.embeddings = {}
        self.metadata = {}
        self.milvus_client = None
        self.bm25_ef = None
        self.sentence_transformer_ef = None

        # Initialize file paths regardless of storage type
        self.embeddings_file = self.index_dir / "embeddings.npz"
        self.metadata_file = self.index_dir / "metadata.yaml"

        # Initialize embedding functions for vector search
        if VECTOR_SEARCH_AVAILABLE:
            try:
                self.sentence_transformer_ef = SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2", device="cpu"
                )
                self._init_milvus()
            except Exception as e:
                print(f"Failed to initialize vector search: {e}", file=sys.stderr)
                self.sentence_transformer_ef = None

    def _init_milvus(self):
        """Initialize Milvus connection and collection."""
        try:
            # Ensure index directory exists
            self.index_dir.mkdir(exist_ok=True)

            # Initialize MilvusClient for easier API
            self.milvus_client = MilvusClient(uri=str(self.index_dir / "milvus.db"))

            # Connect to Milvus Lite (local file-based instance)
            connections.connect(
                alias="default",
                host="localhost",
                port="19530",
                # Use local file storage
                uri=str(self.index_dir / "milvus.db"),
            )
            self.milvus_connected = True

            # Check if collection exists
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                self.collection.load()
            else:
                self._create_collection()

        except Exception as e:
            print(
                f"Milvus initialization failed, using file-based storage: {e}",
                file=sys.stderr,
            )
            self.milvus_connected = False
            # Fall back to simple file-based storage
            self._init_file_storage()

    def _init_file_storage(self):
        """Initialize simple file-based storage for vectors."""
        self.index_dir.mkdir(exist_ok=True)

        # Load existing embeddings if available
        if self.embeddings_file.exists() and self.metadata_file.exists():
            try:
                data = np.load(self.embeddings_file, allow_pickle=True)
                self.embeddings = {k: v for k, v in data.items()}

                with open(self.metadata_file, "r") as f:
                    self.metadata = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"Failed to load existing index: {e}", file=sys.stderr)
                self.embeddings = {}
                self.metadata = {}
        else:
            self.embeddings = {}
            self.metadata = {}

    def _create_collection(self):
        """Create Milvus collection with schema."""
        # Define schema - only include sparse vector field if available
        fields = [
            FieldSchema(
                name="doc_id", dtype=DataType.INT64, is_primary=True, auto_id=True
            ),
            FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="headings", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(
                name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384
            ),  # all-MiniLM-L6-v2 dimension
        ]

        # Only add sparse vector field if the functionality is available
        if SPARSE_SEARCH_AVAILABLE:
            fields.append(
                FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR)
            )

        schema = CollectionSchema(fields, "MkDocs documentation collection")

        # Create collection
        self.collection = Collection(self.collection_name, schema)

        # Create index for dense vector field
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128},
        }
        self.collection.create_index("embedding", index_params)

        # Create index for sparse vector field if available
        if SPARSE_SEARCH_AVAILABLE:
            sparse_index_params = {
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "IP",  # Inner Product for sparse vectors
            }
            self.collection.create_index("sparse_vector", sparse_index_params)

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
        """Build the search index."""
        try:
            # Ensure index directory exists
            self.index_dir.mkdir(exist_ok=True)

            # Collect all documents
            documents = []
            paths = []
            titles = []
            contents = []
            headings_list = []
            embeddings = []
            sparse_vectors = []

            # Initialize BM25 embedding function for sparse vectors
            if SPARSE_SEARCH_AVAILABLE:
                self.bm25_ef = BM25EmbeddingFunction(
                    analyzer=build_default_analyzer(language="en")
                )

            indexed_count = 0
            for file_path in self.docs_dir.rglob("*.md"):
                relative_path = file_path.relative_to(self.docs_dir)
                title, content, headings = self._extract_text_from_markdown(file_path)

                paths.append(str(relative_path))
                titles.append(title)
                contents.append(
                    content[:65535] if len(content) > 65535 else content
                )  # Truncate if too long
                headings_list.append(" ".join(headings)[:5000] if headings else "")

                # Store document text for embeddings
                full_text = f"{title} {' '.join(headings)} {content}"
                documents.append(full_text)

                indexed_count += 1

            # Generate dense embeddings using SentenceTransformerEmbeddingFunction
            if VECTOR_SEARCH_AVAILABLE and self.sentence_transformer_ef and documents:
                embeddings = self.sentence_transformer_ef.encode_documents(documents)

            # Fit BM25 model and generate sparse vectors
            if SPARSE_SEARCH_AVAILABLE and self.bm25_ef and documents:
                self.bm25_ef.fit(documents)
                sparse_vectors_raw = self.bm25_ef.encode_documents(documents)
                
                # Convert sparse vectors to dict format expected by Milvus
                sparse_vectors = []
                for sparse_vec in sparse_vectors_raw:
                    sparse_dict = {}
                    if hasattr(sparse_vec, 'tocoo'):
                        # Convert to COO format first if needed
                        coo = sparse_vec.tocoo()
                        for i, v in zip(coo.col, coo.data):
                            sparse_dict[int(i)] = float(v)
                    else:
                        # Already in COO format
                        for i, v in zip(sparse_vec.col, sparse_vec.data):
                            sparse_dict[int(i)] = float(v)
                    sparse_vectors.append(sparse_dict)

            if indexed_count == 0:
                return {
                    "success": True,
                    "indexed_files": 0,
                    "index_location": str(self.index_dir),
                    "message": "No markdown files found to index",
                }

            # Store in Milvus or file storage
            if self.milvus_connected and self.collection:
                # Clear existing data
                if self.collection.num_entities > 0:
                    self.collection.delete("doc_id >= 0")

                # Insert new data
                entities = [
                    paths,
                    titles,
                    contents,
                    headings_list,
                    embeddings,
                ]

                # Only add sparse vectors if available
                if SPARSE_SEARCH_AVAILABLE and sparse_vectors:
                    entities.append(sparse_vectors)

                self.collection.insert(entities)
                self.collection.flush()
                self.collection.load()

                return {
                    "success": True,
                    "indexed_files": indexed_count,
                    "index_location": str(self.index_dir),
                    "vector_search_available": True,
                    "storage_type": "milvus",
                }
            else:
                # Use file-based storage
                self.embeddings = {}
                self.metadata = {}

                for i, path in enumerate(paths):
                    if i < len(embeddings) if embeddings else 0:
                        self.embeddings[path] = np.array(embeddings[i])

                    self.metadata[path] = {
                        "title": titles[i],
                        "content": contents[i],
                        "headings": headings_list[i],
                    }

                # Save to files
                if self.embeddings:
                    np.savez_compressed(self.embeddings_file, **self.embeddings)

                with open(self.metadata_file, "w") as f:
                    yaml.dump(self.metadata, f)

                return {
                    "success": True,
                    "indexed_files": indexed_count,
                    "index_location": str(self.index_dir),
                    "vector_search_available": bool(self.embeddings),
                    "storage_type": "file",
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def keyword_search(self, query: str, max_results: int = 10) -> list[dict[str, Any]]:
        """Perform keyword-based search using Milvus sparse vectors (BM25) or fallback."""
        # Check if we can use Milvus sparse search
        if SPARSE_SEARCH_AVAILABLE and self.milvus_connected and self.milvus_client:
            try:
                # Check if BM25 model is fitted
                if not self.bm25_ef:
                    # Rebuild index to fit BM25 model
                    self.build_index()

                # Generate sparse vector for the query
                if self.bm25_ef:
                    query_sparse_vec = self.bm25_ef.encode_queries([query])[0]
                    
                    # Convert sparse vector to dict format expected by Milvus
                    # Milvus expects sparse vectors as dict with indices as keys and values as values
                    sparse_dict = {}
                    if hasattr(query_sparse_vec, 'tocoo'):
                        # Convert to COO format first if needed
                        coo = query_sparse_vec.tocoo()
                        for i, v in zip(coo.col, coo.data):
                            sparse_dict[int(i)] = float(v)
                    else:
                        # Already in COO format
                        for i, v in zip(query_sparse_vec.col, query_sparse_vec.data):
                            sparse_dict[int(i)] = float(v)

                    # Search using sparse vector
                    results = self.milvus_client.search(
                        collection_name=self.collection_name,
                        data=[sparse_dict],  # Pass sparse vector as dict
                        anns_field="sparse_vector",
                        limit=max_results,
                        output_fields=["path", "title", "content"],
                        search_params={
                            "metric_type": "IP"
                        },  # Inner Product for sparse vectors
                    )

                    # Format results
                    formatted_results = []
                    if results and len(results) > 0:
                        for hit in results[0]:
                            content = hit["entity"].get("content", "")
                            snippet = (
                                content[:300] + "..." if len(content) > 300 else content
                            )

                            formatted_results.append({
                                "path": hit["entity"].get("path", ""),
                                "title": hit["entity"].get("title", ""),
                                "score": hit[
                                    "distance"
                                ],  # Higher is better for IP metric
                                "snippet": snippet,
                            })

                    return formatted_results

            except Exception as e:
                print(f"Sparse search failed: {e}", file=sys.stderr)

        # Use fallback keyword search
        return self._fallback_keyword_search(query, max_results)

    def _fallback_keyword_search(
        self, query: str, max_results: int = 10
    ) -> list[dict[str, Any]]:
        """Simple fallback keyword search when Milvus sparse search is not available."""
        results = []
        query_lower = query.lower()

        # Search in metadata
        for path, meta in self.metadata.items():
            title = meta.get("title", "").lower()
            content = meta.get("content", "").lower()
            headings = meta.get("headings", "").lower()

            # Simple scoring based on occurrences
            score = 0
            score += title.count(query_lower) * 3  # Title matches are more important
            score += headings.count(query_lower) * 2  # Heading matches are important
            score += content.count(query_lower)

            if score > 0:
                snippet = content[:300] + "..." if len(content) > 300 else content[:300]
                results.append({
                    "path": path,
                    "title": meta.get("title", ""),
                    "score": score,
                    "snippet": snippet,
                })

        # Sort by score and return top results
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:max_results]

    def vector_search(self, query: str, max_results: int = 10) -> list[dict[str, Any]]:
        """Perform semantic vector search."""
        if not VECTOR_SEARCH_AVAILABLE or not self.sentence_transformer_ef:
            return []

        if not self.embeddings:
            self.build_index()

        # Encode query using SentenceTransformerEmbeddingFunction
        query_embeddings = self.sentence_transformer_ef.encode_queries([query])
        query_embedding = np.array(query_embeddings[0])

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


def get_searcher(
    docs_dir: str = "docs", project_root: Optional[Path] = None
) -> DocsSearcher:
    """Get or create the global searcher instance."""
    global _searcher
    if _searcher is None:
        _searcher = DocsSearcher(docs_dir, project_root)
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
        searcher = get_searcher(docs_dir, _project_root)

        # Ensure index is built
        if not hasattr(searcher, "metadata") or (
            not searcher.metadata and not searcher.milvus_connected
        ):
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

        searcher = get_searcher(docs_dir, _project_root)

        if not searcher.sentence_transformer_ef:
            return {
                "success": False,
                "error": "Failed to load the sentence transformer embedding function",
            }

        # Ensure embeddings are built
        if not hasattr(searcher, "embeddings") or (
            not searcher.embeddings and not searcher.milvus_connected
        ):
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
        searcher = get_searcher(docs_dir, _project_root)
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
    global _mkdocs_process, _mkdocs_thread

    if _mkdocs_process:
        try:
            # First try to terminate gracefully
            print("Stopping MkDocs server...", file=sys.stderr)
            _mkdocs_process.terminate()
            
            # Wait for process to terminate
            try:
                _mkdocs_process.wait(timeout=5)
                print("MkDocs server stopped gracefully.", file=sys.stderr)
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't terminate
                print("MkDocs server did not stop gracefully, forcing shutdown...", file=sys.stderr)
                _mkdocs_process.kill()
                _mkdocs_process.wait(timeout=2)
                print("MkDocs server forcefully stopped.", file=sys.stderr)
                
        except Exception as e:
            print(f"Error stopping MkDocs server: {e}", file=sys.stderr)
            # Try to kill the process if it still exists
            try:
                if _mkdocs_process.poll() is None:
                    _mkdocs_process.kill()
            except:
                pass
        finally:
            _mkdocs_process = None
    
    # Also ensure the thread is properly joined
    if _mkdocs_thread and _mkdocs_thread.is_alive():
        try:
            _mkdocs_thread.join(timeout=1)
        except:
            pass
        finally:
            _mkdocs_thread = None


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
    global _searcher, _mkdocs_process

    print("\nShutting down MkDocs RAG Server...", file=sys.stderr)

    # Stop MkDocs server
    stop_mkdocs_serve()

    # Clean up search index and Milvus connection
    if _searcher:
        try:
            # Close Milvus connection if it exists
            if hasattr(_searcher, 'milvus_connected') and _searcher.milvus_connected:
                try:
                    if _searcher.collection:
                        _searcher.collection.release()
                    connections.disconnect("default")
                    print("Milvus connection closed.", file=sys.stderr)
                except Exception as e:
                    print(f"Error closing Milvus connection: {e}", file=sys.stderr)
            
            # Clean up temporary files
            _searcher.cleanup()
            print("Search index cleaned up.", file=sys.stderr)
        except Exception as e:
            print(f"Error during searcher cleanup: {e}", file=sys.stderr)
        finally:
            _searcher = None

    # Final check to ensure MkDocs process is terminated
    if _mkdocs_process:
        try:
            if _mkdocs_process.poll() is None:
                _mkdocs_process.kill()
                print("Ensured MkDocs process termination.", file=sys.stderr)
        except:
            pass

    print("Cleanup complete.", file=sys.stderr)


# Register cleanup
atexit.register(cleanup)


# Handle signals for graceful shutdown
def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    print(f"\nReceived signal {signum}, initiating graceful shutdown...", file=sys.stderr)
    cleanup()
    sys.exit(0)


# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
# On Windows, also handle SIGBREAK
if hasattr(signal, 'SIGBREAK'):
    signal.signal(signal.SIGBREAK, signal_handler)


def main():
    """Main entry point for the MCP server."""
    try:
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
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt, shutting down...", file=sys.stderr)
        cleanup()
        sys.exit(0)
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()
