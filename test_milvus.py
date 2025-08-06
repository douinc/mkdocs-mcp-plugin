#!/usr/bin/env python
"""Test script to verify PyMilvus integration."""

import sys
import json
from pathlib import Path

# Import the server module
import server

def test_initialization():
    """Test initialization of DocsSearcher with PyMilvus."""
    print("Testing DocsSearcher initialization...")
    
    # Use test project directory
    test_project = Path("test_mkdocs_project")
    if not test_project.exists():
        print(f"Error: Test project directory {test_project} does not exist")
        return False
    
    docs_dir = test_project / "docs"
    searcher = server.DocsSearcher(str(docs_dir), test_project)
    
    print(f"✓ DocsSearcher initialized")
    print(f"  - Index directory: {searcher.index_dir}")
    print(f"  - Milvus connected: {searcher.milvus_connected}")
    print(f"  - Embedding function loaded: {searcher.sentence_transformer_ef is not None}")
    
    return True

def test_build_index():
    """Test building the search index."""
    print("\nTesting index building...")
    
    test_project = Path("test_mkdocs_project")
    docs_dir = test_project / "docs"
    searcher = server.DocsSearcher(str(docs_dir), test_project)
    
    result = searcher.build_index()
    print(f"Build index result: {json.dumps(result, indent=2)}")
    
    if result.get("success"):
        print(f"✓ Index built successfully")
        print(f"  - Files indexed: {result.get('indexed_files')}")
        print(f"  - Storage type: {result.get('storage_type', 'file')}")
        print(f"  - Vector search available: {result.get('vector_search_available')}")
        return True
    else:
        print(f"✗ Index build failed: {result.get('error')}")
        return False

def test_keyword_search():
    """Test keyword-based search."""
    print("\nTesting keyword search...")
    
    test_project = Path("test_mkdocs_project")
    docs_dir = test_project / "docs"
    searcher = server.DocsSearcher(str(docs_dir), test_project)
    
    # Build index first
    searcher.build_index()
    
    # Test search
    results = searcher.keyword_search("mkdocs", max_results=5)
    
    print(f"Found {len(results)} results for 'mkdocs'")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['title']} (score: {result['score']:.2f})")
        print(f"     Path: {result['path']}")
    
    return len(results) > 0

def test_vector_search():
    """Test vector-based semantic search."""
    print("\nTesting vector search...")
    
    test_project = Path("test_mkdocs_project")
    docs_dir = test_project / "docs"
    searcher = server.DocsSearcher(str(docs_dir), test_project)
    
    # Build index first
    searcher.build_index()
    
    # Test search
    results = searcher.vector_search("documentation configuration", max_results=5)
    
    print(f"Found {len(results)} results for 'documentation configuration'")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['title']} (score: {result['score']:.3f})")
        print(f"     Path: {result['path']}")
    
    return True

def test_persistent_index():
    """Test that index is persistent in .mkdocs_vector directory."""
    print("\nTesting index persistence...")
    
    test_project = Path("test_mkdocs_project")
    index_dir = test_project / ".mkdocs_vector"
    
    print(f"Checking for index directory: {index_dir}")
    if index_dir.exists():
        print(f"✓ Index directory exists")
        
        # Check for index files
        files = list(index_dir.iterdir())
        print(f"  Files in index directory:")
        for file in files:
            print(f"    - {file.name}")
        
        return True
    else:
        print(f"✗ Index directory does not exist")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing PyMilvus Integration for MkDocs MCP Plugin")
    print("=" * 60)
    
    tests = [
        ("Initialization", test_initialization),
        ("Build Index", test_build_index),
        ("Keyword Search", test_keyword_search),
        ("Vector Search", test_vector_search),
        ("Persistent Index", test_persistent_index),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ {name} failed with error: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print("=" * 60)
    
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{name:20} {status}")
    
    all_passed = all(success for _, success in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed.")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())