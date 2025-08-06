#!/usr/bin/env python
"""Integration test for MkDocs MCP Plugin."""

import json
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_functionality():
    """Test basic server functionality."""
    print("Testing MkDocs MCP Plugin Integration...")
    print("-" * 50)
    
    # Change to test project directory
    test_project = Path(__file__).parent / "test_mkdocs_project"
    if not test_project.exists():
        print("❌ Test project directory not found")
        return False
    
    os.chdir(test_project)
    print(f"✅ Changed to test project: {test_project}")
    
    # Import server module
    try:
        from server import (
            find_mkdocs_config,
            load_mkdocs_config,
            DocsSearcher,
            get_mkdocs_info
        )
        print("✅ Server module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import server module: {e}")
        return False
    
    # Test MkDocs config detection
    config_path = find_mkdocs_config()
    if config_path:
        print(f"✅ Found MkDocs config: {config_path.name}")
        config = load_mkdocs_config(config_path)
        print(f"   Site name: {config.get('site_name', 'Unknown')}")
    else:
        print("❌ MkDocs config not found")
        return False
    
    # Test document search
    searcher = DocsSearcher("docs")
    index_result = searcher.build_index()
    if index_result.get("success"):
        print(f"✅ Search index built: {index_result.get('indexed_files')} files indexed")
    else:
        print(f"❌ Failed to build index: {index_result.get('error')}")
        return False
    
    # Test keyword search
    search_results = searcher.keyword_search("authentication")
    print(f"✅ Keyword search completed: {len(search_results)} results found")
    if search_results:
        first_result = search_results[0]
        print(f"   Top result: {first_result.get('title')} (score: {first_result.get('score', 0):.2f})")
    
    # Test vector search if available
    if index_result.get("vector_search_available"):
        vector_results = searcher.vector_search("getting started")
        print(f"✅ Vector search completed: {len(vector_results)} results found")
    else:
        print("ℹ️  Vector search not available (optional dependencies not installed)")
    
    # Clean up
    searcher.cleanup()
    print("✅ Cleanup completed")
    
    print("-" * 50)
    print("✅ All tests passed!")
    return True

def test_mcp_tools():
    """Test MCP tool definitions."""
    print("\nTesting MCP Tool Definitions...")
    print("-" * 50)
    
    try:
        # Import the decorated functions directly
        import server
        
        # List expected tools
        expected_tools = [
            'read_document',
            'list_documents',
            'keyword_search',
            'vector_search',
            'search',
            'rebuild_search_index',
            'get_mkdocs_info',
            'restart_mkdocs_server'
        ]
        
        # Check if tools exist
        found_tools = []
        for tool_name in expected_tools:
            if hasattr(server, tool_name):
                found_tools.append(tool_name)
        
        print(f"✅ Found {len(found_tools)}/{len(expected_tools)} MCP tools:")
        for tool in sorted(found_tools):
            print(f"   - {tool}")
        
        # Check for MCP resource
        if hasattr(server, 'get_documents_info'):
            print("✅ Found MCP resource: mkdocs://documents")
        
        # Check for MCP prompt
        if hasattr(server, 'mkdocs_rag_search'):
            print("✅ Found MCP prompt: mkdocs-rag-search")
        
        print("-" * 50)
        return len(found_tools) == len(expected_tools)
        
    except Exception as e:
        print(f"❌ Error testing MCP tools: {e}")
        return False

if __name__ == "__main__":
    success = True
    
    # Run basic functionality tests
    if not test_basic_functionality():
        success = False
    
    # Run MCP tool tests
    if not test_mcp_tools():
        success = False
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)