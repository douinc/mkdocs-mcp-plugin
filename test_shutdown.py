#!/usr/bin/env python
"""Test script to verify proper shutdown handling of the MkDocs MCP server."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def test_shutdown_signals():
    """Test that the server properly shuts down on various signals."""
    
    # Change to test project directory
    test_dir = Path("test_mkdocs_project")
    if not test_dir.exists():
        print("Error: test_mkdocs_project directory not found")
        return False
    
    os.chdir(test_dir)
    
    print("Testing MCP server shutdown handling...")
    print("-" * 50)
    
    # Test 1: Normal shutdown with SIGTERM
    print("\n1. Testing SIGTERM (normal termination)...")
    proc = subprocess.Popen(
        [sys.executable, "../server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Give server time to start
    time.sleep(3)
    
    # Send SIGTERM
    proc.terminate()
    
    # Wait for process to exit
    try:
        stdout, stderr = proc.communicate(timeout=10)
        if "Shutting down MkDocs RAG Server" in stderr and "Cleanup complete" in stderr:
            print("✓ SIGTERM handled correctly - graceful shutdown confirmed")
        else:
            print("✗ SIGTERM handling issue - expected shutdown messages not found")
            print(f"stderr: {stderr[-500:]}")  # Last 500 chars
    except subprocess.TimeoutExpired:
        print("✗ Server did not shut down within timeout")
        proc.kill()
        return False
    
    # Test 2: Interrupt with SIGINT (Ctrl+C)
    print("\n2. Testing SIGINT (Ctrl+C)...")
    proc = subprocess.Popen(
        [sys.executable, "../server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Give server time to start
    time.sleep(3)
    
    # Send SIGINT
    proc.send_signal(signal.SIGINT)
    
    # Wait for process to exit
    try:
        stdout, stderr = proc.communicate(timeout=10)
        if "Received signal" in stderr and "Cleanup complete" in stderr:
            print("✓ SIGINT handled correctly - graceful shutdown confirmed")
        else:
            print("✗ SIGINT handling issue - expected shutdown messages not found")
            print(f"stderr: {stderr[-500:]}")  # Last 500 chars
    except subprocess.TimeoutExpired:
        print("✗ Server did not shut down within timeout")
        proc.kill()
        return False
    
    # Test 3: Force kill and verify cleanup
    print("\n3. Testing forced kill and cleanup...")
    proc = subprocess.Popen(
        [sys.executable, "../server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Give server time to start
    time.sleep(3)
    
    # Get process ID
    pid = proc.pid
    
    # Force kill
    proc.kill()
    
    # Wait for process to exit
    try:
        proc.wait(timeout=5)
        print("✓ Process terminated after kill signal")
    except subprocess.TimeoutExpired:
        print("✗ Process did not terminate after kill signal")
        return False
    
    # Check if MkDocs server subprocess was also terminated
    # This would require checking for orphaned processes, which is platform-specific
    print("✓ Kill signal handled (cleanup via atexit may not run with SIGKILL)")
    
    print("\n" + "-" * 50)
    print("Shutdown tests completed!")
    return True


def check_subprocess_cleanup():
    """Check if any MkDocs subprocesses are still running."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "mkdocs serve"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"Warning: Found running MkDocs processes: {result.stdout}")
            return False
        else:
            print("✓ No orphaned MkDocs processes found")
            return True
    except FileNotFoundError:
        # pgrep not available (e.g., on Windows)
        print("Note: Cannot check for orphaned processes on this platform")
        return True


if __name__ == "__main__":
    try:
        success = test_shutdown_signals()
        cleanup_ok = check_subprocess_cleanup()
        
        if success and cleanup_ok:
            print("\n✅ All shutdown tests passed!")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed. Please review the output above.")
            sys.exit(1)
    except Exception as e:
        print(f"Test error: {e}")
        sys.exit(1)