"""
Tests for help documentation features.

These tests verify that markdown documentation is correctly integrated
into the command-line help text.
"""
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
from typer.testing import CliRunner, Result

# Add project root to Python path if needed
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the app for direct invocation in tests
from inventorytracker.main import app
from inventorytracker.docsgen import inject_docs_to_app


def test_help_contains_markdown_content_cli_runner(
    cli_runner: CliRunner,
    temp_docs_file: Path,
) -> None:
    """
    Test that help text contains content from markdown file using CLI Runner.
    
    This test uses Typer's CliRunner to invoke the app's --help command
    and check that the output contains expected content from our markdown docs.
    
    Args:
        cli_runner: Typer CLI runner fixture
        temp_docs_file: Temporary documentation file
    """
    # Make sure the docs are injected with our test file
    from inventorytracker import docsgen
    # Override the docs path to use our temp file
    docsgen.DEFAULT_DOCS_PATH = temp_docs_file
    
    # Force reload the documentation
    if hasattr(docsgen, "_docs_cache"):
        docsgen._docs_cache.clear()
    if hasattr(docsgen, "_command_docs_cache"):
        docsgen._command_docs_cache.clear()
    
    # Reinject the docs
    inject_docs_to_app(app, temp_docs_file)
    
    # Run the CLI with --help
    result = cli_runner.invoke(app, ["--help"])
    
    # Check the exit code
    assert result.exit_code == 0, f"Command failed with exit code {result.exit_code}, output: {result.stdout}"
    
    # Check for unique content from our markdown in the help output
    assert "unique-test-command" in result.stdout, "Expected unique command not found in help"
    assert "distinctive description" in result.stdout, "Expected command description not found in help"
    
    # Verify that command documentation was properly formatted
    assert "Add a new item to the inventory" in result.stdout, "Expected add-item description not found"
    assert "List all items in the inventory" in result.stdout, "Expected list-items description not found"


def test_help_contains_markdown_content_subprocess(
    temp_docs_file: Path, 
    mock_docs_env: Dict[str, str]
) -> None:
    """
    Test that help text contains content from markdown file using subprocess.
    
    This test invokes the application as a subprocess, which more closely
    simulates how a user would run the command.
    
    Args:
        temp_docs_file: Temporary documentation file
        mock_docs_env: Environment with test docs path
    """
    # Run the CLI as a subprocess with --help
    result = subprocess.run(
        [sys.executable, "-m", "inventorytracker", "--help"],
        capture_output=True,
        text=True,
        env=mock_docs_env,
    )
    
    # Check the exit code
    assert result.returncode == 0, f"Command failed with exit code {result.returncode}, output: {result.stdout}"
    
    # Check for unique content from our markdown in the help output
    assert "unique-test-command" in result.stdout, "Expected unique command not found in help"
    assert "distinctive description" in result.stdout, "Expected command description not found in help"
    
    # Verify that command documentation was properly formatted
    assert "Add a new item to the inventory" in result.stdout, "Expected add-item description not found"
    assert "List all items in the inventory" in result.stdout, "Expected list-items description not found"


def test_help_contains_markdown_content_isolation(
    temp_docs_file: Path,
    captured_output: Tuple[List[str], List[str]]
) -> None:
    """
    Test that help text contains markdown content with complete output isolation.
    
    This test uses a custom stdout/stderr capture mechanism to ensure that
    output is completely isolated from pytest's output capture.
    
    Args:
        temp_docs_file: Temporary documentation file
        captured_output: Fixture that captures and isolates output
    """
    stdout_lines, stderr_lines = captured_output
    
    # Set environment variable for docs path
    os.environ["INVENTORY_DOCS_PATH"] = str(temp_docs_file)
    
    try:
        # Run the module with --help directly, capturing output without letting it pollute tests
        # We use a separate process to ensure complete isolation
        with subprocess.Popen(
            [sys.executable, "-m", "inventorytracker", "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=os.environ,
        ) as proc:
            # Read output
            stdout, stderr = proc.communicate()
            
            # Store in our captured lines
            stdout_lines.extend(stdout.splitlines())
            stderr_lines.extend(stderr.splitlines())
            
            # Check exit code
            assert proc.returncode == 0, f"Command failed with exit code {proc.returncode}"
    finally:
        # Clean up environment
        if "INVENTORY_DOCS_PATH" in os.environ:
            del os.environ["INVENTORY_DOCS_PATH"]
    
    # Check for expected content in captured output
    stdout_text = "\n".join(stdout_lines)
    assert "unique-test-command" in stdout_text, "Expected unique command not found in help"
    assert "distinctive description" in stdout_text, "Expected command description not found in help"
    assert "Add a new item to the inventory" in stdout_text, "Expected add-item description not found"


def test_specific_command_help(
    cli_runner: CliRunner,
    temp_docs_file: Path,
) -> None:
    """
    Test that help for a specific command contains markdown content.
    
    This checks that command-specific help (invtrack command --help)
    correctly includes documentation from the markdown file.
    
    Args:
        cli_runner: Typer CLI runner fixture
        temp_docs_file: Temporary documentation file
    """
    # Make sure the docs are injected with our test file
    from inventorytracker import docsgen
    docsgen.DEFAULT_DOCS_PATH = temp_docs_file
    
    # Force reload the documentation
    if hasattr(docsgen, "_docs_cache"):
        docsgen._docs_cache.clear()
    if hasattr(docsgen, "_command_docs_cache"):
        docsgen._command_docs_cache.clear()
    
    # Reinject the docs
    inject_docs_to_app(app, temp_docs_file)
    
    # Run the CLI for specific command help
    result = cli_runner.invoke(app, ["list-items", "--help"])
    
    # Check the exit code
    assert result.exit_code == 0, f"Command failed with exit code {result.exit_code}, output: {result.stdout}"
    
    # Verify command-specific help contains the markdown content
    assert "List all items in the inventory" in result.stdout, "Expected command description not found in help"
    assert "filtered by category" in result.stdout, "Expected command details not found in help"
    assert "Parameters:" in result.stdout, "Expected parameters section not found in help"
    assert "category" in result.stdout, "Expected parameter name not found in help"


def test_help_formatting(
    cli_runner: CliRunner,
    temp_docs_file: Path,
) -> None:
    """
    Test that markdown formatting is properly converted to terminal-friendly text.
    
    This test checks that markdown formatting elements like backticks, links,
    and formatting are properly handled in the terminal output.
    
    Args:
        cli_runner: Typer CLI runner fixture
        temp_docs_file: Temporary documentation file
    """
    # Make sure the docs are injected with our test file
    from inventorytracker import docsgen
    docsgen.DEFAULT_DOCS_PATH = temp_docs_file
    
    # Force reload the documentation
    if hasattr(docsgen, "_docs_cache"):
        docsgen._docs_cache.clear()
    if hasattr(docsgen, "_command_docs_cache"):
        docsgen._command_docs_cache.clear()
    
    # Reinject the docs
    inject_docs_to_app(app, temp_docs_file)
    
    # Run the CLI with --help
    result = cli_runner.invoke(app, ["--help"])
    
    # Check for proper formatting of markdown elements
    assert "**name**" not in result.stdout, "Bold markdown was not converted properly"
    assert "The name of the item" in result.stdout, "Parameter description not found in help"
    
    # Check for proper handling of code examples - they should be present but without markdown formatting
    assert "```" not in result.stdout, "Code block markers should not appear in help text"
    assert "invtrack" in result.stdout, "Code example content should be present"


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v"])