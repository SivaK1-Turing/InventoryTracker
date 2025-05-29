#!/usr/bin/env python3
"""
Documentation generator for Inventory Tracker.

This module parses Markdown documentation in docs/commands.md and
injects the documentation into Typer commands' help text at import time.

Features:
- Robust mapping of Markdown headers to command names
- Support for nested commands and command groups
- Markdown formatting conversion to terminal-friendly text
- Automatic inclusion of examples in help text
- Support for command aliases and multi-word commands
"""
import functools
import importlib
import inspect
import os
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Callable, Any, Union, TypeVar

import typer
from typer.models import CommandInfo, ParameterInfo

from inventorytracker.logging import logger


# Default documentation path
DEFAULT_DOCS_PATH = Path("docs/commands.md")

# Cache for parsed documentation to avoid re-parsing
_docs_cache: Dict[str, str] = {}
_command_docs_cache: Dict[str, "CommandDoc"] = {}


@dataclass
class CommandDoc:
    """
    Parsed command documentation.
    
    Attributes:
        name: Command name
        description: Full command description
        summary: Short summary (first paragraph)
        examples: List of usage examples
        notes: Additional notes about the command
        parameters: Documentation for specific parameters
    """
    name: str
    description: str = ""
    summary: str = ""
    examples: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    parameters: Dict[str, str] = field(default_factory=dict)
    
    @property
    def help_text(self) -> str:
        """
        Format the documentation as help text for Typer.
        
        Returns:
            Formatted help text
        """
        if not self.description:
            return ""
        
        # Start with the full description
        text = self.description.strip()
        
        # Add parameter documentation if available
        if self.parameters:
            text += "\n\nParameters:\n"
            for param, desc in self.parameters.items():
                text += f"  {param}: {desc.strip()}\n"
        
        # Add examples if available
        if self.examples:
            text += "\n\nExamples:\n"
            for example in self.examples:
                text += f"{example.strip()}\n"
        
        # Add notes if available
        if self.notes:
            text += "\n\nNotes:\n"
            for note in self.notes:
                text += f"{note.strip()}\n"
        
        return text


class DocSection(Enum):
    """Sections within command documentation."""
    DESCRIPTION = "description"
    EXAMPLES = "examples"
    NOTES = "notes"
    PARAMETERS = "parameters"
    UNKNOWN = "unknown"


def normalize_command_name(name: str) -> str:
    """
    Normalize command name for consistent matching.
    
    This converts dashes to underscores and removes spaces.
    
    Args:
        name: Command name to normalize
        
    Returns:
        Normalized command name
    """
    # Remove any 'command' suffix that might be in the docs but not the function name
    name = re.sub(r'(?i)\s*command$', '', name)
    
    # Replace dashes, spaces, and dots with underscores
    name = re.sub(r'[\-\s\.]+', '_', name)
    
    # Handle common naming patterns in commands vs functions
    name = name.replace('_command', '')
    
    # Remove any remaining non-alphanumeric characters
    name = re.sub(r'[^a-zA-Z0-9_]', '', name)
    
    # Convert to lowercase for case-insensitive matching
    return name.lower()


def extract_command_info(header: str) -> Tuple[str, int]:
    """
    Extract command name and level from markdown header.
    
    Args:
        header: Markdown header line
        
    Returns:
        Tuple of (command_name, header_level)
    """
    # Match markdown headers (e.g., '## Command Name' or '### Subcommand Name')
    match = re.match(r'^(#+)\s+(.*?)(?:\s*$)', header)
    if not match:
        return "", 0
    
    level = len(match.group(1))
    command_name = match.group(2).strip()
    
    # Remove any command suffix for cleaner names
    command_name = re.sub(r'(?i)\s+Command$', '', command_name)
    
    return command_name, level


def detect_aliases(command_name: str) -> List[str]:
    """
    Detect possible command name aliases from documentation.
    
    Looks for patterns like "command-name (alias: cmd)" or similar.
    
    Args:
        command_name: Raw command name from documentation
        
    Returns:
        List of potential command names (primary and aliases)
    """
    # Check for explicit alias notation
    alias_match = re.search(r'(?i)\((?:alias(?:es)?:?\s*)(.*?)\)', command_name)
    if alias_match:
        # Extract the base command name without the alias part
        base_name = command_name.split('(')[0].strip()
        # Get all aliases
        aliases = [a.strip() for a in alias_match.group(1).split(',')]
        return [base_name] + aliases
    
    # If no explicit aliases, just return the original name
    return [command_name]


def handle_multi_word_commands(normalized_names: List[str]) -> List[str]:
    """
    Handle multi-word command names in various formats.
    
    This covers cases like "add-item", "add_item", "add item" that all map to
    "add_item" in Python code.
    
    Args:
        normalized_names: List of already normalized names
        
    Returns:
        Extended list of potential normalized names
    """
    extended_names = normalized_names.copy()
    
    for name in normalized_names:
        # Handle kebab-case → snake_case
        if '-' in name:
            extended_names.append(name.replace('-', '_'))
        
        # Handle space-separated → snake_case
        if ' ' in name:
            extended_names.append(name.replace(' ', '_'))
        
        # Handle camelCase → snake_case
        if any(c.isupper() for c in name):
            snake_case = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
            extended_names.append(snake_case)
    
    return list(set(extended_names))  # Deduplicate


def parse_command_docs(md_content: str) -> Dict[str, CommandDoc]:
    """
    Parse command documentation from Markdown content.
    
    Args:
        md_content: Markdown document content
        
    Returns:
        Dictionary mapping command names to CommandDoc objects
    """
    # Check if we've already parsed this content
    content_hash = hash(md_content)
    if content_hash in _docs_cache:
        return _command_docs_cache
    
    # Store hash to avoid re-parsing identical content
    _docs_cache[content_hash] = md_content
    
    # Split the document by headers
    current_section = None
    current_command = None
    current_param = None
    current_docs: Dict[str, CommandDoc] = {}
    
    # Process line by line
    lines = md_content.split('\n')
    section_text = []
    
    for i, line in enumerate(lines):
        # Check if this is a header line
        if line.strip().startswith('#'):
            # Process accumulated text from previous section if any
            if current_command and section_text:
                text = '\n'.join(section_text).strip()
                
                if current_section == DocSection.DESCRIPTION:
                    current_command.description = text
                    # First paragraph as summary
                    current_command.summary = text.split('\n\n')[0] if '\n\n' in text else text
                elif current_section == DocSection.EXAMPLES:
                    current_command.examples.append(text)
                elif current_section == DocSection.NOTES:
                    current_command.notes.append(text)
                elif current_section == DocSection.PARAMETERS and current_param:
                    current_command.parameters[current_param] = text
            
            # Reset for new section
            section_text = []
            
            # Extract command name and level from header
            command_name, level = extract_command_info(line)
            
            if level <= 2:
                # Top-level command - create new CommandDoc
                # First, detect any aliases in the command name
                potential_names = detect_aliases(command_name)
                
                # Then process each name to handle multi-word formats
                normalized_names = []
                for name in potential_names:
                    normalized = normalize_command_name(name)
                    normalized_names.extend(handle_multi_word_commands([normalized]))
                
                # Create CommandDoc for the primary name
                current_command = CommandDoc(name=command_name)
                
                # Store under all potential normalized names for robust matching
                for normalized in normalized_names:
                    if normalized:
                        current_docs[normalized] = current_command
                
                current_section = DocSection.DESCRIPTION
                current_param = None
            elif level == 3 and current_command:
                # Subsection of a command
                section_title = command_name.lower()
                if "example" in section_title:
                    current_section = DocSection.EXAMPLES
                elif "note" in section_title or "warning" in section_title:
                    current_section = DocSection.NOTES
                elif "parameter" in section_title or "argument" in section_title or "option" in section_title:
                    current_section = DocSection.PARAMETERS
                    # Reset current parameter
                    current_param = None
                else:
                    # Might be a parameter name
                    current_section = DocSection.PARAMETERS
                    current_param = normalize_command_name(command_name)
            else:
                # Unknown header level or no current command
                current_section = DocSection.UNKNOWN
        elif current_command and current_section == DocSection.PARAMETERS and line.strip().startswith('-'):
            # This looks like a parameter list item
            param_match = re.match(r'\s*[\-\*]\s+`?([\w\-]+)`?(?:[\s\:]+)(.*)', line)
            if param_match:
                # Complete previous parameter
                if current_param and section_text:
                    current_command.parameters[current_param] = '\n'.join(section_text).strip()
                
                # Start new parameter
                current_param = param_match.group(1).strip()
                section_text = [param_match.group(2).strip()]
            else:
                # Not a parameter definition, just add to current section
                section_text.append(line)
        else:
            # Regular content line - add to current section
            if current_command and current_section != DocSection.UNKNOWN:
                section_text.append(line)
    
    # Process any remaining content
    if current_command and section_text:
        text = '\n'.join(section_text).strip()
        
        if current_section == DocSection.DESCRIPTION:
            current_command.description = text
            current_command.summary = text.split('\n\n')[0] if '\n\n' in text else text
        elif current_section == DocSection.EXAMPLES:
            current_command.examples.append(text)
        elif current_section == DocSection.NOTES:
            current_command.notes.append(text)
        elif current_section == DocSection.PARAMETERS and current_param:
            current_command.parameters[current_param] = text
    
    # Store in cache
    _command_docs_cache.update(current_docs)
    
    return current_docs


def load_docs(docs_path: Optional[Path] = None) -> Dict[str, CommandDoc]:
    """
    Load and parse documentation from file.
    
    Args:
        docs_path: Path to commands.md file (or uses default if None)
        
    Returns:
        Dictionary mapping command names to CommandDoc objects
    """
    if not docs_path:
        # Use default path relative to project root
        # First try to find project root
        current_dir = Path.cwd()
        while current_dir != current_dir.parent:
            if (current_dir / "docs" / "commands.md").exists():
                docs_path = current_dir / "docs" / "commands.md"
                break
            current_dir = current_dir.parent
        
        # If not found, try relative to this file
        if not docs_path:
            module_dir = Path(__file__).parent.parent
            docs_path = module_dir / "docs" / "commands.md"
    
    try:
        if docs_path and docs_path.exists():
            with open(docs_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return parse_command_docs(content)
        else:
            logger.warning(f"Documentation file not found: {docs_path}")
            return {}
    except Exception as e:
        logger.error(f"Error loading command documentation: {e}")
        return {}


def clean_formatting(text: str) -> str:
    """
    Clean Markdown formatting to be more terminal-friendly.
    
    Args:
        text: Text with Markdown formatting
        
    Returns:
        Cleaned text for terminal display
    """
    if not text:
        return ""
    
    # Remove backticks used for code formatting
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Convert bold to uppercase
    text = re.sub(r'\*\*([^*]+)\*\*', lambda m: m.group(1).upper(), text)
    
    # Convert headers to uppercase
    text = re.sub(r'^#+\s+(.*?)$', lambda m: m.group(1).upper(), text, flags=re.MULTILINE)
    
    # Convert inline links to just the text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    
    # Convert Markdown lists to plain text lists
    text = re.sub(r'^\s*[\-\*]\s+', '• ', text, flags=re.MULTILINE)
    
    return text


def apply_doc_to_command(command_info: CommandInfo, doc: CommandDoc) -> None:
    """
    Apply documentation to a Typer command.
    
    Args:
        command_info: Typer CommandInfo object
        doc: Parsed command documentation
    """
    # Update help text if not explicitly set in the code
    if command_info.help is None or command_info.help == "":
        help_text = clean_formatting(doc.help_text)
        command_info.help = help_text
    
    # Update parameter help if not explicitly set
    for param_info in command_info.params:
        param_name = param_info.name
        if param_name in doc.parameters and (param_info.help is None or param_info.help == ""):
            param_info.help = clean_formatting(doc.parameters[param_name])


def find_commands_in_typer_app(app: typer.Typer) -> Dict[str, CommandInfo]:
    """
    Find all commands in a Typer app.
    
    Args:
        app: Typer app instance
        
    Returns:
        Dictionary mapping command names to CommandInfo objects
    """
    commands = {}
    
    # Get commands from the app's callback
    if hasattr(app, "registered_commands"):
        for command in app.registered_commands:
            normalized_name = normalize_command_name(command.name)
            commands[normalized_name] = command
    
    # Get commands from subcommands
    if hasattr(app, "registered_groups"):
        for group in app.registered_groups:
            if hasattr(group, "typer_instance"):
                # Recursively process subcommands
                subcommands = find_commands_in_typer_app(group.typer_instance)
                for subname, subcmd in subcommands.items():
                    # For subcommands, we prefix with the group name
                    full_name = f"{normalize_command_name(group.name)}_{subname}"
                    commands[full_name] = subcmd
                    # Also store just the subcommand name for alternative matching
                    commands[subname] = subcmd
    
    return commands


def inject_docs_to_app(app: typer.Typer, docs_path: Optional[Path] = None) -> None:
    """
    Inject documentation from Markdown to all commands in a Typer app.
    
    Args:
        app: Typer app instance
        docs_path: Path to commands.md file (or uses default if None)
    """
    try:
        # Load command documentation
        command_docs = load_docs(docs_path)
        if not command_docs:
            logger.warning("No command documentation found to inject")
            return
        
        # Find all commands in the app
        app_commands = find_commands_in_typer_app(app)
        if not app_commands:
            logger.warning("No commands found in Typer app")
            return
        
        # Match and apply documentation
        matched_count = 0
        for cmd_name, cmd_info in app_commands.items():
            if cmd_name in command_docs:
                apply_doc_to_command(cmd_info, command_docs[cmd_name])
                matched_count += 1
            elif cmd_info.callback and hasattr(cmd_info.callback, "__name__"):
                # Try matching by function name
                func_name = normalize_command_name(cmd_info.callback.__name__)
                if func_name in command_docs:
                    apply_doc_to_command(cmd_info, command_docs[func_name])
                    matched_count += 1
        
        logger.debug(f"Injected documentation for {matched_count} of {len(app_commands)} commands")
    except Exception as e:
        logger.error(f"Error injecting command documentation: {e}")


def create_docs_decorator(docs_path: Optional[Path] = None):
    """
    Create a decorator for applying documentation to a Typer app.
    
    Args:
        docs_path: Path to commands.md file
        
    Returns:
        Decorator function
    """
    def decorator(app: typer.Typer) -> typer.Typer:
        inject_docs_to_app(app, docs_path)
        return app
    
    return decorator


# Create default decorator
apply_docs = create_docs_decorator()


def generate_stub_docs(app: typer.Typer) -> str:
    """
    Generate stub documentation for commands in a Typer app.
    
    This is useful for creating an initial commands.md template.
    
    Args:
        app: Typer app instance
        
    Returns:
        Markdown content for commands.md
    """
    commands = find_commands_in_typer_app(app)
    
    md_content = ["# Inventory Tracker Commands\n\n"]
    
    for cmd_name, cmd_info in sorted(commands.items()):
        # Skip duplicates
        if "_" in cmd_name and cmd_name.split("_", 1)[1] in commands:
            continue
        
        # Get command description from docstring
        cmd_desc = ""
        if cmd_info.callback and cmd_info.callback.__doc__:
            cmd_desc = inspect.getdoc(cmd_info.callback) or ""
        
        # Use command help if available
        if not cmd_desc and cmd_info.help:
            cmd_desc = cmd_info.help
        
        # Remove any doctest blocks
        cmd_desc = re.sub(r'>>>.*?(\n\n|$)', '', cmd_desc, flags=re.DOTALL)
        
        # Command header
        display_name = cmd_info.name.replace("_", "-")
        md_content.append(f"## {display_name}\n")
        
        # Command description
        if cmd_desc:
            md_content.append(f"{cmd_desc.strip()}\n")
        else:
            md_content.append("*No description available*\n")
        
        # Parameters
        if cmd_info.params:
            md_content.append("### Parameters\n")
            for param in cmd_info.params:
                param_help = param.help or "*No description*"
                md_content.append(f"- **{param.name}**: {param_help}\n")
        
        # Example placeholder
        md_content.append("### Example\n")
        md_content.append(f"```\ninvtrack {cmd_info.name} [arguments]\n```\n")
    
    return "\n".join(md_content)


def create_commands_md(app: typer.Typer, output_path: Optional[Path] = None) -> None:
    """
    Create a commands.md file with stub documentation.
    
    Args:
        app: Typer app instance
        output_path: Path to write the file (or uses default if None)
    """
    if not output_path:
        output_path = Path.cwd() / "docs" / "commands.md"
    
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate documentation
    md_content = generate_stub_docs(app)
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"Created command documentation stub at {output_path}")


if __name__ == "__main__":
    # Example usage to generate documentation
    from inventorytracker.main import app as inventory_app
    
    if len(sys.argv) > 1 and sys.argv[1] == "--generate":
        # Generate stub documentation
        output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
        create_commands_md(inventory_app, output_path)
    else:
        # Test documentation injection
        inject_docs_to_app(inventory_app)
        print("Documentation injected into command help")