# inventorytracker/utils/doc_generator.py
import ast
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import inspect
import importlib.util
import logging

# Configure logger
logger = logging.getLogger(__name__)

class PromptInfo:
    """Information about a prompt extracted from code."""
    
    def __init__(self, prompt_text: str, validation_func: Optional[str] = None, 
                 description: Optional[str] = None, field_name: Optional[str] = None,
                 default_value: Optional[str] = None, line_number: int = 0):
        self.prompt_text = prompt_text
        self.validation_func = validation_func
        self.description = description
        self.field_name = field_name
        self.default_value = default_value
        self.line_number = line_number
        
    def __repr__(self) -> str:
        return f"PromptInfo(prompt_text='{self.prompt_text}', validation_func='{self.validation_func}', line={self.line_number})"


class ValidationInfo:
    """Information about a validation function."""
    
    def __init__(self, name: str, docstring: Optional[str] = None, 
                 error_messages: Optional[List[str]] = None,
                 validation_type: Optional[str] = None):
        self.name = name
        self.docstring = docstring
        self.error_messages = error_messages or []
        self.validation_type = validation_type
        
    def __repr__(self) -> str:
        return f"ValidationInfo(name='{self.name}', validation_type='{self.validation_type}')"


class PromptExtractor:
    """Extract prompt information from Typer CLI code using AST."""
    
    def __init__(self, module_path: str):
        """
        Initialize the prompt extractor.
        
        Args:
            module_path: Path to the module file to analyze
        """
        self.module_path = module_path
        self.tree = None
        self.prompts: List[PromptInfo] = []
        self.validations: Dict[str, ValidationInfo] = {}
        self.module_source = ""
        
    def parse(self) -> None:
        """Parse the module file and extract prompt and validation information."""
        try:
            with open(self.module_path, "r", encoding="utf-8") as f:
                self.module_source = f.read()
                
            # Parse the source code into an AST
            self.tree = ast.parse(self.module_source)
            
            # Extract validation functions first
            self._extract_validation_functions()
            
            # Then extract prompts
            self._extract_prompts()
            
            # Associate prompts with validations
            self._associate_prompt_descriptions()
            
            # Sort prompts by line number
            self.prompts.sort(key=lambda p: p.line_number)
            
            logger.info(f"Extracted {len(self.prompts)} prompts and {len(self.validations)} validation functions")
            
        except Exception as e:
            logger.error(f"Error parsing {self.module_path}: {e}")
            raise
    
    def _extract_validation_functions(self) -> None:
        """Extract validation function information from the AST."""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("validate_"):
                # Get docstring
                docstring = ast.get_docstring(node)
                
                # Extract error messages from raise statements
                error_messages = []
                for sub_node in ast.walk(node):
                    if isinstance(sub_node, ast.Raise) and isinstance(sub_node.exc, ast.Call):
                        if isinstance(sub_node.exc.func, ast.Name) and sub_node.exc.func.id == "typer":
                            continue  # Skip typer imports
                        
                        # Try to extract the error message string
                        for arg in sub_node.exc.args:
                            if isinstance(arg, ast.Str):
                                error_messages.append(arg.s)
                            elif isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                                error_messages.append(arg.value)
                
                # Determine validation type from function name and parameters
                validation_type = node.name.replace("validate_", "")
                
                self.validations[node.name] = ValidationInfo(
                    name=node.name,
                    docstring=docstring,
                    error_messages=error_messages,
                    validation_type=validation_type
                )
    
    def _extract_prompts(self) -> None:
        """Extract prompt calls from the AST."""
        for node in ast.walk(self.tree):
            # Look for calls to the prompt_with_validation function
            if (isinstance(node, ast.Call) and 
                isinstance(node.func, ast.Name) and 
                node.func.id == "prompt_with_validation"):
                
                # Extract the prompt text
                if len(node.args) >= 1:
                    prompt_text = self._extract_string_value(node.args[0])
                else:
                    prompt_text = "Unknown prompt text"
                
                # Extract validation function
                validation_func = None
                if len(node.args) >= 2:
                    if isinstance(node.args[1], ast.Name):
                        validation_func = node.args[1].id
                
                # Extract default value
                default_value = None
                for keyword in node.keywords:
                    if keyword.arg == "default":
                        default_value = self._extract_string_value(keyword.value)
                
                # Extract field name from the variable assignment, if present
                field_name = self._extract_assignment_target(node)
                
                # Get line number
                line_number = getattr(node, "lineno", 0)
                
                prompt_info = PromptInfo(
                    prompt_text=prompt_text,
                    validation_func=validation_func,
                    field_name=field_name,
                    default_value=default_value,
                    line_number=line_number
                )
                self.prompts.append(prompt_info)
    
    def _extract_string_value(self, node) -> str:
        """Extract string value from an AST node."""
        if isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        elif isinstance(node, ast.Name):
            # For variable references, try to find the variable definition
            return f"{{{node.id}}}"  # Placeholder for variable reference
        elif isinstance(node, ast.JoinedStr):  # f-string
            parts = []
            for value in node.values:
                if isinstance(value, ast.Str) or isinstance(value, ast.Constant):
                    try:
                        parts.append(value.s if hasattr(value, 's') else value.value)
                    except AttributeError:
                        parts.append(str(value))
                elif isinstance(value, ast.FormattedValue):
                    parts.append(f"{{{self._extract_string_value(value.value)}}}")
            return "".join(parts)
        else:
            return f"<expression: {ast.dump(node)}>"
    
    def _extract_assignment_target(self, call_node) -> Optional[str]:
        """Extract the target variable name from an assignment statement."""
        # Walk up the tree to find the assignment statement this call is part of
        for parent_node in ast.walk(self.tree):
            if isinstance(parent_node, ast.Assign) and any(
                self._is_same_node(call_node, value) for value in parent_node.targets
            ):
                if len(parent_node.targets) == 1 and isinstance(parent_node.targets[0], ast.Name):
                    return parent_node.targets[0].id
        return None
    
    def _is_same_node(self, node1, node2) -> bool:
        """Check if two AST nodes are the same."""
        # This is a simplistic check
        return (hasattr(node1, 'lineno') and hasattr(node2, 'lineno') and 
                node1.lineno == node2.lineno)
    
    def _associate_prompt_descriptions(self) -> None:
        """Associate prompts with their validation function descriptions."""
        for prompt in self.prompts:
            if prompt.validation_func in self.validations:
                validation_info = self.validations[prompt.validation_func]
                prompt.description = validation_info.docstring
                
                # Determine field name from validation type if not already set
                if not prompt.field_name and validation_info.validation_type:
                    prompt.field_name = validation_info.validation_type

    def generate_markdown(self) -> str:
        """Generate markdown documentation from the extracted prompt information."""
        if not self.prompts:
            return "No prompts found in the module."
            
        lines = []
        lines.append("# Command Usage Guide")
        lines.append("")
        lines.append("## Product Management Commands")
        lines.append("")
        lines.append("### Adding a Product")
        lines.append("")
        lines.append("To add a new product to the inventory, use the `product add` command:")
        lines.append("")
        lines.append("```bash")
        lines.append("invtrack product add")
        lines.append("```")
        lines.append("")
        lines.append("You will be prompted to enter the following information:")
        lines.append("")
        
        # Add section for each prompt
        for prompt in self.prompts:
            field_name = prompt.field_name or "Unknown field"
            lines.append(f"#### {field_name.capitalize()}")
            lines.append("")
            lines.append(f"Prompt: `{prompt.prompt_text}`")
            lines.append("")
            
            if prompt.description:
                lines.append(prompt.description)
                lines.append("")
            
            if prompt.validation_func in self.validations:
                validation = self.validations[prompt.validation_func]
                if validation.error_messages:
                    lines.append("Validation rules:")
                    lines.append("")
                    for msg in validation.error_messages:
                        error_pattern = re.sub(r'{.*?}', '_value_', msg)
                        lines.append(f"- {error_pattern}")
                    lines.append("")
            
            if prompt.default_value:
                lines.append(f"Default: `{prompt.default_value}`")
                lines.append("")
        
        # Add example usage
        lines.append("### Example Usage")
        lines.append("")
        lines.append("Here's an example of adding a product:")
        lines.append("")
        lines.append("```")
        lines.append("$ invtrack product add")
        lines.append("Adding a new product")
        lines.append("Please enter the following information:")
        
        example_values = {
            "name": "Water Bottle",
            "sku": "WTRBOTL1",
            "price": "15.99",
            "reorder_level": "20"
        }
        
        for prompt in self.prompts:
            field = prompt.field_name
            example = example_values.get(field, "example_value")
            if prompt.default_value:
                lines.append(f"{prompt.prompt_text} [{prompt.default_value}]: {example}")
            else:
                lines.append(f"{prompt.prompt_text}: {example}")
        
        lines.append("")
        lines.append("Product added successfully:")
        lines.append("  ID: 28f5c2e7-5d76-4d56-a47d-4afb1af9c476")
        lines.append("  Name: Water Bottle")
        lines.append("  SKU: WTRBOTL1")
        lines.append("  Price: $15.99")
        lines.append("  Reorder Level: 20")
        lines.append("```")
        lines.append("")
        
        # Add non-interactive option
        lines.append("### Non-Interactive Mode")
        lines.append("")
        lines.append("You can also add a product in non-interactive mode:")
        lines.append("")
        lines.append("```bash")
        cmd_parts = ["invtrack product add --non-interactive"]
        for prompt in self.prompts:
            if prompt.field_name:
                cmd_parts.append(f"--{prompt.field_name.replace('_', '-')} \"{example_values.get(prompt.field_name, 'value')}\"")
        lines.append(" \\\n  ".join(cmd_parts))
        lines.append("```")
        lines.append("")
        
        # Add information about overwriting existing products
        lines.append("### Overwriting Existing Products")
        lines.append("")
        lines.append("If you try to add a product with an SKU that already exists, you will be asked if you want to overwrite the existing product:")
        lines.append("")
        lines.append("```")
        lines.append("A product with this SKU already exists!")
        lines.append("")
        lines.append("┌──────────────────────────────────────────────────────────────┐")
        lines.append("│            Product Comparison for SKU: WTRBOTL1              │")
        lines.append("├───────────┬─────────────────┬─────────────────┬─────────────┤")
        lines.append("│ Field     │ Existing Value  │ New Value       │ Status      │")
        lines.append("├───────────┼─────────────────┼─────────────────┼─────────────┤")
        lines.append("│ Name      │ Old Bottle      │ Water Bottle    │ Changed     │")
        lines.append("│ Price     │ $12.99          │ $15.99          │ Changed     │")
        lines.append("│ Reorder   │ 10              │ 20              │ Changed     │")
        lines.append("└───────────┴─────────────────┴─────────────────┴─────────────┘")
        lines.append("")
        lines.append("Do you want to overwrite the existing product? [y/N]: ")
        lines.append("```")
        lines.append("")
        lines.append("To force overwrite without prompting in non-interactive mode, add the `--force` flag:")
        lines.append("")
        lines.append("```bash")
        lines.append("invtrack product add --non-interactive --force --name \"New Name\" --sku \"EXISTING\" --price 19.99 --reorder-level 15")
        lines.append("```")
        
        return "\n".join(lines)


def extract_prompts_from_command(command_module_path: str, output_path: str) -> None:
    """
    Extract prompts from a command module and generate documentation.
    
    Args:
        command_module_path: Path to the command module file
        output_path: Path to the output documentation file
    """
    try:
        extractor = PromptExtractor(command_module_path)
        extractor.parse()
        
        markdown = extractor.generate_markdown()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown)
            
        logger.info(f"Generated documentation at {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to generate documentation: {e}")
        raise


# If run as a script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate documentation from command prompt sequences")
    parser.add_argument("command_path", help="Path to the command module file")
    parser.add_argument("output_path", help="Path to the output documentation file")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    extract_prompts_from_command(args.command_path, args.output_path)