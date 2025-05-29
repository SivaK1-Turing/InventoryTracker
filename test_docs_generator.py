# tests/test_doc_generator.py
import pytest
import os
import tempfile
from pathlib import Path
import re

from inventorytracker.utils.doc_generator import PromptExtractor, extract_prompts_from_command

# Sample code to test the extractor
SAMPLE_CODE = """
import typer
from typing import Optional

def validate_name(value: str) -> str:
    \"\"\"Validate product name.\"\"\"
    if not value or len(value.strip()) < 3:
        raise typer.BadParameter("Product name must be at least 3 characters long")
    return value.strip()

def validate_sku(value: str) -> str:
    \"\"\"Validate SKU format.\"\"\"
    value = value.strip().upper()
    if not re.match(r'^[A-Z0-9]+$', value):
        raise typer.BadParameter("SKU must contain only uppercase letters and numbers")
    return value

def prompt_with_validation(prompt_text: str, validation_func, default=None):
    while True:
        try:
            value = typer.prompt(
                prompt_text,
                default=default,
                show_default=default is not None
            )
            return validation_func(value)
        except typer.BadParameter as e:
            print(f"Error: {str(e)}")

def add_product():
    \"\"\"Add a new product to inventory.\"\"\"
    print("Adding a new product")
    print("Please enter the following information:")
    
    name_value = prompt_with_validation(
        "Product name (min 3 characters)",
        validate_name
    )
    
    sku_value = prompt_with_validation(
        "SKU (uppercase alphanumeric)",
        validate_sku
    )
    
    price_value = prompt_with_validation(
        "Price",
        validate_price
    )
    
    reorder_value = prompt_with_validation(
        "Reorder level",
        validate_reorder_level,
        default="10"
    )
"""

def test_prompt_extractor():
    """Test that PromptExtractor correctly extracts prompts."""
    # Create temporary file with sample code
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(SAMPLE_CODE)
        temp_file = f.name
    
    try:
        # Extract prompts
        extractor = PromptExtractor(temp_file)
        extractor.parse()
        
        # Check if prompts were extracted
        assert len(extractor.prompts) == 4
        
        # Check specific prompts
        assert "Product name (min 3 characters)" in [p.prompt_text for p in extractor.prompts]
        assert "SKU (uppercase alphanumeric)" in [p.prompt_text for p in extractor.prompts]
        assert "Reorder level" in [p.prompt_text for p in extractor.prompts]
        
        # Check if any prompt has a default value
        assert any(p.default_value == "10" for p in extractor.prompts)
        
        # Check if validation functions were extracted
        assert "validate_name" in extractor.validations
        assert "validate_sku" in extractor.validations
        
        # Check if docstrings were extracted
        assert extractor.validations["validate_name"].docstring == "Validate product name."
        
        # Check if error messages were extracted
        assert any("3 characters" in msg for msg in 
                  extractor.validations["validate_name"].error_messages)
        
        # Generate markdown and check content
        markdown = extractor.generate_markdown()
        assert "# Command Usage Guide" in markdown
        assert "Product name (min 3 characters)" in markdown
        assert "Validation rules:" in markdown
        
    finally:
        # Clean up temp file
        os.unlink(temp_file)

def test_extract_prompts_from_command():
    """Test the extract_prompts_from_command function."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as src_file:
        src_file.write(SAMPLE_CODE)
        src_path = src_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as dst_file:
        dst_path = dst_file.name
    
    try:
        # Extract prompts and generate documentation
        extract_prompts_from_command(src_path, dst_path)
        
        # Check that the documentation was created
        assert os.path.exists(dst_path)
        
        # Read and check content
        with open(dst_path, 'r') as f:
            content = f.read()
            assert "# Command Usage Guide" in content
            assert "Product name (min 3 characters)" in content
    
    finally:
        # Clean up temp files
        os.unlink(src_path)
        os.unlink(dst_path)