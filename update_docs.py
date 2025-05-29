# scripts/update_docs.py
#!/usr/bin/env python
import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from inventorytracker.utils.doc_generator import extract_prompts_from_command

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    """Update documentation from source code."""
    try:
        # Project paths
        project_root = Path(__file__).parent.parent
        docs_dir = project_root / "docs"
        
        # Ensure docs directory exists
        docs_dir.mkdir(exist_ok=True)
        
        # Add Product command
        add_product_path = project_root / "inventorytracker" / "commands" / "add_product.py"
        usage_doc_path = docs_dir / "usage.md"
        
        if not add_product_path.exists():
            logger.error(f"Command file not found: {add_product_path}")
            return 1
            
        logger.info(f"Extracting prompts from {add_product_path}")
        extract_prompts_from_command(str(add_product_path), str(usage_doc_path))
        
        logger.info(f"Documentation updated: {usage_doc_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error updating documentation: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())