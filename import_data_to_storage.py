async def import_data_to_storage(path, format, adapter_name, dry_run):
    # Phase 1: Validation
    (valid_products, valid_transactions, ...) = await load_and_validate_file(
        path, format, product_errors, transaction_errors
    )
    
    # If it's a dry run or validation failed completely, stop here
    if dry_run or (not valid_products and not valid_transactions):
        return result
    
    # Phase 2: Import of valid records only
    for product in valid_products:
        try:
            await adapter.save_product(product)
            # Track success stats...
        except Exception as e:
            # Track import errors but continue
            product_import_errors[ref_id] = [str(e)]