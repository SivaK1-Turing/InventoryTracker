def _create_backup(self, file_path: Path):
    # Generate backup filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    backup_path = file_path.parent / f"{file_path.name}.{timestamp}.backup"
    
    # Copy file to backup
    shutil.copy2(file_path, backup_path)
    # ...

def _try_recover_from_backup(self):
    # Find the latest backup files
    product_backups = sorted(self.data_dir.glob("products.*.backup"), reverse=True)
    # ...
    
    # Try each backup set until a valid one is found
    for backup_idx in range(min(len(product_backups), len(transaction_backups), len(metadata_backups))):
        try:
            # Restore and verify each backup set
            # ...
            
            # If successful, restore the backup
            # ...
        
        except Exception:
            # Continue to next backup if this one fails
            # ...