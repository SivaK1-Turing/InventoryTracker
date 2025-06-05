def _atomic_write(self, file_path: Path, data: Dict) -> None:
    # Create a unique temp file in the same directory
    temp_file_handle, temp_file_path = tempfile.mkstemp(
        prefix=f"{file_path.stem}_",
        suffix=".tmp",
        dir=str(file_path.parent)
    )
    
    try:
        # Write to the temporary file
        with os.fdopen(temp_file_handle, 'w') as f:
            fcntl.flock(f, fcntl.LOCK_EX)  # Exclusive lock
            json.dump(file_with_checksum, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())  # Flush all buffers to disk
            
        # Create backup of existing file
        if file_path.exists():
            self._create_backup(file_path)
        
        # Atomic rename operation
        os.replace(temp_file_path, file_path)
        
    except Exception:
        # Clean up the temp file if something goes wrong
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        raise