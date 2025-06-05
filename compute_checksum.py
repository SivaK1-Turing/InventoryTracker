def _compute_checksum(self, data) -> str:
    # Sort keys for consistent serialization
    serialized = json.dumps(data, sort_keys=True)
    checksum = hashlib.sha256(serialized.encode('utf-8')).digest()
    return base64.b64encode(checksum).decode('utf-8')

def _verify_data_integrity(self, data, expected_checksum: str, data_type: str, raise_error: bool = True):
    # ...
    actual_checksum = self._compute_checksum(data)
    
    if actual_checksum != expected_checksum:
        message = f"Data corruption detected in {data_type} file. Checksum mismatch."
        # ...