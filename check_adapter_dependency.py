def _check_adapter_dependencies(name: str) -> bool:
    """Check if all dependencies for an adapter are available."""
    _, dependencies = _ADAPTER_REGISTRY[name]
    missing = []
    
    for dependency in dependencies:
        try:
            importlib.import_module(dependency)
        except ImportError:
            missing.append(dependency)
    
    if missing:
        _ADAPTER_MISSING_DEPS[name] = missing
        return False
    
    return True

def validate_available_adapters() -> Dict[str, bool]:
    """Validate which registered adapters are available."""
    availability = {}
    for name in _ADAPTER_REGISTRY:
        availability[name] = _check_adapter_dependencies(name)
    return availability