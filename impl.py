def _load_config(self, config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    paths_to_try = []
    
    # 1. Specified config path
    if config_path:
        paths_to_try.append(Path(config_path))
    
    # 2. Default config path
    paths_to_try.append(self.DEFAULT_CONFIG_PATH)
    
    # 3. Check for config path in environment variable
    env_config_path = os.environ.get("INVENTORY_TRACKER_CONFIG")
    if env_config_path:
        paths_to_try.append(Path(env_config_path))
    
    # Try each path
    for path in paths_to_try:
        try:
            if not path.exists():
                continue
            
            with open(path, "rb") as f:
                config_data = tomli.load(f)
            
            # Extract email specific config
            if "notifications" in config_data and "email" in config_data["notifications"]:
                self._loaded_config_path = path
                return config_data["notifications"]["email"]