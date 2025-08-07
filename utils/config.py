import yaml
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace


def _dict_to_namespace(d):
    """Recursively converts a dictionary to a SimpleNamespace."""
    if not isinstance(d, dict):
        return d
    for k, v in d.items():
        d[k] = _dict_to_namespace(v)
    return SimpleNamespace(**d)


def load_config():
    """Loads configuration from a YAML file and merges it with command-line arguments."""
    parser = ArgumentParser(description="YAML-based configuration loader")
    parser.add_argument(
        "-c", "--config", type=Path, default=Path("config/default.yml"),
        help="Path to the configuration YAML file."
    )
    parser.add_argument(
        "-m", "--model_path", type=str,
        help="Override the output model path (data.model_path in YAML)."
    )
    parser.add_argument(
        "-s", "--source_path", type=str,
        help="Override the data source path (data.source_path in YAML)."
    )
    args = parser.parse_args()

    # Load default config
    with open("config/default.yml", 'r') as f:
        config_dict = yaml.safe_load(f)

    # Load user-specified config and merge if it's not the default
    if args.config.exists() and args.config.resolve() != Path("config/default.yml").resolve():
        print(f"Loading user config from: {args.config}")
        with open(args.config, 'r') as f:
            user_config_dict = yaml.safe_load(f)
        # Simple recursive update
        for key, value in user_config_dict.items():
            if key in config_dict and isinstance(value, dict):
                config_dict[key].update(value)
            else:
                config_dict[key] = value

    # Override with command-line arguments
    if args.model_path:
        config_dict['data']['model_path'] = args.model_path
    if args.source_path:
        config_dict['data']['source_path'] = args.source_path

    config_ns = _dict_to_namespace(config_dict)

    # Return both the config object and the path to the file that was loaded
    return config_ns, args.config.resolve()
