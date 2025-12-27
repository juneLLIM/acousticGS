import yaml
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace


def _dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
    return d


def _recursive_update(d1, d2):
    """Recursively merges dictionary d2 into dictionary d1."""
    for k, v in d2.items():
        if (isinstance(d1.get(k), dict) and isinstance(v, dict)):
            _recursive_update(d1[k], v)
        else:
            d1[k] = v


def load_config():
    """Loads configuration from a YAML file and merges it with command-line arguments."""
    parser = ArgumentParser(description="YAML-based configuration loader")
    parser.add_argument(
        "-c", "--config", type=Path, default=Path("config/default.yml"),
        help="Path to the configuration YAML file."
    )
    parser.add_argument(
        "-d", "--data_path", type=str,
        help="Override the data source path (path.data in YAML)."
    )
    parser.add_argument(
        "-o", "--output_path", type=str,
        help="Override the output directory path (path.output in YAML)."
    )
    parser.add_argument(
        "-ck", "--checkpoint_path", type=str,
        help="Override the checkpoint path (path.checkpoint in YAML)."
    )
    parser.add_argument(
        "-w", "--wandb", type=bool,
        help="Enable or disable W&B logging."
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
        _recursive_update(config_dict, user_config_dict)

    config = _dict_to_namespace(config_dict)

    # Override with command-line arguments if they are provided
    config.path.config = args.config
    if args.data_path is not None:
        config.path.data = args.data_path
    if args.output_path is not None:
        config.path.output = args.output_path
    if args.checkpoint_path is not None:
        config.path.checkpoint = args.checkpoint_path
    if args.wandb is not None:
        config.logging.wandb = args.wandb
    if args.num_samples is not None:
        config.logging.num_samples = args.num_samples

    # Return the config namespace
    return config
