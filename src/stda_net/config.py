from argparse import Namespace
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def namespace_from_config(path: str) -> Namespace:
    return Namespace(**load_yaml_config(path))


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)
