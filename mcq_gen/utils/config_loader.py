from pathlib import Path
import os
import yaml


def _project_root() -> Path:
    # E:\Project\MCQ-Generator\mcq_gen\utils\config_loader.py
    # project root - E:\Project\MCQ-Generator
    return Path(__file__).resolve().parents[2]  

def load_config(config_path: str | None = None) -> dict:
    # get env from env variables
    env_path = os.getenv("CONFIG_PATH")

    # if there is not env_path
    if config_path is None:
        config_path = env_path or str(_project_root() / "mcq_gen" / "config" / "config.yaml")

    path = Path(config_path)
    if not path.absolute():
        path = _project_root() / path

    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

