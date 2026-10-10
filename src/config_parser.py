import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import yaml
except ImportError:
    yaml = None


@dataclass
class BenchmarkConfig:
    models: List[str]
    dataset_path: str
    output_dir: str = "outputs"
    max_new_tokens: int = 50
    temperature: float = 0.7
    device: str = "auto"
    warmup: bool = True
    min_quality_tokens: int = 5


def _parse_raw_content(file_path: str) -> dict:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file not found: {file_path}")

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if ext == ".json":
        return json.loads(content)

    if ext in (".yaml", ".yml"):
        if yaml is not None:
            data = yaml.safe_load(content)
            return data if isinstance(data, dict) else {}
        
        # Simple fallback parser if PyYAML is not yet installed
        return _fallback_yaml_parse(content)

    # Attempt JSON first, then YAML
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        if yaml is not None:
            return yaml.safe_load(content)
        return _fallback_yaml_parse(content)


def _fallback_yaml_parse(content: str) -> dict:
    """Basic line parser for simple key-value/list YAML when PyYAML is unavailable."""
    result = {}
    current_list_key = None

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("- ") and current_list_key:
            item = line[2:].strip().strip('"').strip("'")
            result[current_list_key].append(item)
            continue

        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip().split("#")[0].strip()

            if not val:
                result[key] = []
                current_list_key = key
            else:
                current_list_key = None
                val = val.strip('"').strip("'")
                if val.lower() == "true":
                    result[key] = True
                elif val.lower() == "false":
                    result[key] = False
                elif val.isdigit():
                    result[key] = int(val)
                else:
                    try:
                        result[key] = float(val)
                    except ValueError:
                        result[key] = val

    return result


def load_config(file_path: str) -> BenchmarkConfig:
    data = _parse_raw_content(file_path)
    if not isinstance(data, dict):
        raise ValueError("Configuration file must contain a key-value mapping.")

    models = data.get("models")
    if not models or not isinstance(models, list):
        raise ValueError("Configuration must contain a 'models' list.")

    unique_models = list(dict.fromkeys([m.strip() for m in models if isinstance(m, str) and m.strip()]))
    if len(unique_models) < 3:
        raise ValueError(
            f"Configuration requires at least 3 distinct models, found {len(unique_models)}: {unique_models}"
        )

    dataset_path = data.get("dataset_path")
    if not dataset_path or not isinstance(dataset_path, str):
        raise ValueError("'dataset_path' must be specified as a string path.")

    max_new_tokens = int(data.get("max_new_tokens", 50))
    if max_new_tokens <= 0:
        raise ValueError("'max_new_tokens' must be a positive integer.")

    return BenchmarkConfig(
        models=unique_models,
        dataset_path=dataset_path,
        output_dir=str(data.get("output_dir", "outputs")),
        max_new_tokens=max_new_tokens,
        temperature=float(data.get("temperature", 0.7)),
        device=str(data.get("device", "auto")),
        warmup=bool(data.get("warmup", True)),
        min_quality_tokens=int(data.get("min_quality_tokens", 5)),
    )
