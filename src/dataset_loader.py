import csv
import json
import os
from typing import Dict, List


def load_dataset(file_path: str) -> List[Dict[str, str]]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    prompts = []

    if ext in (".jsonl", ".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    prompt_text = obj.get("prompt") or obj.get("text") or line
                    prompt_id = str(obj.get("id", f"p_{idx}"))
                except json.JSONDecodeError:
                    prompt_text = line
                    prompt_id = f"p_{idx}"

                prompts.append({"id": prompt_id, "prompt": prompt_text})

    elif ext == ".csv":
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, 1):
                prompt_text = row.get("prompt") or row.get("text") or list(row.values())[0]
                prompt_id = row.get("id", f"p_{idx}")
                prompts.append({"id": str(prompt_id), "prompt": prompt_text})

    elif ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for idx, item in enumerate(data, 1):
                    if isinstance(item, dict):
                        prompts.append({
                            "id": str(item.get("id", f"p_{idx}")),
                            "prompt": item.get("prompt") or item.get("text") or str(item),
                        })
                    else:
                        prompts.append({"id": f"p_{idx}", "prompt": str(item)})

    if not prompts:
        raise ValueError(f"No valid prompts found in dataset file: {file_path}")

    return prompts
