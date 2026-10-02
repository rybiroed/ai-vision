import json


def load_config(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)
