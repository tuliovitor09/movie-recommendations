import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"


def load_users():
    with open(DATA_DIR / "users.json", "r") as f:
        return json.load(f)


def load_movies():
    with open(DATA_DIR / "movies.json", "r") as f:
        return json.load(f)


def load_interactions():
    with open(DATA_DIR / "interactions.json", "r") as f:
        return json.load(f)
