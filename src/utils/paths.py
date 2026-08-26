from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]


def build_path(*paths: str) -> Path:
    return BASE_DIR.joinpath(*paths)
