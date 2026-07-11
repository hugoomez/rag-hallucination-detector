"""Download the RAGTruth dataset by cloning its GitHub repository."""

import json
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/ParticleMedia/RAGTruth.git"
TARGET_DIR = Path("data/raw/ragtruth")

LICENSE_CANDIDATES = ["LICENSE", "LICENSE.md", "LICENSE.txt"]
SAMPLE_FILES = ["dataset/source_info.jsonl", "dataset/response.jsonl"]


def clone_repo() -> None:
    if TARGET_DIR.exists():
        print(f"'{TARGET_DIR}' already exists, skipping clone.")
        return

    TARGET_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning {REPO_URL} into {TARGET_DIR} (shallow clone)...")
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(TARGET_DIR)],
        check=True,
    )
    print("Clone completed.")


def print_license() -> None:
    for name in LICENSE_CANDIDATES:
        license_path = TARGET_DIR / name
        if license_path.exists():
            print(f"\n--- Contents of {name} ---")
            print(license_path.read_text(encoding="utf-8", errors="replace"))
            return

    print("\nWarning: no visible license file found in the repo.")


def print_first_line_as_json(relative_path: str) -> None:
    file_path = TARGET_DIR / relative_path
    print(f"\n--- First line of {relative_path} ---")

    if not file_path.exists():
        print(f"Warning: file {relative_path} was not found.")
        return

    with file_path.open("r", encoding="utf-8") as f:
        first_line = f.readline()

    if not first_line.strip():
        print(f"Warning: {relative_path} is empty.")
        return

    record = json.loads(first_line)
    print(json.dumps(record, indent=4, ensure_ascii=False))


def main() -> None:
    clone_repo()
    print_license()
    for sample_file in SAMPLE_FILES:
        print_first_line_as_json(sample_file)


if __name__ == "__main__":
    main()
