"""Publish the Lab to a Hugging Face Space.

    HF_TOKEN=hf_... python lab/deploy.py rhu3

Creates (or updates) the Space <user>/acento-lab and a private dataset
<user>/acento-lab-data for recordings and ratings, and gives the Space the
token it needs to write to that dataset.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parent.parent


def main(user):
    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    space, dataset = f"{user}/acento-lab", f"{user}/acento-lab-data"
    api.create_repo(space, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.create_repo(dataset, repo_type="dataset", private=True, exist_ok=True)
    api.add_space_secret(space, "HF_TOKEN", token)
    api.add_space_variable(space, "DATASET_REPO", dataset)
    if os.environ.get("ELEVENLABS_API_KEY"):  # spoken feedback; can also be added in the Space settings
        api.add_space_secret(space, "ELEVENLABS_API_KEY", os.environ["ELEVENLABS_API_KEY"])

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for f in ("app.py", "storage.py", "analysis.py", "phrases.py", "voice.py", "requirements.txt", "packages.txt"):
            shutil.copy(REPO_ROOT / "lab" / f, tmp / f)
        shutil.copy(REPO_ROOT / "lab" / "SPACE_README.md", tmp / "README.md")
        shutil.copytree(REPO_ROOT / "grader", tmp / "grader", ignore=shutil.ignore_patterns("tests", "__pycache__", "dev_synth.py"))
        api.upload_folder(repo_id=space, repo_type="space", folder_path=tmp, commit_message="Deploy Acento Lab")
    print(f"https://huggingface.co/spaces/{space}")


if __name__ == "__main__":
    main(sys.argv[1])
