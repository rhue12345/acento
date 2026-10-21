"""Where the Lab keeps recordings and ratings.

Everything lives in a local folder. When HF_TOKEN and DATASET_REPO are set (on
the Hugging Face Space), the folder is first filled from a private dataset repo
and every new file is pushed back to it, so nothing is lost when the Space
restarts.

    recordings/<id>.wav   the audio
    recordings/<id>.json  who, which phrase, which take, the grader's result
    ratings/<rec>__<rater>.json  a native speaker's word-by-word verdict
"""

import json
import os
import threading
import time
import uuid
from pathlib import Path

DATA = Path(os.environ.get("LAB_DATA", "lab_data"))
DATASET_REPO = os.environ.get("DATASET_REPO", "")
TOKEN = os.environ.get("HF_TOKEN", "")
_lock = threading.Lock()


def _api():
    from huggingface_hub import HfApi
    return HfApi(token=TOKEN)


def remote_enabled():
    return bool(DATASET_REPO and TOKEN)


def init():
    (DATA / "recordings").mkdir(parents=True, exist_ok=True)
    (DATA / "ratings").mkdir(parents=True, exist_ok=True)
    if remote_enabled():
        from huggingface_hub import snapshot_download
        _api().create_repo(DATASET_REPO, repo_type="dataset", private=True, exist_ok=True)
        snapshot_download(DATASET_REPO, repo_type="dataset", local_dir=DATA, token=TOKEN)


def _push(paths):
    if not remote_enabled():
        return
    def run():
        for attempt in range(3):
            try:
                from huggingface_hub import CommitOperationAdd
                ops = [CommitOperationAdd(path_in_repo=str(p.relative_to(DATA)), path_or_fileobj=str(p)) for p in paths]
                _api().create_commit(DATASET_REPO, ops, repo_type="dataset", commit_message=f"add {paths[0].name}")
                return
            except Exception as e:  # network hiccup: retry, the local copy is safe meanwhile
                print("push failed:", e)
                time.sleep(2 ** attempt)
    threading.Thread(target=run, daemon=True).start()


def save_recording(audio_path, meta):
    rec_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    wav = DATA / "recordings" / f"{rec_id}.wav"
    js = DATA / "recordings" / f"{rec_id}.json"
    with _lock:
        wav.write_bytes(Path(audio_path).read_bytes())
        js.write_text(json.dumps({"id": rec_id, "created": time.time(), **meta}, ensure_ascii=False, indent=1))
    _push([wav, js])
    return rec_id


def recordings():
    out = []
    for js in sorted((DATA / "recordings").glob("*.json")):
        try:
            out.append(json.loads(js.read_text()))
        except Exception:
            pass
    return out


def audio_path(rec_id):
    return str(DATA / "recordings" / f"{rec_id}.wav")


def save_rating(rec_id, rater, not_native_words, note):
    safe = "".join(c for c in rater.lower() if c.isalnum()) or "anon"
    path = DATA / "ratings" / f"{rec_id}__{safe}.json"
    with _lock:
        path.write_text(json.dumps({"recording": rec_id, "rater": rater, "not_native": sorted(not_native_words),
                                    "note": note, "created": time.time()}, ensure_ascii=False, indent=1))
    _push([path])


def ratings():
    out = []
    for js in sorted((DATA / "ratings").glob("*.json")):
        try:
            out.append(json.loads(js.read_text()))
        except Exception:
            pass
    return out
