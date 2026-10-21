"""Run the phoneme recognizer on a recording.

Uses facebook/wav2vec2-xlsr-53-espeak-cv-ft, a free multilingual model that
transcribes speech into IPA phones.

    python -m grader.recognize recording.wav "el perro corre" --variety es-MX
"""

import argparse
import json
import os
import sys

import numpy as np

MODEL_ID = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
SAMPLE_RATE = 16000

_model = None


def _load(model_path=None):
    global _model
    if _model is None:
        import torch
        from transformers import Wav2Vec2ForCTC
        path = model_path or os.environ.get("ACENTO_MODEL", MODEL_ID)
        model = Wav2Vec2ForCTC.from_pretrained(path).eval()
        with open(os.path.join(path, "vocab.json")) if os.path.isdir(path) else open(_hub_vocab(path)) as f:
            vocab = json.load(f)
        id2tok = {i: t for t, i in vocab.items()}
        _model = (torch, model, id2tok)
    return _model


def _hub_vocab(model_id):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(model_id, "vocab.json")


def load_audio(path):
    """Mono float32 at 16 kHz."""
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = resample(audio, sr, SAMPLE_RATE)
    return audio


def resample(audio, sr, target):
    n = int(round(len(audio) * target / sr))
    return np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)


def recognize(audio, model_path=None):
    """Return the list of phone tokens the model heard (CTC greedy decoding)."""
    torch, model, id2tok = _load(model_path)
    audio = (audio - audio.mean()) / (audio.std() + 1e-7)
    with torch.no_grad():
        logits = model(torch.from_numpy(audio)[None]).logits[0]
    ids = logits.argmax(-1).tolist()
    pad = model.config.pad_token_id
    tokens, prev = [], None
    for i in ids:
        if i != prev and i != pad:
            tokens.append(id2tok[i])
        prev = i
    return tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("text")
    ap.add_argument("--variety", default="es-MX")
    ap.add_argument("--model", default=None, help="local model directory")
    args = ap.parse_args()

    from .compare import diagnose
    from .g2p_es import to_ipa
    tokens = recognize(load_audio(args.audio), args.model)
    words, issues, heard = diagnose(args.text, tokens, args.variety)
    print("expected:", to_ipa(words))
    print("heard:   ", " ".join(heard))
    if not issues:
        print("no issues")
    for i in issues:
        print(f"[{i.severity}] {i.word or '-'}: expected [{i.expected}] heard [{i.heard}] {i.kind}: {i.note}")


if __name__ == "__main__":
    sys.exit(main())
