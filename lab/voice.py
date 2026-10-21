"""Spoken feedback: explain each mistake in Spanish and say the word correctly.

Uses ElevenLabs. Set ELEVENLABS_API_KEY (a Space secret on Hugging Face).
ELEVENLABS_VOICE_ID is optional; without it the voice is picked from your
ElevenLabs library by the accent being practiced, as in the main app.
"""

import hashlib
import json
import os
import re
import urllib.error
import urllib.request

import storage

API = "https://api.elevenlabs.io/v1"
KEY = os.environ.get("ELEVENLABS_API_KEY", "")
VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "")
MODEL = "eleven_multilingual_v2"
MAX_ISSUES = 3  # keep each explanation short (and cheap)

ACCENT = {
    "es-MX": r"mexic", "es-ES": r"spain|spanish \(spain\)|castil|peninsular|españ|madrid|european",
    "es-AR": r"argent|rioplat", "es-CO": r"colomb", "es-US": r"latin|mexic|\bus\b|american|neutral",
}
_voices = None


def enabled():
    return bool(KEY)


def _request(path, body=None):
    req = urllib.request.Request(API + path, data=json.dumps(body).encode() if body else None,
                                 headers={"xi-api-key": KEY, "content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="ignore")
        print(f"ElevenLabs {path.split('?')[0]} -> {e.code}: {detail[:500]}", flush=True)
        try:  # ElevenLabs explains the problem in detail.message (missing permission, plan, quota...)
            reason = json.loads(detail)["detail"]["message"]
        except Exception:
            reason = detail[:200]
        if e.code == 401 and "quota" in detail.lower():
            raise RuntimeError("Se acabaron tus caracteres de ElevenLabs este mes.")
        if e.code == 429:
            raise RuntimeError("ElevenLabs está ocupado o se acabaron tus caracteres.")
        raise RuntimeError(f"ElevenLabs devolvió un error ({e.code}): {reason}")


def _voice_text(v):
    langs = [x for l in v.get("verified_languages") or [] for x in (l.get("language"), l.get("accent"), l.get("locale"))]
    return " ".join(filter(None, [v.get("name"), v.get("description"), *(v.get("labels") or {}).values(), *langs]))


def pick_voice(variety):
    """The configured voice, or the library voice that best matches the accent."""
    global _voices
    if VOICE_ID:
        return VOICE_ID
    if _voices is None:
        _voices = json.loads(_request("/voices")).get("voices", [])
    if not _voices:
        raise RuntimeError("Tu biblioteca de ElevenLabs no tiene voces.")
    accent = ACCENT.get(variety, ACCENT["es-MX"])
    def score(v):
        t = _voice_text(v)
        return (10 if re.search(accent, t, re.I) else 0) + (5 if re.search(r"spanish|español|\bes\b|es-", t, re.I) else 0)
    return max(_voices, key=score)["voice_id"]


def script(meta):
    """What the coach says: each mistake once, the words said correctly, then the whole phrase."""
    groups = {}  # note -> words with that mistake, so the same explanation isn't repeated
    shown = [i for i in meta["issues"] if i["severity"] in ("error", "minor") and i["word"]]
    for i in sorted(shown, key=lambda i: i["severity"] != "error"):  # clear errors first
        words = groups.setdefault(i["note"], [])
        if i["word"] not in words:
            words.append(i["word"])
    if not groups:
        return f'¡Muy bien! No encontré errores. <break time="0.5s" /> {meta["text"]}'
    parts = []
    for note, words in list(groups.items())[:MAX_ISSUES]:
        names = " y ".join(f"«{w}»" for w in words)
        demo = " ".join(f'{w}. <break time="0.8s" /> {w}. <break time="1.0s" />' for w in words)
        parts.append(f'En {names}: {note} <break time="0.6s" /> Escucha: <break time="0.4s" /> {demo}')
    parts.append(f'Ahora la frase completa: <break time="0.5s" /> {meta["text"]}')
    return " ".join(parts)


def speak(text, variety):
    """Return the path of an mp3 of `text`, generating it only once."""
    voice = pick_voice(variety)
    out = storage.DATA / "tts" / (hashlib.sha1(f"{voice}|{MODEL}|{text}".encode()).hexdigest() + ".mp3")
    if not out.exists():
        audio = _request(f"/text-to-speech/{voice}?output_format=mp3_44100_128", {
            "text": text, "model_id": MODEL,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.8, "style": 0.2, "speed": 0.9},
        })
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(audio)
    return str(out)
