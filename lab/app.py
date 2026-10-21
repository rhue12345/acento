"""Acento Lab: record test phrases, have a native speaker rate them blind,
and measure how well the grader agrees.

Run locally:  python lab/app.py
"""

import random
import sys
import threading
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent)]  # grader/ sits next to app.py on the Space, one level up in the repo

import gradio as gr

import analysis
import storage
from grader.compare import diagnose
from grader.g2p_es import to_ipa
from grader.recognize import load_audio, recognize
from phrases import BY_ID, PHRASES

VARIETY_LABELS = {"México": "es-MX", "España": "es-ES", "Argentina": "es-AR", "Colombia": "es-CO", "Estados Unidos": "es-US"}
ROLES = ["Aprendiz", "Nativo/a"]
TAKES = {"Lo mejor que puedas": "bien", "Con el error a propósito": "error"}
SEVERITY_LABEL = {"error": "❌ Error claro", "minor": "⚠️ Detalle", "unsure": "· Dudoso (no se muestra a alumnos)"}


def phrase_card(phrase_id, role, take):
    p = BY_ID[phrase_id]
    idx = next(i for i, x in enumerate(PHRASES) if x["id"] == phrase_id) + 1
    lines = [f"### {idx}/{len(PHRASES)} · {p['text']}", f"Sonidos: **{p['sounds']}**"]
    if role == "Aprendiz" and TAKES.get(take) == "error":
        lines.append(f"🎯 **Error a propósito:** {p['mistake']}")
    elif role == "Aprendiz":
        lines.append("Dila lo más nativa que puedas.")
    else:
        lines.append("Dila de forma natural, como hablas normalmente.")
    return "\n\n".join(lines)


def progress(name):
    if not name.strip():
        return ""
    mine = [r for r in storage.recordings() if r["speaker"].strip().lower() == name.strip().lower()]
    done = {(r["phrase_id"], r["take"]) for r in mine}
    return f"Llevas **{len(mine)}** grabaciones ({len(done)} combinaciones frase/toma distintas)."


def result_markdown(meta):
    lines = [f"**Esperado:** `{meta['expected']}`", f"**Oído:** `{meta['heard']}`", ""]
    shown = [i for i in meta["issues"] if i["severity"] != "unsure"]
    if not shown:
        lines.append("✅ El evaluador no encontró errores.")
    for i in meta["issues"]:
        heard = i["heard"] or "∅"
        lines.append(f"- {SEVERITY_LABEL[i['severity']]} · **{i['word'] or '–'}** [{i['expected']}]→[{heard}] — {i['note']}")
    return "\n".join(lines)


def analyze(audio, name, role, variety_label, phrase_id, take):
    if not name.strip():
        raise gr.Error("Escribe tu nombre arriba.")
    if not audio:
        raise gr.Error("Graba la frase primero.")
    p = BY_ID[phrase_id]
    variety = VARIETY_LABELS[variety_label]
    words, issues, heard = diagnose(p["text"], recognize(load_audio(audio)), variety)
    meta = {
        "speaker": name.strip(), "role": "learner" if role == "Aprendiz" else "native",
        "take": TAKES.get(take, "bien") if role == "Aprendiz" else "nativo",
        "phrase_id": phrase_id, "text": p["text"], "variety": variety,
        "words": [w.text for w in words], "expected": to_ipa(words), "heard": " ".join(heard),
        "issues": [asdict(i) for i in issues],
    }
    storage.save_recording(audio, meta)
    return result_markdown(meta) + "\n\n*Guardado.*", progress(name)


def next_phrase(phrase_id):
    ids = [p["id"] for p in PHRASES]
    return ids[(ids.index(phrase_id) + 1) % len(ids)], None


# ---- rating ----

def pick_for_rating(rater):
    if not rater.strip():
        raise gr.Error("Escribe tu nombre arriba.")
    who = rater.strip().lower()
    rated = {r["recording"] for r in storage.ratings() if r["rater"].strip().lower() == who}
    pool = [r for r in storage.recordings() if r["id"] not in rated and r["speaker"].strip().lower() != who]
    learners = [r for r in pool if r["role"] == "learner"]
    rec = random.choice(learners or pool) if pool else None
    if not rec:
        return (None, "🎉 No quedan grabaciones por calificar.", gr.update(choices=[], value=[]), None, "", "")
    natives = [r for r in storage.recordings() if r["phrase_id"] == rec["phrase_id"] and r["role"] == "native"
               and r["id"] != rec["id"]]
    ref = storage.audio_path(natives[-1]["id"]) if natives else None
    choices = [(f"{i + 1}. {w}", i) for i, w in enumerate(rec["words"])]
    left = len(pool) - 1
    md = f"### {rec['text']}\n\nMarca las palabras que **no** suenan nativas. Si toda la frase suena nativa, no marques nada.\n\n*Quedan {left} más.*"
    return storage.audio_path(rec["id"]), md, gr.update(choices=choices, value=[]), ref, rec["id"], ""


def save_and_next(rec_id, rater, chosen, note):
    if not rec_id:
        raise gr.Error("Primero pulsa «Siguiente grabación».")
    storage.save_rating(rec_id, rater.strip(), [int(c) for c in chosen], note.strip())
    return pick_for_rating(rater)


def results():
    recs, rts = storage.recordings(), storage.ratings()
    _, rows = analysis.agreement(recs, rts)
    return analysis.summary_markdown(recs, rts), rows


# ---- UI ----

with gr.Blocks(title="Acento Lab") as demo:
    gr.Markdown("# Acento Lab\nGraba frases, califícalas a ciegas y mide si el evaluador acierta.")
    with gr.Tab("Grabar"):
        with gr.Row():
            name = gr.Textbox(label="Tu nombre", scale=2)
            role = gr.Radio(ROLES, value="Aprendiz", label="Soy")
            variety = gr.Dropdown(list(VARIETY_LABELS), value="México", label="Acento")
        with gr.Row():
            phrase = gr.Dropdown([(f"{i + 1}. {p['text']}", p["id"]) for i, p in enumerate(PHRASES)],
                                 value=PHRASES[0]["id"], label="Frase", scale=3)
            take = gr.Radio(list(TAKES), value="Lo mejor que puedas", label="Toma")
        card = gr.Markdown(phrase_card(PHRASES[0]["id"], "Aprendiz", "Lo mejor que puedas"))
        audio = gr.Audio(sources=["microphone"], type="filepath", format="wav", label="Grabación")
        with gr.Row():
            go = gr.Button("Guardar y analizar", variant="primary")
            nxt = gr.Button("Siguiente frase →")
        result = gr.Markdown()
        prog = gr.Markdown()

        for comp in (phrase, role, take):
            comp.change(phrase_card, [phrase, role, take], card)
        role.change(lambda r: gr.update(visible=r == "Aprendiz"), role, take)
        name.change(progress, name, prog)
        go.click(analyze, [audio, name, role, variety, phrase, take], [result, prog])
        nxt.click(next_phrase, phrase, [phrase, audio]).then(lambda: "", None, result)

    with gr.Tab("Calificar"):
        gr.Markdown("Para la persona nativa. No verás quién grabó ni qué dijo el evaluador.")
        rater = gr.Textbox(label="Tu nombre")
        start = gr.Button("Siguiente grabación", variant="primary")
        rec_state = gr.State("")
        rate_md = gr.Markdown()
        rate_audio = gr.Audio(label="Grabación a calificar", interactive=False)
        ref_audio = gr.Audio(label="Referencia nativa (si existe)", interactive=False)
        words = gr.CheckboxGroup(label="Palabras que NO suenan nativas", choices=[])
        note = gr.Textbox(label="Comentario (opcional): ¿qué sonó mal?", lines=2)
        save = gr.Button("Guardar y siguiente", variant="primary")
        outs = [rate_audio, rate_md, words, ref_audio, rec_state, note]
        start.click(pick_for_rating, rater, outs)
        save.click(save_and_next, [rec_state, rater, words, note], outs)

    with gr.Tab("Resultados"):
        refresh = gr.Button("Actualizar")
        summary = gr.Markdown()
        table = gr.Dataframe(headers=["Frase", "Palabra", "Toma", "Habló", "Calificó", "Nativa dice", "Evaluador", "Detalle"],
                             wrap=True, label="Desacuerdos")
        refresh.click(results, None, [summary, table])
        demo.load(results, None, [summary, table])

storage.init()


def _warm():  # load the model in the background so the first recording isn't slow
    import numpy as np
    recognize(np.zeros(16000, dtype=np.float32))


threading.Thread(target=_warm, daemon=True).start()

if __name__ == "__main__":
    demo.launch()
