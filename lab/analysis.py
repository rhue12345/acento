"""How well does the grader agree with a native speaker?

Unit of comparison: one word in one recording, judged by one rater.
  - the native speaker says the word sounded native or not
  - the grader flagged the word or not (at two strictness levels:
    clear errors only, or errors + minor notes)

Reported per level:
  caught       = native said "not native" and the grader flagged it
  missed       = native said "not native", grader said nothing
  false alarm  = native said it was fine, grader flagged it
  precision    = of the grader's flags, how many the native agreed with
  recall       = of the native's "not native" words, how many the grader caught
"""


def _flags(rec, severities):
    return {i["word_index"] for i in rec.get("issues", []) if i["severity"] in severities and i["word_index"] >= 0}


def agreement(recordings, ratings):
    recs = {r["id"]: r for r in recordings}
    levels = {"errores": {"error"}, "errores + menores": {"error", "minor"}}
    out = {}
    rows = []
    for name, sev in levels.items():
        c = {"caught": 0, "missed": 0, "false_alarm": 0, "agree_fine": 0}
        for rt in ratings:
            rec = recs.get(rt["recording"])
            if not rec:
                continue
            flagged = _flags(rec, sev)
            human = set(rt["not_native"])
            for wi in range(len(rec["words"])):
                g, h = wi in flagged, wi in human
                key = "caught" if g and h else "missed" if h else "false_alarm" if g else "agree_fine"
                c[key] += 1
                if name == "errores + menores" and key in ("missed", "false_alarm"):
                    notes = "; ".join(f'{i["kind"]}: [{i["expected"]}]→[{i["heard"] or "∅"}]'
                                      for i in rec.get("issues", []) if i["word_index"] == wi)
                    rows.append([rec["text"], rec["words"][wi], rec.get("take", ""), rec.get("speaker", ""), rt["rater"],
                                 "no nativa" if h else "bien", "marcó" if g else "nada", notes])
        flagged_total = c["caught"] + c["false_alarm"]
        human_total = c["caught"] + c["missed"]
        c["precision"] = c["caught"] / flagged_total if flagged_total else None
        c["recall"] = c["caught"] / human_total if human_total else None
        out[name] = c
    return out, rows


def summary_markdown(recordings, ratings):
    stats, _ = agreement(recordings, ratings)
    pct = lambda x: "–" if x is None else f"{x:.0%}"
    lines = [f"**{len(recordings)} grabaciones, {len(ratings)} calificaciones.**", "",
             "| Nivel | Acertó | Se le pasó | Falsa alarma | Precisión | Cobertura |",
             "|---|---|---|---|---|---|"]
    for name, c in stats.items():
        lines.append(f"| {name} | {c['caught']} | {c['missed']} | {c['false_alarm']} | {pct(c['precision'])} | {pct(c['recall'])} |")
    lines += ["", "*Precisión*: de las palabras que marcó el evaluador, cuántas también le sonaron mal a la persona nativa. "
              "*Cobertura*: de las palabras que a la persona nativa le sonaron mal, cuántas detectó el evaluador."]
    return "\n".join(lines)
