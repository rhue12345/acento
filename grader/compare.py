"""Compare expected Spanish phones with what the phoneme recognizer heard.

The recognizer was trained on dozens of languages, so it can output English
sounds (ɹ, oʊ, eɪ, ə, pʰ, ɫ) that never occur in native Spanish. Those are
exactly the marks of an English accent, so each one maps to a specific issue.
"""

from dataclasses import dataclass
import re

from .g2p_es import transcribe, Word

# ---- normalizing recognizer output -----------------------------------------

# Tokens the multilingual model emits that mean the same thing for Spanish.
_EQUIV = {
    "g": "ɡ", "e̞": "e", "o̞": "o", "r.": "r", "a.": "a", "i.": "i", "u.": "u",
    "s̪": "s", "t̪": "t", "d̪": "d", "n̪": "n", "ä": "a", "ʎ": "ʝ",
}


def normalize_heard(tokens):
    """Clean recognizer tokens: drop tones/length marks and specials, unify symbols."""
    out = []
    for t in tokens:
        if t in ("<pad>", "<s>", "</s>", "<unk>", "|", "") or t.isspace():
            continue
        t = re.sub(r"[0-9ɜ]+$", "", t) if not t.startswith("ɜ") else t  # tone numbers
        t = t.replace("ː", "")
        out.append(_EQUIV.get(t, t))
    return out


# ---- alignment --------------------------------------------------------------

_GROUPS = [
    {"r", "ɾ", "ɹ", "ɻ", "ɽ", "ʁ", "r̝"},
    {"a", "ɑ", "æ", "ɐ", "ʌ", "ə"},
    {"e", "ɛ", "eɪ", "ə", "ɪ"},
    {"i", "ɪ", "j", "iː"},
    {"o", "ɔ", "oʊ", "əʊ", "ə"},
    {"u", "ʊ", "w", "uː"},
    {"b", "β", "v", "p", "pʰ"},
    {"d", "ð", "t", "tʰ", "ɾ"},
    {"ɡ", "ɣ", "k", "kʰ", "x"},
    {"p", "pʰ"}, {"t", "tʰ"}, {"k", "kʰ"},
    {"l", "ɫ"},
    {"s", "z", "θ", "ʃ"},
    {"x", "h", "χ"},
    {"ʝ", "j", "ʒ", "dʒ", "ʃ"},
    {"tʃ", "ʃ"},
    {"n", "m", "ŋ", "ɲ"},
]


def _sub_cost(a, b):
    if a == b:
        return 0.0
    if any(a in g and b in g for g in _GROUPS):
        return 0.6
    return 1.0


def align(expected, heard):
    """Edit-distance alignment. Returns list of (expected_index | None, heard_phone | None)."""
    n, m = len(expected), len(heard)
    INS = DEL = 0.8
    d = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i * DEL
    for j in range(1, m + 1):
        d[0][j] = j * INS
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + DEL, d[i][j - 1] + INS,
                          d[i - 1][j - 1] + _sub_cost(expected[i - 1], heard[j - 1]))
    pairs, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + _sub_cost(expected[i - 1], heard[j - 1]):
            pairs.append((i - 1, heard[j - 1])); i -= 1; j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + DEL:
            pairs.append((i - 1, None)); i -= 1
        else:
            pairs.append((None, heard[j - 1])); j -= 1
    return pairs[::-1]


# ---- diagnosis --------------------------------------------------------------

@dataclass
class Issue:
    word: str
    expected: str
    heard: str          # "" when the sound was dropped
    kind: str           # short id, e.g. "rr_tap"
    severity: str       # "error" = clear English accent, "minor" = non-native but common
    note: str           # Spanish explanation for the learner


VOWELS = set("aeiou")


def _classify(exp, heard, stressed_vowel):
    """Return (kind, severity, note) for a mismatch, or None if it's acceptable."""
    if heard is None:
        if exp in VOWELS:
            return ("vowel_dropped", "error", "Se perdió una vocal. En español todas las vocales se pronuncian completas.")
        return ("dropped", "minor", f"No se oyó el sonido [{exp}].")
    if exp == "r":
        if heard == "ɾ":
            return ("rr_tap", "error", "Esta r es fuerte (vibrante múltiple): la punta de la lengua debe vibrar varias veces, no solo tocar una vez.")
        if heard in ("ɹ", "ɻ"):
            return ("r_english", "error", "Sonó como la r inglesa. En español la punta de la lengua vibra contra el paladar, detrás de los dientes.")
    if exp == "ɾ":
        if heard in ("ɹ", "ɻ"):
            return ("r_english", "error", "Sonó como la r inglesa. Esta r es un toque rápido de la punta de la lengua, como la tt de 'butter' en inglés americano.")
        if heard == "r":
            return ("r_trilled", "minor", "Esta r es suave: un solo toque de la lengua, sin vibrar.")
        if heard == "d":
            return None
    if exp in VOWELS:
        if heard in ("oʊ", "əʊ") and exp == "o":
            return ("vowel_glide", "error", "La o se deslizó hacia 'ou' como en inglés. Mantén los labios redondos y quietos hasta el final.")
        if heard in ("eɪ",) and exp == "e":
            return ("vowel_glide", "error", "La e se deslizó hacia 'ei' como en inglés. Mantén la boca quieta: una e corta y pura.")
        if heard in ("ə", "ɪ", "ʊ", "ʌ"):
            what = "átona" if not stressed_vowel else ""
            return ("vowel_reduced", "error", f"La vocal {what} sonó apagada ('uh'). En español nunca se reducen: di una {exp} clara.".replace("  ", " "))
        if heard in ("æ", "ɛ", "ɔ", "ɑ") and heard[0] != exp:
            return ("vowel_quality", "minor", f"La {exp} no sonó del todo española. Ábrela o ciérrala como en el modelo.")
    if exp in ("p", "t", "k") and heard == exp + "ʰ":
        return ("aspirated", "error", f"La {exp} salió con un soplo de aire, como en inglés. En español es seca: pon la mano frente a la boca, no debe notarse aire.")
    if exp in ("β", "ð", "ɣ") and heard in ("b", "d", "ɡ"):
        letter = {"β": "b", "ð": "d", "ɣ": "g"}[exp]
        return ("hard_stop", "minor", f"Entre vocales la {letter} es suave: casi sin cerrar la boca (la d casi como la 'th' de 'this').")
    if exp == "ð" and heard in ("ɾ",):
        return ("d_flap", "error", "La d sonó como la d rápida del inglés americano ('ladder'). Pon la lengua entre los dientes, suave.")
    if exp == "l" and heard == "ɫ":
        return ("dark_l", "error", "La l sonó oscura, como en inglés 'ball'. En español la l es clara: punta de la lengua arriba, sin echar la lengua hacia atrás.")
    if exp == "x" and heard in ("h",):
        return None
    if heard == "ɹ":
        return ("r_english", "error", "Se oyó una r inglesa.")
    return ("different", "minor", f"Se esperaba [{exp}] y se oyó [{heard}].")


def diagnose(text, heard_tokens, variety="es-MX"):
    """Grade a recording. Returns (words, issues, heard_phones)."""
    words = transcribe(text, variety)
    expected, owner = [], []
    for wi, w in enumerate(words):
        for pi, p in enumerate(w.phones):
            expected.append(p); owner.append((wi, pi))
    heard = normalize_heard(heard_tokens)
    issues = []
    for ei, h in align(expected, heard):
        if ei is None:
            if h in ("ə", "ɹ"):  # an inserted schwa or English r
                issues.append(Issue("", "", h, "inserted", "minor", f"Se oyó un sonido de más [{h}]."))
            continue
        wi, pi = owner[ei]
        w = words[wi]
        exp = expected[ei]
        if h is not None and h in w.variants[pi]:
            continue
        syl = next(k for k, (a, b) in enumerate(w.syllables) if a <= pi < b)
        result = _classify(exp, h, syl == w.stress)
        if result:
            kind, severity, note = result
            issues.append(Issue(w.text, exp, h or "", kind, severity, note))
    return words, issues, heard
