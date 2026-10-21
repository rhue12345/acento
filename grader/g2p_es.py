"""Spanish spelling -> expected pronunciation (IPA), per regional variety.

Spanish spelling is regular enough that rules beat a dictionary. The output keeps
what a grader needs: one list of phones per word, the stressed syllable, and for
each phone the variants a native speaker might also produce (so natural variation
is never marked as an accent error).

Varieties:
  es-MX  Mexico (central)           seseo, yeísmo [ʝ], j = [x]
  es-ES  Spain (central/Madrid)     distinción c/z = [θ], yeísmo [ʝ], j = [x]
  es-AR  Argentina (Buenos Aires)   seseo, ll/y = [ʃ] (sheísmo)
  es-CO  Colombia (Bogotá)          seseo, yeísmo [ʝ], j = [x]
  es-US  US / general Latin American, same as es-MX
"""

from dataclasses import dataclass, field
import re
import unicodedata

VARIETIES = ("es-MX", "es-ES", "es-AR", "es-CO", "es-US")

VOWELS = set("aeiou")
STRONG = set("aeo")
ACCENTED = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u"}

# Phones that are fine in place of the expected one. Keys are expected phones.
# Hard b/d/g between vowels are NOT here: that's a typical English-speaker habit,
# graded as a minor issue by compare.py.
FREE_VARIANTS = {
    "ʝ": {"j", "ɟʝ", "dʒ"},
    "s": {"z"},          # voicing before voiced consonants ("mismo")
    "ŋ": {"n"},
    "m": {"n"},          # nasal assimilation is gradient
    "x": {"h", "χ"},
    "ʃ": {"ʒ"},          # Buenos Aires: older [ʒ], younger [ʃ]
    "θ": {"s"},          # only reached for es-ES; seseo speakers exist in Spain too
    "j": {"i"}, "w": {"u"},  # glide vs. full vowel ("muy" [mwi] / [muj])
}

# Words whose spelling breaks the rules.
EXCEPTIONS = {
    "méxico": "ˈme.xi.ko", "mexicano": "me.xiˈka.no", "mexicana": "me.xiˈka.na",
    "oaxaca": "waˈxa.ka", "texas": "ˈte.xas",
}


@dataclass
class Word:
    text: str
    phones: list            # IPA phones in order
    stress: int             # index of the stressed syllable
    syllables: list         # list of (start, end) phone indexes, end exclusive
    variants: list = field(default_factory=list)  # per phone: set of acceptable alternatives


def _strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def normalize(text):
    """Lowercase words, keeping á é í ó ú ü ñ."""
    text = text.lower()
    text = re.sub(r"[^a-záéíóúüñ' ]+", " ", text)
    return text.split()


def _letters_to_units(word):
    """Split a written word into spelling units (digraphs kept together)."""
    units, i = [], 0
    while i < len(word):
        two = word[i:i + 2]
        if two in ("ch", "ll", "rr"):
            units.append(two); i += 2
        elif two == "qu" and i + 2 < len(word) and word[i + 2] in "eiéí":
            units.append("qu"); i += 2
        elif two == "gu" and i + 2 < len(word) and word[i + 2] in "eiéí":
            units.append("gu"); i += 2
        elif two == "gü":
            units.append("g"); units.append("u"); i += 2  # pingüino: [ɡw]
        else:
            units.append(word[i]); i += 1
    return units


def _syllabify(units):
    """Group spelling units into syllables. Returns list of lists of units.

    Standard Spanish rules: a consonant between vowels starts the next syllable;
    two consonants split unless they form an onset cluster (pr, bl, tr...);
    weak unstressed i/u next to another vowel form a diphthong.
    """
    def is_vowel(u):
        return _strip_accents(u) in VOWELS or u == "ü"

    def is_weak(u):
        return u in ("i", "u", "ü")  # accented í/ú are strong (hiatus)

    # Mark nuclei: indexes of vowel units, merging diphthongs/triphthongs.
    nuclei = []  # list of (start, end) unit index ranges
    i = 0
    while i < len(units):
        if is_vowel(units[i]):
            j = i + 1
            while j < len(units) and is_vowel(units[j]):
                a, b = units[j - 1], units[j]
                # hiatus: two strong vowels, or an accented weak vowel
                if not (is_weak(a) or is_weak(b)):
                    break
                if a in ("í", "ú") or b in ("í", "ú"):
                    break
                j += 1
            nuclei.append((i, j))
            i = j
        else:
            i += 1
    # "y" at the end of a word ("hoy", "muy", "y") is a vowel glide.
    if not nuclei:
        return [units]

    onset_ok = {"pr", "br", "tr", "dr", "cr", "gr", "fr", "pl", "bl", "cl", "gl", "fl", "kr", "kl"}
    syllables, start = [], 0
    for k, (ns, ne) in enumerate(nuclei):
        if k + 1 < len(nuclei):
            cons = units[ne:nuclei[k + 1][0]]
            if len(cons) == 0:
                cut = ne
            elif len(cons) == 1:
                cut = ne
            elif "".join(_strip_accents(c) for c in cons[-2:]) in onset_ok:
                cut = nuclei[k + 1][0] - 2
            else:
                cut = nuclei[k + 1][0] - 1
            syllables.append(units[start:cut]); start = cut
        else:
            syllables.append(units[start:])
    return syllables


def _stressed_syllable(word, syllables):
    for idx, syl in enumerate(syllables):
        if any(u in ACCENTED for u in syl):
            return idx
    if len(syllables) == 1:
        return 0
    last = _strip_accents(word)[-1]
    return len(syllables) - 2 if last in VOWELS or last in "ns" else len(syllables) - 1


def _unit_phones(u, nxt, prev, word_start, variety):
    """Phones for one spelling unit, before allophone rules."""
    base = _strip_accents(u) if u in ACCENTED else u
    distincion = variety == "es-ES"
    if base in VOWELS:
        return [base]
    if u == "ü":
        return ["u"]
    return {
        "ch": ["tʃ"], "ll": ["ʃ" if variety == "es-AR" else "ʝ"], "rr": ["r"],
        "qu": ["k"], "gu": ["ɡ"], "ñ": ["ɲ"], "h": [], "k": ["k"], "w": ["w"],
        "b": ["b"], "v": ["b"], "d": ["d"], "f": ["f"], "l": ["l"], "m": ["m"],
        "n": ["n"], "p": ["p"], "s": ["s"], "t": ["t"], "j": ["x"], "x": ["k", "s"],
        "z": ["θ" if distincion else "s"],
    }.get(u) or (
        ["θ" if distincion else "s"] if u == "c" and nxt and nxt[0] in "eiéí" else
        ["k"] if u == "c" else
        ["x"] if u == "g" and nxt and nxt[0] in "eiéí" else
        ["ɡ"] if u == "g" else
        ["r"] if u == "r" and (word_start or prev in ("l", "n", "s")) else
        ["ɾ"] if u == "r" else
        ["ʃ" if variety == "es-AR" else "ʝ"] if u == "y" else
        []
    )


def word_phones(word, variety="es-MX"):
    """Phones and syllables for a single word, before cross-word rules."""
    if word in EXCEPTIONS:
        return _from_ipa(word, EXCEPTIONS[word])
    units = _letters_to_units(word)
    # "y" as a vowel: alone, or at the end of a word after a vowel (hoy, muy).
    units = ["i" if u == "y" and (len(units) == 1 or (k == len(units) - 1 and k > 0)) else u
             for k, u in enumerate(units)]
    syllables = _syllabify(units)
    stress = _stressed_syllable(word, syllables)

    phones, spans, prev = [], [], None
    flat = [u for s in syllables for u in s]
    pos = 0
    for syl in syllables:
        start = len(phones)
        vowel_units = [k for k, u in enumerate(syl) if _strip_accents(u) in VOWELS or u == "ü"]
        # The syllable's nucleus is its strong (or accented) vowel; with only weak
        # vowels (ciudad, muy, pingüino) it's the last one. Other vowels are glides.
        strong = [k for k in vowel_units if syl[k] not in ("i", "u", "ü")]
        nucleus = strong[0] if strong else (vowel_units[-1] if vowel_units else None)
        for k, u in enumerate(syl):
            nxt = flat[pos + 1] if pos + 1 < len(flat) else None
            ps = _unit_phones(u, nxt, prev, pos == 0, variety)
            # Weak vowels in a diphthong are glides.
            if k in vowel_units and k != nucleus and u in ("i", "u", "ü"):
                ps = ["j" if u == "i" else "w"]
            phones.extend(ps)
            prev = u
            pos += 1
        spans.append((start, len(phones)))
    return Word(word, phones, stress, spans)


def _from_ipa(word, ipa):
    phones, spans, stress = [], [], 0
    for idx, syl in enumerate(ipa.split(".")):
        if syl.startswith("ˈ"):
            stress, syl = idx, syl[1:]
        start = len(phones)
        phones.extend(_split_ipa(syl))
        spans.append((start, len(phones)))
    return Word(word, phones, stress, spans)


def _split_ipa(s):
    out, i = [], 0
    while i < len(s):
        if s[i:i + 2] in ("tʃ", "ɟʝ", "dʒ"):
            out.append(s[i:i + 2]); i += 2
        else:
            out.append(s[i]); i += 1
    return out


def transcribe(text, variety="es-MX"):
    """Expected pronunciation of a phrase, with connected-speech rules applied.

    Rules across the whole phrase (Spanish links words together):
      - b/d/g are stops after a pause or a nasal (and d after l), otherwise
        approximants [β ð ɣ]
      - n becomes [m] before p/b/m and [ŋ] before k/g/x
      - s voices to [z] before voiced consonants (accepted, not required)
    """
    if variety not in VARIETIES:
        raise ValueError(f"unknown variety {variety}")
    words = [word_phones(w, variety) for w in normalize(text)]
    flat = [(wi, pi) for wi, w in enumerate(words) for pi in range(len(w.phones))]

    def phone(k):
        wi, pi = flat[k]
        return words[wi].phones[pi]

    for k, (wi, pi) in enumerate(flat):
        p = phone(k)
        prev = phone(k - 1) if k > 0 else None
        nxt = phone(k + 1) if k + 1 < len(flat) else None
        new = p
        if p in ("b", "d", "ɡ"):
            stop = prev is None or prev in ("m", "n", "ŋ") or (p == "d" and prev == "l")
            if not stop:
                new = {"b": "β", "d": "ð", "ɡ": "ɣ"}[p]
        elif p == "n" and nxt in ("p", "b", "m", "β"):
            new = "m"
        elif p == "n" and nxt in ("k", "ɡ", "x", "ɣ"):
            new = "ŋ"
        words[wi].phones[pi] = new

    for w in words:
        w.variants = [{p} | FREE_VARIANTS.get(p, set()) for p in w.phones]
    return words


def to_ipa(words):
    """Readable IPA string with syllable dots and stress marks."""
    out = []
    for w in words:
        syls = []
        for idx, (a, b) in enumerate(w.syllables):
            s = "".join(w.phones[a:b])
            syls.append(("ˈ" if idx == w.stress and len(w.syllables) > 1 else "") + s)
        out.append(".".join(syls))
    return " ".join(out)
