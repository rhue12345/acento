from grader.compare import diagnose, normalize_heard


def kinds(text, heard, variety="es-MX"):
    _, issues, _ = diagnose(text, heard.split(), variety)
    return [(i.word, i.kind, i.severity) for i in issues]


def test_native_has_no_issues():
    assert kinds("el perro corre", "e l p e r o k o r e") == []


def test_tapped_rr():
    assert kinds("el perro", "e l p e ɾ o") == [("perro", "rr_tap", "error")]


def test_english_r_and_glide():
    assert ("pero", "r_english", "error") in kinds("pero no", "p e ɹ oʊ n o")
    assert ("pero", "vowel_glide", "error") in kinds("pero no", "p e ɹ oʊ n o")


def test_aspiration_and_dark_l():
    ks = kinds("tal", "tʰ a ɫ")
    assert ("tal", "aspirated", "error") in ks and ("tal", "dark_l", "error") in ks


def test_hard_d_between_vowels_is_minor():
    assert kinds("nada", "n a d a") == [("nada", "hard_stop", "minor")]


def test_free_variants_accepted():
    assert kinds("ciudad", "s j u ð a ð") == []
    assert kinds("ciudad", "θ j u ð a ð", "es-ES") == []


def test_normalize_strips_tones_and_length():
    assert normalize_heard(["a1", "oː", "g", "<pad>", "e̞"]) == ["a", "o", "ɡ", "e"]
