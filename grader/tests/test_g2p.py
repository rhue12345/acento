import pytest
from grader.g2p_es import transcribe, to_ipa

CASES = [
    ("es-MX", "el perro corre rápido", "el ˈpe.ro ˈko.re ˈra.pi.ðo"),
    ("es-MX", "pero no es caro", "ˈpe.ɾo no es ˈka.ɾo"),
    ("es-MX", "nada de nada", "ˈna.ða ðe ˈna.ða"),
    ("es-MX", "todo lo que tengo", "ˈto.ðo lo ke ˈteŋ.ɡo"),
    ("es-MX", "un vaso de agua", "um ˈba.so ðe ˈa.ɣwa"),
    ("es-MX", "lo bebo", "lo ˈβe.βo"),
    ("es-MX", "el lago", "el ˈla.ɣo"),
    ("es-MX", "honrado", "on.ˈra.ðo"),
    ("es-MX", "ciudad", "sju.ˈðað"),
    ("es-ES", "ciudad", "θju.ˈðað"),
    ("es-ES", "zapato", "θa.ˈpa.to"),
    ("es-MX", "zapato", "sa.ˈpa.to"),
    ("es-MX", "ella llega", "ˈe.ʝa ˈʝe.ɣa"),
    ("es-AR", "ella llega", "ˈe.ʃa ˈʃe.ɣa"),
    ("es-AR", "yo", "ʃo"),
    ("es-MX", "hoy muy bien", "oj mwi βjen"),
    ("es-MX", "guitarra", "ɡi.ˈta.ra"),
    ("es-MX", "pingüino", "piŋ.ˈɡwi.no"),
    ("es-MX", "queso", "ˈke.so"),
    ("es-MX", "gente", "ˈxen.te"),
    ("es-MX", "mi hijo juega", "mi ˈi.xo ˈxwe.ɣa"),
    ("es-MX", "niño", "ˈni.ɲo"),
    ("es-MX", "país", "pa.ˈis"),
    ("es-MX", "día", "ˈdi.a"),
    ("es-MX", "árbol", "ˈaɾ.βol"),
    ("es-MX", "canción", "kan.ˈsjon"),
    ("es-MX", "México", "ˈme.xi.ko"),
    ("es-MX", "chocolate", "tʃo.ko.ˈla.te"),
    ("es-MX", "el doctor", "el dok.ˈtoɾ"),
    ("es-MX", "aldea", "al.ˈde.a"),
    ("es-MX", "hablar", "a.ˈβlaɾ"),
    ("es-MX", "Israel", "is.ra.ˈel"),
]

@pytest.mark.parametrize("variety,text,expected", CASES)
def test_transcription(variety, text, expected):
    assert to_ipa(transcribe(text, variety)) == expected

def test_variants_accept_stops_for_approximants():
    w = transcribe("nada", "es-MX")[0]
    assert "d" in w.variants[2]
