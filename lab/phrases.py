"""Test phrases for the Accent Lab.

Each phrase targets specific sounds. Learners record it twice: once as well as
they can, and once with the deliberate mistake described in `mistake`, so we
know the grader should find something in that take.
"""

PHRASES = [
    {"id": "perro", "text": "El perro corre rápido", "sounds": "rr (vibrante)",
     "mistake": "Di todas las r como en inglés (la r de 'red')."},
    {"id": "caro", "text": "Pero no es caro", "sounds": "r suave (un toque)",
     "mistake": "Di las r como en inglés americano."},
    {"id": "carro", "text": "El carro es caro", "sounds": "rr vs r",
     "mistake": "Di 'carro' con una r suave, igual que 'caro'."},
    {"id": "roberto", "text": "Roberto ríe en la radio", "sounds": "r inicial (vibrante)",
     "mistake": "Di las r iniciales como en inglés."},
    {"id": "todo", "text": "Todo lo que tengo es tuyo", "sounds": "o pura",
     "mistake": "Desliza las o hacia 'ou', como en inglés 'go'."},
    {"id": "nose", "text": "No sé, no lo sé", "sounds": "o y e puras",
     "mistake": "Di 'no' como en inglés y 'sé' como 'say'."},
    {"id": "casa", "text": "La casa es amarilla", "sounds": "a pura, vocales átonas",
     "mistake": "Reduce las vocales sin acento a 'uh' (como en inglés 'sofa')."},
    {"id": "como", "text": "¿Cómo estás?", "sounds": "vocales puras",
     "mistake": "Desliza la o final hacia 'ou' y la e hacia 'ei'."},
    {"id": "nada", "text": "Nada de nada", "sounds": "d suave",
     "mistake": "Di las d duras como en inglés ('daddy')."},
    {"id": "mesa", "text": "La mesa de madera", "sounds": "d suave, r suave",
     "mistake": "Di 'madera' con d dura y r inglesa."},
    {"id": "vaso", "text": "Un vaso de agua", "sounds": "b/g suaves, n",
     "mistake": "Di 'agua' con una g dura y la v como en inglés 'vase'."},
    {"id": "papa", "text": "Papá toma café con tu tía", "sounds": "p, t, k sin aire",
     "mistake": "Di las p, t y k con soplo de aire, como en inglés 'papa', 'top'."},
    {"id": "tio", "text": "Tu tío tiene tres tazas", "sounds": "t sin aire",
     "mistake": "Di las t con soplo de aire, como en inglés 'tea'."},
    {"id": "lola", "text": "Hablo con Lola todos los días", "sounds": "l clara",
     "mistake": "Di las l oscuras, como en inglés 'ball'."},
    {"id": "hijo", "text": "Mi hijo juega en el jardín", "sounds": "j",
     "mistake": "Di las j como una h inglesa muy suave, y 'mi' como 'my'."},
    {"id": "nino", "text": "El niño tiene sueño", "sounds": "ñ",
     "mistake": "Di la ñ como 'n' + 'y' separadas ('nin-yo')."},
    {"id": "llamo", "text": "Me llamo Yolanda", "sounds": "ll / y",
     "mistake": "Di 'llamo' como 'lamo', con l."},
    {"id": "ciudad", "text": "La universidad de la ciudad", "sounds": "d suave final, acento",
     "mistake": "Di 'universidad' con acento inglés: d dura al final y vocales reducidas."},
    {"id": "telefono", "text": "Mi teléfono no funciona", "sounds": "acento (sílaba fuerte)",
     "mistake": "Pon la fuerza en la sílaba equivocada: 'telefóno'."},
    {"id": "cafe", "text": "Quiero un café con leche", "sounds": "e pura, ch",
     "mistake": "Di 'café' como en inglés ('ca-fay') y 'leche' como 'lay-chay'."},
]

BY_ID = {p["id"]: p for p in PHRASES}
