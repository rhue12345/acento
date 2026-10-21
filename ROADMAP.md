# Acento roadmap

Goal: an accent trainer for Spanish learners whose grading is accurate enough that a native speaker agrees with it. Built on free, open-source models so it costs nothing to run.

Native-speaker review is the bar for every phase: if the grader says a sound is right and it doesn't sound right to a native ear, the grader is wrong.

## How the grader works

1. **Expected sounds.** A Spanish rule set turns each practice phrase into the sounds a native speaker of the chosen region would make (IPA), including allophones: tap [ɾ] vs trill [r], soft [β ð ɣ] between vowels, seseo vs distinción, yeísmo / Rioplatense [ʃ].
2. **What you said.** An open-source phoneme recognizer (wav2vec2 trained on phonetic transcription) listens to the recording and outputs the sounds actually produced.
3. **Comparison.** The two sequences are aligned sound by sound; each mismatch becomes a specific note ("[ɾ] where *perro* needs [r]").
4. **Acoustic checks** for what the recognizer can't hear well: aspiration of p/t/k (voice onset time), vowel glides like "o" → "ow" (formant movement), stress and rhythm (pitch, loudness, syllable length).

## Phases

### 0. Feasibility
- [x] Generate expected Spanish phonemes (espeak-ng works; needs Spanish-specific fixes)
- [x] Run the phoneme recognizer (works; smoke test on espeak audio: 1 false alarm on native-style speech, 21 catches on English-accented speech)
- [x] Custom Spanish pronunciation rules with regional variants (es-MX, es-ES, es-AR, es-CO, es-US)

### 1. Accent Lab (testing tool)
Status: built (`lab/`). Next step: deploy with `HF_TOKEN=... python lab/deploy.py rhu3`
(needs the HF_TOKEN environment variable), then record and rate.

A page for building and checking the grader, used by the two of us:
- Record a test phrase, see expected vs. heard sounds side by side.
- **Rating mode:** a native speaker listens and marks each word or sound as native / not native.
- Agreement report: how often the grader catches what the native speaker heard (misses) and how often it flags what was fine (false alarms), per sound.
- Test set: each phrase recorded clean, and with deliberate English-speaker mistakes, plus native recordings.

### 2. Make the grader trustworthy
- Tune per-sound thresholds against the ratings until agreement is high for every sound in the course.
- Add the acoustic checks above.
- If needed, fine-tune the recognizer on our own rated recordings.
- Nothing ships to the trainer until a sound's grading agrees with the native speaker.

### 3. The accent trainer
- **Diagnóstico:** read a short paragraph, get a personal plan of the sounds to fix.
- **Sound units** (r/rr, pure vowels, soft d/b/g, unaspirated p/t/k, j, ñ, ll/y, stress, linking and rhythm), each with:
  - Aprende: how to make the sound, native audio
  - Escucha: ear training with minimal pairs (pero / perro)
  - Pronuncia: minimal pairs → words → phrases, graded
  - Imita: shadowing with your pitch line over the native one
- **Práctica en contexto:** the conversation scenes, coaching only the sounds you're working on.
- **Progreso:** a score per sound over time.

### 4. Run it for free
- In the browser (model converted to ONNX), or on a free Hugging Face Space if it's too heavy for phones.
