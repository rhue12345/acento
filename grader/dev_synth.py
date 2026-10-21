"""Development only: synthesize test speech with espeak-ng.

espeak-ng is robotic, but it's phonetically controlled, which makes it a useful
smoke test before real recordings exist:
  - the Spanish voice reading Spanish ~ "native" (clean Spanish phones)
  - the English voice reading the same Spanish text ~ a heavy English accent

    python -m grader.dev_synth "el perro corre" --voice en-us out.wav
"""

import argparse
import ctypes

import numpy as np


def synth(text, voice="es"):
    import espeakng_loader
    lib = ctypes.cdll.LoadLibrary(espeakng_loader.get_library_path())
    AUDIO_OUTPUT_SYNCHRONOUS = 0x02
    sr = lib.espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, espeakng_loader.get_data_path().encode(), 0)
    if sr <= 0:
        raise RuntimeError("espeak-ng failed to initialize")
    samples = []
    CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.c_void_p)

    def cb(wav, n, events):
        if wav and n > 0:
            samples.append(np.ctypeslib.as_array(wav, shape=(n,)).copy())
        return 0

    callback = CB(cb)
    lib.espeak_SetSynthCallback(callback)
    if lib.espeak_SetVoiceByName(voice.encode()) != 0:
        raise RuntimeError(f"unknown espeak voice {voice}")
    data = text.encode("utf-8")
    espeakCHARS_UTF8 = 1
    lib.espeak_Synth(data, len(data) + 1, 0, 0, 0, espeakCHARS_UTF8, None, None)
    lib.espeak_Synchronize()
    audio = np.concatenate(samples).astype(np.float32) / 32768 if samples else np.zeros(0, np.float32)
    return audio, sr


def main():
    import soundfile as sf
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("out")
    ap.add_argument("--voice", default="es")
    args = ap.parse_args()
    audio, sr = synth(args.text, args.voice)
    sf.write(args.out, audio, sr)


if __name__ == "__main__":
    main()
