# Acento

A Spanish conversation coach. Pick a scene (café, taxi, market…), talk out loud, and an AI partner answers in spoken Spanish while a coach flags the words that didn't come through clearly, with a tip and a "try again" button for each.

Live app: https://rhue12345.github.io/acento/

- `index.html` – conversation coach (Claude via the Anthropic API)
- `drills.html` – sound-by-sound pronunciation drills
- `vendor/anthropic-sdk.mjs` – browser bundle of `@anthropic-ai/sdk` 0.131.0

## Setup

Open the app in Chrome or Edge, click **Settings**, choose a model (Claude Haiku 5.5 by default, Claude Opus 5.5 for the strongest feedback), and paste an Anthropic API key from https://console.anthropic.com/settings/keys. The key is stored in your browser's localStorage and sent only to api.anthropic.com. This is fine for personal use; before sharing the app publicly, move the API call behind a small server so the key isn't in the browser.

## Natural voices (ElevenLabs)

Paste an ElevenLabs API key in **Ajustes** and click **Cargar mis voces**. Add voices with the accent you're practicing from the ElevenLabs Voice Library first (e.g. search "Mexican Spanish"). On "Automática", the app picks a voice from your library whose labels match the selected accent and the character's gender (Lucía, Marisol, Vale, Carmen are female; Ramón and Andrés are male). Each line is generated once and replays from memory. Without a key, or if ElevenLabs fails, it falls back to the best Spanish browser voice.

## How pronunciation feedback works

The browser's Spanish speech recognizer returns several guesses with confidence scores. Claude compares those to what you most likely meant in context (e.g. "perro" heard as "pero" → weak rr trill) and reports at most two issues per turn. It's inferred from recognition errors, so it catches sounds that change words, not subtle accent details.
