# Pre-registered betting test

Written and committed **before** looking at the holdout games below. The rule
must not be changed after seeing the results; any new idea gets its own new
holdout.

## Where the idea came from

A first sample of 1,600 DraftKings games (2026–2027), collected before this
file was written, suggested the opening line tends to move toward the "full"
model (efficiency + rest + key players who missed their last game), most of
all when a key player had sat out.

## Holdout

Every D-I vs D-I DraftKings game with an opening and closing spread that was
**not** in that first sample, i.e. game ids that are not in the first 9,772
lines of `data/lines_raw.jsonl`. These were collected after the idea was formed.

## Rules (bet the model's side at the opening spread, -110)

- **Primary, Rule A:** a key player missed one team's previous game
  (|miss_last_diff| ≥ 5 Game Score) **and** the full model is at least 1 point
  off the opening line.
- **Secondary, Rule B:** every game where the full model is 0–4 points off the
  opening line.

Pushes are excluded. Break-even is 52.4%; the target is 55%.

## How it will be judged

- Win rate, number of bets, and the 95% range around the win rate.
- "Promising": at least 55% and above 52.4% by 2+ standard errors.
- "Not shown": anything else. With a few hundred bets the 95% range is about
  ±5 points, so a small holdout may be unable to tell 55% from break-even. If
  so, that will be said plainly and the rule keeps running on new games as
  they are played, without changes.
