# Team COOKED — Frame-level ball possession, BAR-ATH

*More than a Hack 2027, Phase 1. Technical document. Draft — sections marked **TODO** change as
the model does.*

## 1. Approach in one paragraph

For every frame we compute about 40 features from the tracked player and ball positions, a
gradient-boosted classifier turns them into P(home), P(away), P(dead), and a three-state Viterbi
decoder turns those per-frame probabilities into whole possessions. The classifier is trained only
on **other matches** — Metrica Sports' public sample games — with labels derived by code from their
event data. Nothing is fitted or selected on the competition match's ground truth.

## 2. Why this design

The ranking metric, spellF1, scores whole possessions, so the main failure to avoid is
*fragmentation*: one frame where a defender is nearest the ball splits one possession into three.
The organisers' baseline shows it — the same nearest-player rule scores 0.05 unsmoothed and 0.24
smoothed on the full match.

A per-frame smoothing window fixes this crudely and has to be tuned. A hidden Markov model fixes it
in a principled way: a change of state has a cost, and that cost is the rate at which possession
actually changes in labelled matches. The decoder only switches team when the evidence for the
other team is sustained.

Per-frame evidence comes from a classifier rather than from "who is nearest the ball" alone,
because the ball is missing on 41% of graded frames. On those frames the nearest-player rule has
no answer, but player speed and team shape still carry signal: the team in possession spreads out
and pushes up, the other compacts, and everyone slows down during a stoppage.

## 3. Assets

| asset | used? | how |
|---|---|---|
| Tactical ATD (tracking) | **yes** | the only input to the pipeline |
| Smart Tagging | analysis only | set-piece times were used to diagnose where stoppages are missed (section 7); not an input |
| Homography | no | the ATD is already in pitch coordinates; nothing needed pixels |
| Broadcast ATD and video | no | not frame-aligned to the tactical video; nothing gained over tactical ATD yet |
| Tactical video | no | — |
| Public 5-minute ground truth | pipeline check only | confirming frames are aligned and labels mean what we think; never used to choose anything |

## 4. Training data

| dataset | source | who labelled it | amount |
|---|---|---|---|
| Metrica Sports sample games 1 and 2 | <https://github.com/metrica-sports/sample-data> (public) | Metrica's event data (passes, recoveries, set pieces, ball out); per-frame possession labels derived from it by our code, no human decisions | 2 matches, ~286,000 frames |
| **TODO** PFF FC 2022 World Cup | form-gated, free | PFF event data | 64 matches — permission requested from PFF and organisers |

No data about the competition match other than the provided bundle is used.

### Labels from events

Each sample frame gets H / A / D by fixed rules ([sample_games.py](sample_games.py)):

- **Live frames** belong to the team of the most recent *possession-asserting* event: pass, set
  piece, shot, recovery, ball lost, ball out. **Challenges are skipped** — both teams appear in
  them, and a challenge alone is not a change of possession (CHALLENGE.md: "a simple touch of the
  ball without control" does not count). A loose ball therefore stays with the last team that
  controlled it, as CHALLENGE.md specifies.
- **Dead frames** run from the end of the last event before a set piece to the start of that set
  piece.
- Frames outside each period's first-to-last event are not graded.

This gives about 370 possessions and 75 stoppages per sample match.

### Making the sample games look like the competition tracker

The sample games' ball is tracked on ~87% of live play; the competition tracker loses it far more
often. Each sample game is used twice for training: as recorded, and with the ball hidden in
random 0.5–4 s gaps covering 40% of live play. Every copy is also used with home and away swapped,
so the model cannot learn a team bias.

## 5. Features

Per frame, computed per half so nothing crosses halftime ([features.py](features.py)). Each team is
oriented to attack +x, using the side its players stand on on average in that half.

- **Ball vs players:** each team's nearest distance to the ball and its difference; players within
  5 m of the ball.
- **Speed:** mean player speed per team (finite differences; jumps over 12 m/s from identity
  switches discarded).
- **Team shape:** mean and most-advanced x, spread in x and y, forward velocity of the team centre,
  tracked player count.
- **Ball context:** whether the ball is tracked, frames since it was last seen, and how far its
  last seen position is from the touchline and goal line.
- 1 s and 5 s centred rolling means of the speed, shape, velocity and ball-visibility features.

## 6. Model and decoding

- **Classifier:** scikit-learn `HistGradientBoostingClassifier` (300 iterations, learning rate
  0.05, fixed seed), trained on every 5th frame — neighbours are near-duplicates.
- **Emissions:** the classifier's P(state | features) divided by the class prior (a scaled
  likelihood), then divided by a *temper* factor. Neighbouring frames are nearly copies of each
  other, so treating each as independent evidence overcounts it and makes the decoder flip on short
  runs.
- **Transitions:** counted per frame from the sample labels. Possession changes about once every
  330 frames.
- **Decoding:** Viterbi over H / A / D separately in each half; frames outside the halves are `X`.

**The one free setting, temper, is chosen leave-one-game-out:** train on sample game 1, score
game 2 (made competition-like as above), then the reverse. spellF1 was computed with the
organisers' own `possession_metrics.py`:

| temper | 1 | **2** | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| mean held-out spellF1 | 0.552 | **0.575** | 0.568 | 0.545 | 0.520 | 0.434 | 0.331 |

Settings 2–4 are equally good on both held-out games; 2 is used.

## 7. What we tried and dropped

| idea | result | why dropped |
|---|---|---|
| Nearest-player HMM with hand-set probabilities | 996 possessions on BAR-ATH, heavy flicker | probabilities were guesses; replaced by values counted from labelled matches |
| Nearest-player HMM, missing-ball rate fitted by EM on BAR-ATH | EM collapsed to "ball missing = dead ball" | not identifiable from one observation; replaced by the feature classifier |
| Separate ball-blind "is play stopped?" model | held-out spellF1 0.575 → 0.526; no change in stoppages detected on BAR-ATH | did not help |
| Training with a still ball placed at each restart spot, plus a ball-speed feature | no change on BAR-ATH | did not help |

The last two came from a diagnostic. Using Smart Tagging set-piece times as an independent,
automatic reference, corners, free kicks and kick-offs after goals are detected as stoppages almost
every time. Some goal kicks and throw-ins are not. In those windows the tracking looks exactly like
live play: the ball is visible and moving at ~4.5 m/s, and players run at ~2 m/s. Either the
stoppage really lasts only a moment, or the tag time is off by a second or two. Either way, the
tracking alone cannot separate them.

## 8. Result

On BAR-ATH the submission reports about 320 possessions with a mean of 13 s, and calls about a
third of graded-half frames dead — in line with the sample matches.

Pipeline check on the public 5 minutes (17 possessions — a sanity check, not a measurement; one
possession moves spellF1 by ~0.06):

| | spellF1 | macroF1 | possessions (truth 17) |
|---|---|---|---|
| ours | 0.489 | 0.694 | 19 |
| baseline_smooth125 | 0.424 | 0.581 | 16 |
| baseline_smooth50 (the published floor's setting) | 0.326 | 0.621 | 26 |

**TODO** update after the final model.

## 9. Limitations

- **Two training matches.** Labels, transitions and the temper choice all come from them; more
  matches is the most direct improvement (PFF data, in progress).
- **Domain gap.** The sample games come from a different tracker and the ball-loss simulation is
  only an approximation of the competition tracker.
- **Short stoppages.** Quick throw-ins and goal kicks whose tracking looks like live play are
  labelled as possession (section 7).
- **Event-derived labels are an approximation** of the corrected-tracking definition used for the
  ground truth, especially around loose balls and the exact frame play stops and restarts.
- **No player identities** are used; everything is team-level, as the ATD provides.

## 10. Reproduction

See [README.md](README.md). In short: Python 3.13, `pip install -r solution/requirements.txt`, clone
the Metrica sample data, set `MTAH_DATA_DIR` (and `METRICA_SAMPLE_DIR` if not in the default
place), then `python solution/run.py`. It takes about 3 minutes, and two runs produce
byte-identical files.
