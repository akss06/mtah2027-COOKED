# Team COOKED — Frame-level ball possession, BAR-ATH

*More than a Hack 2027, Phase 1. Technical document.*

## 1. Approach in one paragraph

For every frame we compute about 40 features from the tracked player and ball positions, a
gradient-boosted classifier turns them into P(home), P(away), P(dead), and a three-state Viterbi
decoder turns those per-frame probabilities into whole possessions. The classifier is trained only
on **other matches** — PFF FC's 64 World Cup 2022 matches and Metrica Sports' two public sample
games — with labels derived by code from their event data. Nothing is fitted or selected on the
competition match's ground truth.

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

The competition tracker is different from every tracker we can train on, so the training data was
chosen for how well a model **generalises to a tracker it has not seen** (section 7).

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
| PFF FC 2022 World Cup | PFF FC, free on request | PFF's event data; per-frame labels derived by our code | 64 matches, ~9.5 million frames at 25 fps |
| Metrica Sports sample games 1 and 2 | <https://github.com/metrica-sports/sample-data> (public) | Metrica's event data; per-frame labels derived by our code | 2 matches, ~286,000 frames |

No data about the competition match other than the provided bundle is used. No human decided any
per-frame label: every label comes from the providers' event logs through fixed rules.

### Labels from events

The same rule for both sources ([pff_games.py](pff_games.py), [sample_games.py](sample_games.py)):

- **Live frames** belong to the team that owns the ball at the most recent on-the-ball event.
  For PFF this is the possession-sequence owner (`home_ball`). For Metrica it is the team of the
  last pass, set piece, shot, recovery, ball lost or ball out. **Challenges are skipped**: both
  teams appear in them, and a challenge alone is not a change of possession (CHALLENGE.md: "a simple
  touch of the ball without control" does not count). A loose ball therefore stays with the last
  team that controlled it, as CHALLENGE.md specifies.
- **Dead frames** run from the end of the last event before a restart (kick-off, throw-in, goal
  kick, corner, free kick) to the start of that restart.
- Frames outside each period's first-to-last event are not graded.

Sanity of the result: a PFF match has a median of 271 possessions and 35% dead time; a Metrica
sample match about 370 possessions and 33% dead time.

### Matching the competition tracker

- **PFF** is broadcast tracking at 29.97 fps, resampled to 25 fps by nearest video time. Its ball
  is genuinely visible on a median 41% of frames, close to the competition tracker, so it is used
  as is. Only the raw `VISIBLE` ball is used; `ESTIMATED` balls count as missing. Players include
  the tracker's own estimates for off-screen players, which is closer to the competition's
  full-pitch view than dropping them. **PFF's "smoothed" positions are never used**: PFF builds them
  from its event data, so they would leak the labels into the features.
- **Metrica's** ball is tracked on ~87% of live play. Each game is used as recorded and again with
  the ball hidden in random 0.5–4 s gaps covering 40% of live play.
- Every match is also used with home and away swapped, so the model cannot learn a team bias.

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
  0.05, fixed seed), trained on every 10th PFF frame and every 5th Metrica frame — neighbours are
  near-duplicates.
- **Emissions:** the classifier's P(state | features) divided by the class prior (a scaled
  likelihood), then divided by a *temper* factor. Neighbouring frames are nearly copies of each
  other, so treating each as independent evidence overcounts it and makes the decoder flip on short
  runs.
- **Transitions:** counted per frame from the training labels.
- **Decoding:** Viterbi over H / A / D separately in each half; frames outside the halves are `X`.

**The one free setting, temper, is chosen on 11 PFF matches held out from training** (every sixth
by id). spellF1 was computed with the organisers' own `possession_metrics.py`:

| temper | 1 | **2** | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| mean held-out spellF1 | 0.539 | **0.550** | 0.547 | 0.530 | 0.477 | 0.392 | 0.280 |

The final model is then retrained on all 64 PFF matches plus both Metrica games.

## 7. What we tried and dropped

### Choice of training data

We first trained on Metrica's two games only, then on PFF only, then on both. Each was tested on
matches it had not trained on, at temper 2:

| trained on | test: Metrica 1 | test: Metrica 2 | test: held-out PFF | worst |
|---|---|---|---|---|
| Metrica only | 0.582* | 0.568* | 0.427 | 0.427 |
| PFF only | 0.463 | 0.513 | 0.552 | 0.463 |
| **PFF + Metrica** | **0.516**† | **0.516**† | **0.550** | **0.516** |

\* trained on the other Metrica game: same tracking provider. † trained on PFF plus the other
Metrica game.

Every model loses accuracy on a tracker it has not seen. The competition tracker is unseen by all
of them, so we chose on the worst case, where training on both is clearly best, at almost no cost
on PFF.

### Other ideas

| idea | result | why dropped |
|---|---|---|
| Nearest-player HMM with hand-set probabilities | 996 possessions on BAR-ATH, heavy flicker | probabilities were guesses; replaced by values counted from labelled matches |
| Nearest-player HMM, missing-ball rate fitted by EM on BAR-ATH | EM collapsed to "ball missing = dead ball" | not identifiable from one observation; replaced by the feature classifier |
| Separate ball-blind "is play stopped?" model (Metrica training) | held-out spellF1 0.575 → 0.526; no change in stoppages detected on BAR-ATH | did not help |
| Training with a still ball placed at each restart spot, plus a ball-speed feature (Metrica training) | no change on BAR-ATH | did not help |

The last two came from a diagnostic. Using Smart Tagging set-piece times as an independent,
automatic reference, the Metrica-only model detected corners, free kicks and kick-offs after goals
as stoppages almost every time, but missed some goal kicks and throw-ins. In those windows the
tracking looks exactly like live play: the ball is visible and moving at ~4.5 m/s, and players run
at ~2 m/s. Training on PFF raised the share of pre-restart frames called dead from 65% to 73%,
and for goal kicks from 25% to 46%.

## 8. Result

On BAR-ATH the submission reports 236 possessions with a mean of 15 s, and calls 44% of
graded-half frames dead. That is higher than the 33–35% typical of the training matches, but
within PFF's range of 24–46%.

Pipeline check on the public 5 minutes (17 possessions — a sanity check, not a measurement; one
possession moves spellF1 by ~0.06):

| | spellF1 | macroF1 | possessions (truth 17) |
|---|---|---|---|
| **ours** | **0.607** | **0.833** | 12 |
| baseline_smooth125 | 0.424 | 0.581 | 16 |
| baseline_smooth50 (the published floor's setting) | 0.326 | 0.621 | 26 |

## 9. Limitations

- **Domain gap.** Neither training source uses the competition tracker. PFF is broadcast tracking
  with most players estimated off-screen; Metrica's sample is clean tracking with the ball removed
  during stoppages. The worst-case test in section 7 is our best estimate of how much that costs.
- **Short stoppages.** Quick throw-ins and goal kicks whose tracking looks like live play can be
  labelled as possession.
- **Possibly too many dead-ball frames** on BAR-ATH (section 8). It may also be a real feature of
  this match; we have no way to check without using the ground truth.
- **Event-derived labels are an approximation** of the corrected-tracking definition used for the
  ground truth, especially around loose balls and the exact frame play stops and restarts.
- **No player identities** are used; everything is team-level, as the ATD provides.

## 10. Reproduction

See [README.md](README.md). In short: Python 3.13, `pip install -r solution/requirements.txt`,
get the Metrica sample data and the PFF dataset, set `MTAH_DATA_DIR`, run
`python solution/pff_games.py` once (~15 min), then `python solution/run.py` (~17 min). Two runs
produce byte-identical files.
