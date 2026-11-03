# Team COOKED — Frame-level ball possession, BAR-ATH

*More than a Hack 2027, Phase 1. Technical document.*

## 1. Approach in one paragraph

For every frame we compute about 50 features from the tracked player and ball positions, a
gradient-boosted classifier turns them into P(home), P(away), P(dead), and a three-state Viterbi
decoder turns those per-frame probabilities into whole possessions. The classifier is trained only
on **other matches**: the 7 matches of the IDSSE open dataset (official Bundesliga tracking with
per-frame ball-status and possession flags) and Metrica Sports' two public sample games. Nothing
is fitted, tuned or selected on the competition match's ground truth or video.

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
no answer, but player speed, team shape and where the ball was last and next seen still carry
signal.

The competition tracker is different from every tracker we can train on, so choices were made on
how well a model does on **matches and trackers it has not seen** (sections 6 and 7).

## 3. Assets

| asset | used? | how |
|---|---|---|
| Tactical ATD (tracking) | **yes** | the only input to the pipeline |
| Smart Tagging | automatic checks only | set-piece times were used in automatic diagnostics of where stoppages are missed; not an input |
| Homography | no | the ATD is already in pitch coordinates |
| Broadcast ATD and video | no | not frame-aligned to the tactical video |
| Tactical video | understanding only | watched to understand what counts as a stoppage; not an input |
| Public 5-minute ground truth | pipeline check only | confirming frames are aligned and labels mean what we think; never used to choose anything |

## 4. Training data

| dataset | source | licence | who labelled it | amount |
|---|---|---|---|---|
| IDSSE open data: 7 matches, Bundesliga / 2. Bundesliga 2022/23 | Bassek, M. et al. (2025). *An integrated dataset of spatiotemporal and event data in elite soccer.* Scientific Data 12, 195. <https://doi.org/10.1038/s41597-025-04505-y> (data owner DFL; tracking by Sportec Solutions) | CC BY 4.0 | DFL's official `BallStatus` and `BallPossession` flags in the position data | 7 matches, ~1.02 million frames at 25 fps |
| Metrica Sports sample games 1 and 2 | <https://github.com/metrica-sports/sample-data> | public sample data | Metrica's event data; per-frame labels derived by our code | 2 matches, ~286,000 frames |

No data about the competition match other than the provided bundle is used. No human decided any
per-frame label.

### Labels

- **IDSSE** ([idsse_games.py](idsse_games.py)): straight from the ball's per-frame flags in the
  official position data. `BallStatus = 0` → `D`; otherwise `BallPossession` 1 → `H`, 2 → `A`.
  This is the same kind of source as the competition ground truth, which comes from corrected
  tracking's dead-ball flag and owner. As in the public ground truth, the owner can change on a
  single touch: the median IDSSE possession lasts 3.8 s, and 41–52% of frames are dead.
- **Metrica** ([sample_games.py](sample_games.py)): live frames belong to the team of the most
  recent pass, set piece, shot, recovery, ball lost or ball out. Challenges are skipped, so a
  loose ball stays with the last team that had it. Dead frames run from the end of the last event
  before a restart to the restart itself.

### Matching the competition tracker

Both sources track the ball almost always; the competition tracker loses it on 41% of graded
frames, and on about 73% of the seconds before a tagged restart (aggregates of the provided input
data, not of any labels). Each match is therefore used twice: as recorded, and with the ball hidden
in random 0.5–4 s gaps on 25% of live and 70% of dead frames, which matches that overall rate.
Every copy is also used with home and away swapped, so the model cannot learn a team bias.

## 5. Features

Per frame, computed per half so nothing crosses halftime ([features.py](features.py)). Each team is
oriented to attack +x, using the side its players stand on on average in that half.

- **Ball vs players:** each team's nearest distance to the ball and its difference; players within
  5 m of the ball.
- **Speed:** mean player speed per team (finite differences; jumps over 12 m/s from identity
  switches discarded).
- **Team shape:** mean and most-advanced x, spread in x and y, forward velocity of the team centre,
  tracked player count.
- **Ball context:** whether the ball is tracked, frames since it was last seen and until it is next
  seen, and how far its last seen position is from the touchline and goal line.
- **Across ball gaps:** each team's distance to the ball at the last and the next frame the ball
  is seen, and its mean over the 2 s before and the 2 s after the frame. These carry possession
  through the frames without a ball, and a hand-over shows up as "close before, far after".
- 1 s and 5 s centred rolling means of the speed, shape, velocity and ball-visibility features;
  10 s means of speed and ball visibility.

Using frames after the one being labelled is allowed: the task is post-match.

## 6. Model and decoding

- **Classifier:** scikit-learn `HistGradientBoostingClassifier` (600 iterations, 63 leaves per
  tree, learning rate 0.05, fixed seed), trained on every 5th frame — neighbours are
  near-duplicates. The size was chosen leave-one-match-out (section 7).
- **Emissions:** the classifier's P(state | features) divided by the class prior (a scaled
  likelihood), then divided by a *temper* factor. Neighbouring frames are nearly copies of each
  other, so treating each as independent evidence overcounts it and makes the decoder flip on short
  runs.
- **Transitions:** counted per frame from the training labels.
- **Decoding:** Viterbi over H / A / D separately in each half; frames outside the halves are `X`.

**The one free setting, temper, is chosen leave-one-match-out** over the 9 training matches: each
match, with its ball hidden as above, is scored by a model trained on the other 8. spellF1 is
computed with the organisers' own `possession_metrics.py`:

| temper | 1 | **2** | 4 | 8 | 16 |
|---|---|---|---|---|---|
| mean held-out spellF1 | 0.568 | **0.569** | 0.558 | 0.531 | 0.474 |
| worst held-out match | 0.498 | 0.502 | 0.491 | 0.452 | 0.363 |

Per held-out match at temper 2: IDSSE 0.502–0.615, Metrica 0.657 and 0.643. The final model is
then trained on all 9 matches.

## 7. What we tried and dropped

### Training data

We first trained on Metrica alone, then added PFF FC's 64 World Cup 2022 matches (free on request,
broadcast tracking plus event data), which helped a lot. PFF's terms of use, however, restrict its
data to personal, non-commercial use and prohibit using it to train machine-learning models, so we
dropped it and replaced it with IDSSE, which is CC BY 4.0. On exactly the same held-out cases (ball
hidden as in section 4, temper 2):

| trained on | 7 IDSSE matches (mean) | Metrica 1 | Metrica 2 | mean | worst |
|---|---|---|---|---|---|
| PFF (64) + Metrica | 0.472 | 0.599 | 0.619 | 0.502 | 0.417 |
| **IDSSE + Metrica** (kept) | **0.534** | **0.643** | **0.641** | **0.558** | **0.476** |

For each Metrica test game both models were trained without it but with the other Metrica game,
so those two columns are the fairest comparison. The IDSSE column favours the IDSSE model, which
had the other six IDSSE matches in training. We believe the gain also comes from the labels:
IDSSE's per-frame official flags are closer to how the competition ground truth is defined than
labels reconstructed from event logs. (This comparison used the earlier, smaller classifier of
300 iterations and 31 leaves.)

### Classifier size and decoder (IDSSE + Metrica, leave-one-match-out)

Most errors are short possessions: 29% of true possessions last under 2 s, and the model found
3% of those under 1 s and 16% of those of 1–2 s. Re-tuning the decoder (temper 1–3 × team-switch
cost ×1/3/10) left the mean within 0.551–0.560: flat. A larger classifier helped on every match:

| classifier | 7 IDSSE matches (mean) | Metrica 1 | Metrica 2 | mean | worst | training time |
|---|---|---|---|---|---|---|
| 300 iterations, 31 leaves | 0.534 | 0.643 | 0.641 | 0.558 | 0.476 | 1× |
| **600 iterations, 63 leaves** (submitted) | **0.545** | **0.657** | **0.643** | **0.569** | **0.502** | 2× |
| 1200 iterations, 127 leaves | 0.551 | 0.655 | 0.658 | 0.575 | 0.486 | 7× |

600 / 63 improved 8 of 9 matches and left the ninth unchanged. 1200 / 127 raised the mean by a
further 0.006, inside noise, but lowered the worst match, so we kept 600 / 63.

### Features and decoder (measured with PFF + Metrica training, before the switch)

On 13 held-out cases (11 PFF matches, Metrica 1 and 2 under leave-one-Metrica-out), temper 2:

| change | held-out PFF | Metrica 1 | Metrica 2 | worst |
|---|---|---|---|---|
| before | 0.550 | 0.494 | 0.480 | 0.480 |
| separate team-switch and live↔dead switch costs (best of 60 settings) | 0.553 | 0.503 | 0.490 | 0.490 |
| **"across ball gaps" features** (adopted) | **0.574** | **0.559** | **0.535** | **0.535** |
| + restart features: each team's speed over the 2 s before and after, ball speed and its peak over the next 1 s | 0.574 | 0.570 | 0.555 | 0.555 |

A diagnosis on the same cases explains the pattern. The model found 2,900 possessions where the
truth has 3,946, mostly missing short ones (16% of true possessions last under 2 s), and split
stoppages into too many pieces (1,465 dead spells against 1,211). Re-weighting the decoder cannot
recover evidence the classifier never had, so that change (+0.01, inside noise) was dropped.
Features that carry the ball across gaps can; with them, dead spells drop to 1,278.

The restart features were aimed at short bursts of play between two stoppages (a corner, a few
seconds of play, a goal kick), which the model often merges into one stoppage: 43% of live
stretches under 5 s between two stoppages were swallowed whole. They did not change that (44%
after), their gain came from the two Metrica games only, and +0.02 is at the edge of noise. We
dropped them rather than risk fitting two matches.

### Other ideas

| idea | result | why dropped |
|---|---|---|
| Nearest-player HMM with hand-set probabilities | 996 possessions on BAR-ATH, heavy flicker | probabilities were guesses; replaced by values learned from labelled matches |
| Nearest-player HMM, missing-ball rate fitted by EM on BAR-ATH | EM collapsed to "ball missing = dead ball" | not identifiable from one observation; replaced by the feature classifier |
| Separate ball-blind "is play stopped?" model (Metrica training) | held-out spellF1 0.575 → 0.526 | did not help |
| Training with a still ball placed at each restart spot, plus a ball-speed feature (Metrica training) | no change | did not help |
| Filling the ATD's ball gaps with a pretrained ball detector (RT-DETR r18, Apache-2.0, published COCO / Objects365 weights, no training) run on the tactical video | found a ball on ~41% of in-play frames where the ATD has none; where both exist, median 1.2 m apart, but 23% of detections were over 10 m off (it loses fast and lofted balls) | too noisy where it matters most, and no training match has video, so the model could not learn how far to trust it; not used |

An automatic diagnostic using Smart Tagging set-piece times showed that some goal kicks and
throw-ins look exactly like live play in the tracking: the ball is visible and moving at ~4.5 m/s,
and players run at ~2 m/s. The tracking alone cannot separate those.

## 8. Result

On BAR-ATH the submission reports 333 possessions with a mean of 11 s, and calls 42% of
graded-half frames dead, in line with the 41–52% of the IDSSE matches.

Pipeline check on the public 5 minutes (17 possessions — a sanity check, not a measurement; one
possession moves spellF1 by ~0.06):

| | spellF1 | macroF1 | possessions (truth 17) |
|---|---|---|---|
| **ours** | **0.710** | **0.823** | 14 |
| baseline_smooth125 | 0.424 | 0.581 | 16 |
| baseline_smooth50 (the published floor's setting) | 0.326 | 0.621 | 26 |

## 9. Limitations

- **Domain gap.** Neither training source uses the competition tracker; ball loss is simulated by
  hiding the ball at the competition tracker's overall rate, which only approximates where and
  when it really loses the ball.
- **Small training set.** 9 matches; the worst held-out match scores 0.50 against a mean of 0.57.
- **Short possessions.** A touch or a deflection lasting under 2 s is mostly missed: the features
  say who is near the ball, not who just touched it.
- **Short stoppages and short bursts.** Quick throw-ins and goal kicks whose tracking looks like
  live play can be labelled as possession, and a few seconds of play between two stoppages can be
  merged into one stoppage.
- **Long breaks.** During a drinks break, players kicking balls around can look like live play.
- **Metrica's labels are event-derived**, an approximation of the corrected-tracking definition.
- **No player identities** are used; everything is team-level, as the ATD provides.

## 10. Reproduction

See [README.md](README.md). In short: Python 3.13, `pip install -r solution/requirements.txt`,
get the Metrica sample data and the IDSSE files, set `MTAH_DATA_DIR`, run
`python solution/idsse_games.py` once (~30 s), then `python solution/run.py` (~35 min). Two runs
produce byte-identical files.
