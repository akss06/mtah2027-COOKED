# Team COOKED — Frame-level ball possession, BAR-ATH

*More than a Hack 2027, Phase 1. Technical document.*

## 1. Approach in one paragraph

For every frame we compute 78 features from the tracked player and ball positions, a
gradient-boosted classifier turns them into P(home), P(away), P(dead), and a three-state Viterbi
decoder turns those per-frame probabilities into whole possessions. The classifier is trained only
on **other matches**: the 7 matches of the IDSSE open dataset (official Bundesliga tracking with
per-frame ball-status and possession flags) and Metrica Sports' two public sample games. Their
ball is smoothed and hidden to look like the competition tracker's, and their frames are weighted
so that every possession counts about equally, as it does in the metric. Nothing is fitted, tuned or
selected on the competition match's ground truth or video.

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
| Tactical ATD (tracking) | **yes** | the only input to the pipeline; aggregates of its ball (missing rate, sharp velocity changes per minute, noise floor) set how the training ball is hidden and smoothed |
| Smart Tagging | automatic checks only | set-piece times were used in automatic diagnostics of where stoppages are missed, and to shape a simulation that was not adopted (section 7); not an input |
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

**The competition ball is smooth.** Per minute of visible ball, the training balls show 15–100×
more sharp velocity changes than the competition ball, the kind a touch produces. So every
training ball is first replaced by a centred moving average over 37 frames (~1.5 s) plus 1.5 cm
of Gaussian jitter. [calibrate_ball.py](calibrate_ball.py) chooses those two numbers on a grid.
They best match three aggregates of the competition ball inside the halves: peaks over 30 and over
60 m/s² per minute (2.2 and 0.4) and the median acceleration (5.7 m/s²). No labels and no video
are involved. The real tracker's filter is unknown; this only reproduces its statistics.

**The competition ball is often missing.** The tracker loses it on 41% of graded frames, and on
about 73% of the seconds before a tagged restart (again aggregates of the input data). Each
smoothed match is therefore used twice: as is, and with the ball hidden in random 0.5–4 s gaps on
25% of live and 70% of dead frames, which matches that overall rate. Every copy is also used with
home and away swapped, so the model cannot learn a team bias.

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
- **Ball touches:** a touch changes the ball's velocity. Two measures of that change: frame to
  frame (acceleration), and between the 0.5 s before and the 0.5 s after the frame ("turn"), which
  survives the tracker's smoothing better. Each is credited to the team whose player is nearest
  the ball, scaled down linearly to zero at 2 m, and summed over the past and the next 2 s and
  5 s. Also ball speed and each team's rate of approach to the ball.
- 1 s and 5 s centred rolling means of the speed, shape, velocity and ball-visibility features;
  10 s means of speed and ball visibility.

Using frames after the one being labelled is allowed: the task is post-match.

## 6. Model and decoding

- **Classifier:** scikit-learn `HistGradientBoostingClassifier` (600 iterations, 63 leaves per
  tree, learning rate 0.05, fixed seed), trained on every 5th frame — neighbours are
  near-duplicates. The size was chosen leave-one-match-out (section 7).
- **Weights:** each training frame is weighted by 10 s ÷ the length of the run (H, A or D) it
  belongs to, clipped to 0.1–10. spellF1 counts a 1 s possession like a 60 s one; unweighted, the
  classifier sees 60× fewer frames of it and learns to ignore it. The class prior is the weighted
  one.
- **Emissions:** the classifier's P(state | features) divided by the class prior (a scaled
  likelihood), then divided by a *temper* factor. Neighbouring frames are nearly copies of each
  other, so treating each as independent evidence overcounts it and makes the decoder flip on short
  runs.
- **Transitions:** counted per frame from the training labels.
- **Decoding:** Viterbi over H / A / D separately in each half; frames outside the halves are `X`.

**The one free setting, temper, is chosen leave-one-match-out** over the 9 training matches: each
match, with its ball smoothed and hidden as above, is scored by a model trained on the other 8.
spellF1 is computed with the organisers' own `possession_metrics.py`:

| temper | 1 | **2** | 4 | 8 | 16 |
|---|---|---|---|---|---|
| mean held-out spellF1 | 0.5628 | **0.5745** | 0.570 | 0.539 | 0.475 |
| worst held-out match | 0.476 | **0.486** | 0.484 | 0.467 | 0.381 |

Per held-out match at temper 2: IDSSE 0.486–0.615, Metrica 0.603 and 0.635. The final model is
then trained on all 9 matches. These scores are lower than earlier versions in section 7 because
the held-out ball is now smoothed like the competition's, a harder and more realistic test.

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
| **600 iterations, 63 leaves** (kept) | **0.545** | **0.657** | **0.643** | **0.569** | **0.502** | 2× |
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
| Ball-touch features trained on the training trackers' sharp ball | held-out mean 0.569 → 0.694 on those trackers, but with 5 cm of noise added to the held-out ball it fell to 0.514, below the model without it (0.565) | the competition ball shows 15–100× fewer sharp velocity changes per minute (it is smoothed), so the clue would not exist on BAR-ATH; replaced by touch features trained on a smoothed ball (adopted, below) |
| Filling the ATD's ball gaps with a pretrained ball detector (RT-DETR r18, Apache-2.0, published COCO / Objects365 weights, no training) run on the tactical video | found a ball on ~41% of in-play frames where the ATD has none; where both exist, median 1.2 m apart, but 23% of detections were over 10 m off (it loses fast and lofted balls) | too noisy where it matters most, and no training match has video, so the model could not learn how far to trust it; not used |

### Variants of the sharp-ball model (IDSSE + Metrica, leave-one-match-out)

Each changes one thing in the 600 / 63 model trained and tested on the sharp training ball (mean
0.569, worst 0.502). Retraining it with two other random seeds moved the mean by +0.006 / +0.007
and the worst match by −0.007 / −0.017, so changes of that size are noise. Our bar for adopting a
change: mean +0.015 or more, worst match not lower, gains on both IDSSE and Metrica. None passed.

| variant | mean | worst |
|---|---|---|
| sharp-ball model | 0.569 | 0.502 |
| 3× / 6× sample weight on short possessions and within 0.5 s of a change of team | 0.577 / 0.576 | 0.495 / 0.496 |
| average of 3 models with different seeds | 0.571 | 0.500 |
| train on every 2nd frame instead of every 5th | 0.570 | 0.499 |
| 30 s and 60 s averages of player speed, ball visibility and team spread (aimed at long stoppages) | 0.568 | 0.502 |
| players in each box, players off the pitch, keeper-to-ball distance (aimed at set pieces) | 0.566 | 0.488 |
| IDSSE only, without Metrica | 0.559 | 0.503 (Metrica games −0.051) |
| simulated Smart Tagging set pieces as features (below) | 0.571 | 0.491 |

Smart Tagging cannot be learned from directly, since no training match has it, so we simulated it
on the training matches. Every set-piece tag on BAR-ATH lasts 35 s and starts ~4.7 s before the
restart. That offset was measured against our own output, not the ground truth. There are 59
tags, against ~100 restarts in a training match. Fake tags were drawn from each training match's
true restarts at similar rates per restart type, with timing noise, 10% false tags and 10% wrong
teams. The model barely used them: with the held-out match's tags removed it scored 0.568, and
with deliberately worse tags 0.572. The tags mark restarts the model already finds; its errors are
short possessions in open play.

### Competition-like ball (adopted)

The variants above were all tested on the training trackers' sharp ball, which the competition
match does not have. A fair test gives both models the same held-out ball, smoothed like the
competition's (section 4) and hidden with the same seeds:

| model | held-out ball | mean | worst | IDSSE | Metrica |
|---|---|---|---|---|---|
| sharp-ball model | sharp | 0.569 | 0.502 | 0.545 | 0.650 |
| sharp-ball model | smoothed | 0.534 | 0.480 | 0.521 | 0.579 |
| trained on the smoothed ball | smoothed | 0.537 | 0.461 | 0.520 | 0.594 |
| **trained on the smoothed ball + touch features** (adopted) | smoothed | **0.559** | **0.479** | **0.541** | **0.623** |

(temper 2 for all four, for a like-for-like comparison.) So the sharp-ball model's 0.569 was
optimistic: on a competition-like ball it scores 0.534. The adopted model is better on 8 of 9
matches, and the ninth is 0.002 lower, inside seed noise.

The real tracker may blur or jitter its ball differently, so we stress-tested both models on
held-out balls smoother or noisier than calibrated:

| held-out ball | sharp-ball model | adopted | matches better |
|---|---|---|---|
| calibrated (37 frames, 1.5 cm) | 0.534 / 0.480 | 0.559 / 0.479 | 8 of 9 |
| smoother: 50 frames | 0.516 / 0.467 | 0.547 / 0.470 | 8 of 9 |
| smoother: 75 frames | 0.491 / 0.437 | 0.517 / 0.448 | 9 of 9 |
| calibrated + 5 cm noise | 0.534 / 0.480 | 0.549 / 0.478 | 8 of 9 |
| calibrated + 10 cm noise | 0.534 / 0.483 | 0.540 / 0.479 | 6 of 9 |
| calibrated + 20 cm noise | 0.530 / 0.480 | 0.530 / 0.449 | 4 of 9 |

(mean / worst, temper 2.) The adopted model wins wherever the ball is smoother and loses its edge
only with 20 cm of extra noise. The competition ball's aggregates point the other way: it has
fewer peaks than our calibrated ball (2.2 / 0.4 per minute against 2.8 / 0.7) and a lower noise
floor (5.7 against 8.4 m/s²).

### Short possessions

29% of true possessions in the training matches last under 2 s, and spellF1 counts each one like
any other. Pasting every true one into our held-out output would raise spellF1 from 0.56 to about
0.75, so we measured where that value lies (adopted model, competition-like ball):

| short possession, by what surrounds it | count (9 matches) | found under 1 s / 1–2 s | spellF1 if all fixed |
|---|---|---|---|
| one touch between two spells of the other team (deflection, block, interception lost at once) | 645 | 3% / 18% | +0.14 |
| last touch before the ball goes out | 335 | 13% / 27% | +0.05 |
| first touch after a restart | 129 | 25% / 59% | +0.01 |

Missing one costs more than one possession: its neighbours merge into one long spell, and that
usually fails to match too. The classifier does react: during 67–79% of these short possessions its
evidence for the touching team rises against the 2 s before. But the rise is small (median margin
−1.5 → −0.5 in log-likelihood) and similar rises happen everywhere, so a decoder that switches
more readily (lower temper) adds as many false possessions as it finds.

We then tried a second stage. It took peaks of that margin inside decoded possessions as
candidates (12,790; 6% real), described each by the margin's shape and by physical cues (nearest
distances, touch and turn credit, ball visibility), and inserted the confident ones. It ranked
candidates reasonably (AUC 0.80), but its 50 most confident were only ~30% real. With the
threshold chosen on the other 8 matches, it scored 0.5588 against 0.5594 without it. On the
smoothed ball the turn credit is near zero even at real touches: the touch is blurred into the
movement around it. We conclude that most short possessions are not recoverable from this tracking
data; we did not keep the second stage. Allowing it to insert only within 2 s or 5 s of a
possession's start or end, where a wrong insertion leaves the long remainder still matching, did not
change that (0.5587 and 0.5592, threshold chosen the same way).

The competition's *player* tracks, unlike its ball, are as sharp as the training ones (the 99th
percentile of a player's velocity change over ±0.5 s is 2.6 m/s against 2.3–2.4 on IDSSE). So we
added each team's sharpest player turn near the ball, with sums over the 2 s and 5 s either side:
0.5588 against 0.5594, not kept.

**What did work (adopted): weighting training frames by possession length** (section 6).
Unweighted, the classifier's loss is dominated by long possessions and long stoppages. Weighted:

| seed | without weights | with weights | worst match | IDSSE | Metrica | matches better |
|---|---|---|---|---|---|---|
| 0 | 0.5594 | **0.5745** | 0.482 → 0.486 | +0.019 | ±0.000 | 7 of 9 |
| 1 | 0.5624 | **0.5851** | 0.482 → 0.495 | +0.024 | +0.018 | 8 of 9 |

(each at the temper it chooses for itself: 1 and 2 without weights, 2 with them). The share of
true sub-2 s possessions found rises from 19% to 22%. The seed changes both the classifier and
which frames lose the ball, so each seed is compared with a baseline run at the same seed.

An automatic diagnostic using Smart Tagging set-piece times showed that some goal kicks and
throw-ins look exactly like live play in the tracking: the ball is visible and moving at ~4.5 m/s,
and players run at ~2 m/s. The tracking alone cannot separate those.

## 8. Result

On BAR-ATH the submission reports 379 possessions with a median of 5.6 s (15% under 2 s), and
calls 43% of graded-half frames dead, within the 41–52% of the IDSSE matches.

Pipeline check on the public 5 minutes (17 possessions — a sanity check, not a measurement; one
possession moves spellF1 by ~0.06):

| | spellF1 | macroF1 | possessions (truth 17) |
|---|---|---|---|
| **ours** | **0.589** | **0.773** | 19 |
| baseline_smooth125 | 0.424 | 0.581 | 16 |
| baseline_smooth50 (the published floor's setting) | 0.326 | 0.621 | 26 |

Earlier versions scored 0.710 (sharp training ball) and 0.581 (smoothed ball, unweighted) on these
5 minutes. Each change was adopted on the leave-one-match-out evidence above, which covers about
3,900 possessions on 9 matches; this window was not used for any choice and could not be, under
the rules.

## 9. Limitations

- **Domain gap.** Neither training source uses the competition tracker. Its ball is imitated by a
  moving average plus jitter matched on three statistics, and its ball loss by hiding the ball at
  its overall rate; the real filter and where it loses the ball are unknown.
- **Small training set.** 9 matches; the worst held-out match scores 0.49 against a mean of 0.57.
- **Short possessions.** Most touches and deflections lasting under 2 s are still missed (about
  22% are found), and fixing all of them would be worth about +0.19 (section 7). On a ball smoothed
  over ~1.5 s, a touch this short barely leaves a trace. This also assumes IDSSE's possession flag switches on every touch,
  as the competition ground truth does; we could not check it against video.
- **Short stoppages and short bursts.** Quick throw-ins and goal kicks whose tracking looks like
  live play can be labelled as possession, and a few seconds of play between two stoppages can be
  merged into one stoppage.
- **Long breaks.** During a drinks break, players kicking balls around can look like live play.
- **Metrica's labels are event-derived**, an approximation of the corrected-tracking definition.
- **No player identities** are used; everything is team-level, as the ATD provides.

## 10. Reproduction

See [README.md](README.md). Quickest: Python 3.13, `pip install -r solution/requirements.txt`,
set `MTAH_DATA_DIR`, run `python solution/predict.py` (~3 min). It labels the match with the
committed model (`solution/model/`, trained on external data only) and reproduces the submission
byte for byte. To retrain: get the Metrica sample data and the IDSSE files, run
`python solution/idsse_games.py` once (~30 s), then `python solution/run.py` (~45 min).
`python solution/calibrate_ball.py` (~2 min) re-derives the ball smoothing settings. Two runs
produce byte-identical files. A run from a fresh clone with a freshly built IDSSE cache and a
different thread count (14 instead of 16) was also identical, checked on the previous version.
