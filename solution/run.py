"""Train on IDSSE open data + Metrica's sample games, then label the competition match.

    python solution/idsse_games.py   # once: converts the IDSSE matches into a cache (~30 s)
    python solution/run.py           # writes evaluation/results/hmm.csv

1. Features per frame (features.py): ball vs players, player speed, team shape, ball context.
2. A gradient-boosted classifier turns them into P(H), P(A), P(D) per frame. It is trained on
   other matches only -- the 7 IDSSE matches and Metrica's 2 sample games. Their ball is smoothed
   like the competition tracker's; each is used as is and as a copy whose ball is hidden as often
   as the competition tracker loses it, and each with home and away swapped.
3. The Viterbi judge (hmm.py) turns those per-frame probabilities into whole possessions.

The one free knob, `temper`, is chosen leave-one-match-out over those 9 matches: each is scored by
a model trained on the other 8. The final model is then trained on all 9. The competition ground
truth is never used for any choice.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "starter"), os.path.join(ROOT, "evaluation")]
import features  # noqa: E402
import hmm  # noqa: E402
import idsse_games  # noqa: E402
import possession_metrics as pm  # noqa: E402
import sample_games  # noqa: E402
from baseline import RESULTS_DIR, team_of_each_track, write_submission  # noqa: E402
from games import GAME, GAMES, half_video_blocks, require  # noqa: E402

TEMPERS = (1, 2, 4, 8, 16)
TRAIN_STEP = 5  # neighbouring frames are near-duplicates; every 5th is plenty to train on
# The training matches track the ball almost always. The competition tracker loses it on 41% of
# graded frames, and on ~73% of the seconds before a restart (both aggregates of its input data,
# not of any labels). Hiding it on 25% of live and 70% of dead frames matches that overall rate.
ALIVE_MISSING = 0.25
DEAD_MISSING = 0.70
# The competition tracker's ball is smooth: 15-100x fewer sharp velocity changes per minute than
# the training balls. A centred mean over 37 frames plus 1.5 cm of jitter matches its peak rates
# and noise floor best (calibrate_ball.py, from aggregates of its input data only).
BALL_WINDOW = 37
BALL_JITTER_CM = 1.5
SEED = 0
SWAP_LABEL = {"H": "A", "A": "H", "D": "D", "X": "X"}


def labelled_blocks(labels: np.ndarray) -> list:
    """(first, last) of each contiguous non-X stretch -- the periods of a labelled match."""
    graded = np.r_[False, labels != "X", False]
    edges = np.flatnonzero(graded[1:] != graded[:-1])
    return [(a, b - 1) for a, b in zip(edges[0::2], edges[1::2])]


def training_matches() -> list:
    """[(name, track, labels, pitch)] for every external match used."""
    out = [("idsse_" + m, *idsse_games.load(m)) for m in idsse_games.match_ids()]
    for game in (1, 2):
        track, labels = sample_games.load(game)
        out.append(("metrica_%d" % game, track, labels, sample_games.PITCH))
    return out


def smooth_ball(ball: np.ndarray, window: int, jitter_cm: float, rng: np.random.Generator) -> np.ndarray:
    """Centred moving average over `window` frames, then Gaussian jitter. Gaps stay gaps."""
    out = pd.DataFrame(ball).rolling(window, center=True, min_periods=1).mean().to_numpy()
    out = out + rng.normal(0, jitter_cm / 100, out.shape)
    out[np.isnan(ball[:, 0])] = np.nan
    return out


def degrade(track: dict, labels: np.ndarray, rng: np.random.Generator) -> dict:
    """Hide the ball as often as the competition tracker loses it."""
    track = features.mask_ball(track, labels, ALIVE_MISSING, rng, ("H", "A"))
    return features.mask_ball(track, labels, DEAD_MISSING, rng, ("D",))


def prepare(matches: list) -> list:
    """Per match: training rows (ball smoothed like the competition tracker's, with and without
    the competition's ball gaps, both team orders) and held-out features.

    Built once and reused by every leave-one-out fold. Each match gets its own seed, so the
    degraded copy does not depend on which other matches are loaded.
    """
    prepared = []
    for i, (name, track, labels, pitch) in enumerate(matches):
        blocks = labelled_blocks(labels)
        ball = smooth_ball(track["ball"], BALL_WINDOW, BALL_JITTER_CM, np.random.default_rng(SEED + 2000 + i))
        track = {**track, "ball": ball}
        degraded = degrade(track, labels, np.random.default_rng(SEED + i))
        held_out = features.build(degraded, blocks, pitch)
        rows, targets = [], []
        for frame in (features.build(track, blocks, pitch), held_out):
            frame = frame.iloc[::TRAIN_STEP]
            y = labels[frame.index]
            rows += [frame.to_numpy(), features.swap_teams(frame).to_numpy()]
            targets += [y, np.array([SWAP_LABEL[v] for v in y])]
        prepared.append({"name": name, "labels": labels, "blocks": blocks, "held_out": held_out,
                         "x": np.vstack(rows), "y": np.concatenate(targets)})
        print("  prepared {}".format(name), flush=True)
    return prepared


def train(prepared: list) -> tuple:
    """(classifier, class prior, transitions) from prepared matches."""
    x = np.vstack([p["x"] for p in prepared])
    y = np.concatenate([p["y"] for p in prepared])
    model = HistGradientBoostingClassifier(max_iter=600, max_leaf_nodes=63, learning_rate=0.05,
                                           random_state=SEED).fit(x, y)
    prior = np.array([np.mean(y == s) for s in hmm.STATES])
    trans = hmm.count_transitions(np.concatenate([np.append(p["labels"], "X") for p in prepared]))
    return model, prior, trans


def log_likelihoods(model, prior: np.ndarray, frame) -> np.ndarray:
    """(n, 3) scaled log-likelihoods in H, A, D order: log P(state | x) - log P(state)."""
    proba = model.predict_proba(frame.to_numpy())
    order = [list(model.classes_).index(s) for s in hmm.STATES]
    return np.log(proba[:, order] + 1e-6) - np.log(prior)


def decode(log_like: np.ndarray, index: np.ndarray, blocks: list, n_frames: int,
           trans: np.ndarray, temper: float) -> np.ndarray:
    """H/A/D inside each block, X everywhere else."""
    out = np.full(n_frames, "X", dtype="<U1")
    for a, b in blocks:
        rows = (index >= a) & (index <= b)
        out[index[rows]] = hmm.STATES[hmm.viterbi(log_like[rows], trans, temper)]
    return out


def spell_f1(truth: np.ndarray, predicted: np.ndarray) -> float:
    return pm.evaluate(pm.label_codes(truth), pm.label_codes(predicted))["spell_f1_mean"]


def choose_temper(prepared: list) -> float:
    """Leave-one-match-out: each match scored by a model trained on all the others."""
    scores = {t: [] for t in TEMPERS}
    for i, held in enumerate(prepared):
        model, prior, trans = train(prepared[:i] + prepared[i + 1:])
        ll = log_likelihoods(model, prior, held["held_out"])
        index = held["held_out"].index.to_numpy()
        row = []
        for temper in TEMPERS:
            score = spell_f1(held["labels"], decode(ll, index, held["blocks"], len(held["labels"]),
                                                    trans, temper))
            scores[temper].append(score)
            row.append(score)
        print("  {:<14} ".format(held["name"]) + "  ".join("{:.4f}".format(s) for s in row),
              flush=True)

    print("  {:<14} ".format("mean") + "  ".join("{:.4f}".format(np.mean(scores[t]))
                                                for t in TEMPERS))
    print("  {:<14} ".format("worst") + "  ".join("{:.4f}".format(np.min(scores[t]))
                                                 for t in TEMPERS))
    best = max(TEMPERS, key=lambda t: np.mean(scores[t]))
    print("  -> temper {}\n".format(best), flush=True)
    return best


if __name__ == "__main__":
    print("preparing external matches", flush=True)
    prepared = prepare(training_matches())
    print("\nleave-one-match-out spellF1 (ball degraded like the competition tracker)")
    print("  {:<14} ".format("temper") + "  ".join("{:>6}".format(t) for t in TEMPERS))
    temper = choose_temper(prepared)

    model, prior, trans = train(prepared)
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    pitch = GAMES[GAME]["pitch"]
    track = features.parse_metrica_atd(raw_path, team_of_each_track(meta_path), pitch)
    n_frames = GAMES[GAME]["n_video_frames"]
    assert len(track["ball"]) == n_frames, "ATD is not one record per video frame"

    blocks = [(a, b) for _, a, b in half_video_blocks(GAME)]
    frame = features.build(track, blocks, pitch)
    labels = decode(log_likelihoods(model, prior, frame), frame.index.to_numpy(), blocks,
                    n_frames, trans, temper)
    path = write_submission(os.path.join(RESULTS_DIR, "hmm.csv"), labels)
    print("wrote {}".format(path))
