"""Train on IDSSE open data + Metrica's sample games, then label the competition match.

    python solution/idsse_games.py   # once: converts the IDSSE matches into a cache (~30 s)
    python solution/run.py           # writes evaluation/results/hmm.csv

1. Features per frame (features.py): ball vs players, player speed, team shape, ball context.
2. A gradient-boosted classifier turns them into P(H), P(A), P(D) per frame. It is trained on
   other matches only -- the 7 IDSSE matches and Metrica's 2 sample games. Their ball is smoothed
   like the competition tracker's; each is used as is and as a copy whose ball is hidden as often
   as the competition tracker loses it, and each with home and away swapped. Frames are weighted
   so every possession counts about equally, as it does in the metric.
   Two such classifiers are trained: one on a ball smoothed by a 37-frame moving average, one on a
   ball smoothed by a 47-frame Savitzky-Golay filter (the shape that best matches the competition
   ball's statistics). Their log-probabilities are averaged, 0.65 / 0.35.
3. The Viterbi judge (hmm.py) turns those per-frame probabilities into whole possessions.

`temper` is chosen leave-one-match-out over those 9 matches: each is scored by classifiers trained
on the other 8. The final classifiers are then trained on all 9. The pool weight (0.35) was fixed on
held-out external matches (COOKED.md section 7). The competition ground truth is never used
for any choice.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.signal import savgol_coeffs
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "starter"), os.path.join(ROOT, "evaluation")]
import features  # noqa: E402
import hmm  # noqa: E402
import idsse_games  # noqa: E402
import possession_metrics as pm  # noqa: E402
import sample_games  # noqa: E402
from baseline import RESULTS_DIR, team_of_each_track  # noqa: E402
from games import GAME, GAMES, half_video_blocks, require  # noqa: E402

TEMPERS = (1, 2, 3, 4, 8, 16)
TRAIN_STEP = 5  # neighbouring frames are near-duplicates; every 5th is plenty to train on
# The training matches track the ball almost always. The competition tracker loses it on 41% of
# graded frames, and on ~73% of the seconds before a restart (both aggregates of its input data,
# not of any labels). We aim to hide it on 25% of live and 70% of dead frames; gaps can spill across
# a change of state, so the real rates on the training matches are 28% and 76%, 47% overall.
ALIVE_MISSING = 0.25
DEAD_MISSING = 0.70
# The competition tracker's ball is smooth: 15-100x fewer sharp velocity changes per minute than
# the training balls. A centred mean over 37 frames plus 1.5 cm of jitter matches its peak rates
# and noise floor best (calibrate_ball.py, from aggregates of its input data only).
BALL_WINDOW = 37
BALL_JITTER_CM = 1.5
# The competition ball's acceleration autocorrelation and its profile around kicks (negative
# side-lobes ~25 frames out) match a Savitzky-Golay filter (order 2, ~47 frames) and rule out a
# moving average (COOKED.md section 4). A second classifier trained on that
# ball is pooled with the first: on held-out matches it gains +0.022 on an SG 51 ball and loses at
# most 0.014 on moving-average balls (COOKED.md section 7).
SG_WINDOW = 47
SG_ORDER = 2
POOL_WEIGHT = 0.35  # weight of the Savitzky-Golay classifier's log-probabilities
# spellF1 counts every possession once, however long. Unweighted, a 1 s possession gives the
# classifier 25 frames and a 60 s one 1,500, so short possessions barely register. Weighting each
# frame by 10 s / its run's length makes runs count about equally (clipped to 0.1-10).
WEIGHT_SECONDS = 10
SEED = 0
SWAP_LABEL = {"H": "A", "A": "H", "D": "D", "X": "X"}
MODEL_PATH = os.path.join(HERE, "model", "cooked_model.joblib")  # trained on external data only


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


def sg_ball(ball: np.ndarray, window: int, jitter_cm: float, rng: np.random.Generator) -> np.ndarray:
    """Savitzky-Golay smoothing (order SG_ORDER) over visible frames, then Gaussian jitter. Its
    weights go negative, so frames where the visible weight falls below half stay gaps."""
    k = savgol_coeffs(window, SG_ORDER)
    m = (~np.isnan(ball[:, 0])).astype(float)
    den = np.convolve(m, k, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.stack([np.convolve(np.nan_to_num(ball[:, d]) * m, k, mode="same") / den
                        for d in range(2)], 1)
    out[(den < 0.5) | (m == 0)] = np.nan
    return out + rng.normal(0, jitter_cm / 100, out.shape)


def run_weights(labels: np.ndarray) -> np.ndarray:
    """Per-frame training weight: WEIGHT_SECONDS / length of the run (of any label) it sits in."""
    out = np.ones(len(labels))
    breaks = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    for a, b in zip(np.r_[0, breaks], np.r_[breaks, len(labels)]):
        out[a:b] = np.clip(WEIGHT_SECONDS * features.FPS / (b - a), 0.1, 10)
    return out


def degrade(track: dict, labels: np.ndarray, rng: np.random.Generator) -> dict:
    """Hide the ball as often as the competition tracker loses it."""
    track = features.mask_ball(track, labels, ALIVE_MISSING, rng, ("H", "A"))
    return features.mask_ball(track, labels, DEAD_MISSING, rng, ("D",))


def prepare(matches: list, smooth=smooth_ball, window: int = BALL_WINDOW) -> list:
    """Per match: training rows (ball smoothed like the competition tracker's, with and without
    the competition's ball gaps, both team orders) and held-out features.

    Built once and reused by every leave-one-out fold. Each match gets its own seed, so the
    degraded copy does not depend on which other matches are loaded.
    """
    prepared = []
    for i, (name, track, labels, pitch) in enumerate(matches):
        blocks = labelled_blocks(labels)
        ball = smooth(track["ball"], window, BALL_JITTER_CM, np.random.default_rng(SEED + 2000 + i))
        track = {**track, "ball": ball}
        degraded = degrade(track, labels, np.random.default_rng(SEED + i))
        held_out = features.build(degraded, blocks, pitch)
        frame_weights = run_weights(labels)
        rows, targets, weights = [], [], []
        for frame in (features.build(track, blocks, pitch), held_out):
            frame = frame.iloc[::TRAIN_STEP]
            y = labels[frame.index]
            rows += [frame.to_numpy(), features.swap_teams(frame).to_numpy()]
            targets += [y, np.array([SWAP_LABEL[v] for v in y])]
            weights += [frame_weights[frame.index]] * 2
        prepared.append({"name": name, "labels": labels, "blocks": blocks, "held_out": held_out,
                         "x": np.vstack(rows), "y": np.concatenate(targets),
                         "w": np.concatenate(weights)})
        print("  prepared {}".format(name), flush=True)
    return prepared


def train(prepared: list) -> tuple:
    """(classifier, class prior, transitions) from prepared matches."""
    x = np.vstack([p["x"] for p in prepared])
    y = np.concatenate([p["y"] for p in prepared])
    w = np.concatenate([p["w"] for p in prepared])
    model = HistGradientBoostingClassifier(max_iter=600, max_leaf_nodes=63, learning_rate=0.05,
                                           random_state=SEED).fit(x, y, sample_weight=w)
    prior = np.array([np.average(y == s, weights=w) for s in hmm.STATES])
    trans = hmm.count_transitions(np.concatenate([np.append(p["labels"], "X") for p in prepared]))
    return model, prior, trans


def log_likelihoods(model, prior: np.ndarray, frame) -> np.ndarray:
    """(n, 3) scaled log-likelihoods in H, A, D order: log P(state | x) - log P(state)."""
    proba = model.predict_proba(frame.to_numpy())
    order = [list(model.classes_).index(s) for s in hmm.STATES]
    return np.log(proba[:, order] + 1e-6) - np.log(prior)


def pooled_log_likelihoods(box: tuple, sg: tuple, frame) -> np.ndarray:
    """POOL_WEIGHT x SG classifier + the rest x moving-average classifier; each is (model, prior)."""
    return POOL_WEIGHT * log_likelihoods(*sg, frame) + (1 - POOL_WEIGHT) * log_likelihoods(*box, frame)


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


def choose_temper(prepared: list, prepared_sg: list) -> float:
    """Leave-one-match-out: each match scored by both classifiers trained on all the others, on its
    moving-average ball (the held-out ball every earlier choice was made on)."""
    scores = {t: [] for t in TEMPERS}
    for i, held in enumerate(prepared):
        model, prior, trans = train(prepared[:i] + prepared[i + 1:])
        sg_model, sg_prior, _ = train(prepared_sg[:i] + prepared_sg[i + 1:])
        ll = pooled_log_likelihoods((model, prior), (sg_model, sg_prior), held["held_out"])
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


def write_csv(path: str, labels: np.ndarray) -> str:
    """A frame,label CSV. Always LF: the default newline depends on the OS, and the organisers must
    be able to regenerate this exact file on any machine."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(labels))
    return path


def label_competition(box: tuple, sg: tuple, trans: np.ndarray, temper: float) -> str:
    """Label the competition match with both trained (model, prior) pairs; returns the path."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    pitch = GAMES[GAME]["pitch"]
    track = features.parse_metrica_atd(raw_path, team_of_each_track(meta_path), pitch)
    n_frames = GAMES[GAME]["n_video_frames"]
    assert len(track["ball"]) == n_frames, "ATD is not one record per video frame"

    blocks = [(a, b) for _, a, b in half_video_blocks(GAME)]
    frame = features.build(track, blocks, pitch)
    labels = decode(pooled_log_likelihoods(box, sg, frame), frame.index.to_numpy(), blocks,
                    n_frames, trans, temper)
    return write_csv(os.path.join(RESULTS_DIR, "hmm.csv"), labels)


if __name__ == "__main__":
    print("preparing external matches", flush=True)
    matches = training_matches()
    prepared = prepare(matches)
    prepared_sg = prepare(matches, sg_ball, SG_WINDOW)
    for p in prepared_sg:
        del p["held_out"]  # both are scored on the moving-average ball; frees memory
    print("\nleave-one-match-out spellF1 (ball degraded like the competition tracker)")
    print("  {:<14} ".format("temper") + "  ".join("{:>6}".format(t) for t in TEMPERS))
    temper = choose_temper(prepared, prepared_sg)

    model, prior, trans = train(prepared)
    sg_model, sg_prior, _ = train(prepared_sg)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": model, "prior": prior, "sg_model": sg_model, "sg_prior": sg_prior,
                 "trans": trans, "temper": temper}, MODEL_PATH, compress=3)
    print("saved {}".format(MODEL_PATH))
    print("wrote {}".format(label_competition((model, prior), (sg_model, sg_prior), trans, temper)))
