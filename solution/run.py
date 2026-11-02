"""Train on Metrica's public sample games, then label the competition match.

    python solution/run.py           # writes evaluation/results/hmm.csv

1. Features per frame (features.py): ball vs players, player speed, team shape, ball near a line.
2. A gradient-boosted classifier turns them into P(H), P(A), P(D) per frame. It is trained on the
   sample games only -- each with and without the ball artificially hidden, and with home and
   away swapped -- never on the competition match.
3. The Viterbi judge (hmm.py) turns those per-frame probabilities into whole possessions.

The one free knob, `temper`, is chosen leave-one-game-out on the sample games: train on game 1,
score game 2, and the reverse. The competition ground truth is never used for any choice.
"""

import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "starter"), os.path.join(ROOT, "evaluation")]
import features  # noqa: E402
import hmm  # noqa: E402
import possession_metrics as pm  # noqa: E402
import sample_games  # noqa: E402
from baseline import RESULTS_DIR, team_of_each_track, write_submission  # noqa: E402
from games import GAME, GAMES, half_video_blocks, require  # noqa: E402

TEMPERS = (1, 2, 4, 8, 16, 32, 64)
MASK_FRACTION = 0.4  # share of live sample frames whose ball is hidden in the augmented copy
TRAIN_STEP = 5  # neighbouring frames are near-duplicates; every 5th is plenty to train on
SEED = 0
SWAP_LABEL = {"H": "A", "A": "H", "D": "D"}


def labelled_blocks(labels: np.ndarray) -> list:
    """(first, last) of each contiguous non-X stretch -- the periods of a sample game."""
    graded = np.r_[False, labels != "X", False]
    edges = np.flatnonzero(graded[1:] != graded[:-1])
    return [(a, b - 1) for a, b in zip(edges[0::2], edges[1::2])]


def train(games: dict, rng: np.random.Generator) -> tuple:
    """(classifier, class prior) from the given sample games."""
    rows, targets = [], []
    for track, labels in games.values():
        blocks = labelled_blocks(labels)
        for version in (track, features.mask_ball(track, labels, MASK_FRACTION, rng)):
            frame = features.build(version, blocks, sample_games.PITCH).iloc[::TRAIN_STEP]
            y = labels[frame.index]
            rows += [frame, features.swap_teams(frame)]
            targets += [y, np.array([SWAP_LABEL[v] for v in y])]

    x, y = np.vstack([r.to_numpy() for r in rows]), np.concatenate(targets)
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                           random_state=SEED).fit(x, y)
    prior = np.array([np.mean(y == s) for s in hmm.STATES])
    return model, prior


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


def choose_temper(games: dict) -> float:
    """Leave-one-game-out spellF1, on the held-out game with its ball hidden like the training copy."""
    folds = []
    for fit, test in ((1, 2), (2, 1)):
        rng = np.random.default_rng(SEED)
        model, prior = train({fit: games[fit]}, rng)
        track, truth = games[test]
        blocks = labelled_blocks(truth)
        masked = features.mask_ball(track, truth, MASK_FRACTION, rng)
        frame = features.build(masked, blocks, sample_games.PITCH)
        folds.append((log_likelihoods(model, prior, frame), frame.index.to_numpy(), blocks,
                      truth, hmm.count_transitions(games[fit][1])))

    print("leave-one-game-out spellF1 on Metrica sample games (ball hidden on 40% of live play)")
    best, best_score = None, -1.0
    for temper in TEMPERS:
        scores = [spell_f1(truth, decode(ll, index, blocks, len(truth), trans, temper))
                  for ll, index, blocks, truth, trans in folds]
        mean = float(np.mean(scores))
        print("  temper {:>3}   test game2 {:.4f}   test game1 {:.4f}   mean {:.4f}".format(
            temper, *scores, mean))
        if mean > best_score:
            best, best_score = temper, mean
    print("  -> temper {}\n".format(best))
    return best


if __name__ == "__main__":
    games = {g: sample_games.load(g) for g in (1, 2)}
    temper = choose_temper(games)

    model, prior = train(games, np.random.default_rng(SEED))
    trans = hmm.count_transitions(np.concatenate([np.append(games[g][1], "X") for g in games]))

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
