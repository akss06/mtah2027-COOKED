"""Train on PFF FC's 2022 World Cup matches, then label the competition match.

    python solution/pff_games.py     # once: converts the PFF matches into a cache (~15 min)
    python solution/run.py           # writes evaluation/results/hmm.csv

1. Features per frame (features.py): ball vs players, player speed, team shape, ball near a line.
2. A gradient-boosted classifier turns them into P(H), P(A), P(D) per frame. It is trained on
   PFF's 64 World Cup matches plus Metrica's two public sample games, each also with home and
   away swapped -- never on the competition match.
3. The Viterbi judge (hmm.py) turns those per-frame probabilities into whole possessions.

The one free knob, `temper`, is chosen on PFF matches held out from training; the final model is
then retrained on every match. The competition ground truth is never used for any choice.
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
import pff_games  # noqa: E402
import possession_metrics as pm  # noqa: E402
import sample_games  # noqa: E402
from baseline import RESULTS_DIR, team_of_each_track, write_submission  # noqa: E402
from games import GAME, GAMES, half_video_blocks, require  # noqa: E402

TEMPERS = (1, 2, 4, 8, 16, 32, 64)
HOLDOUT_EVERY = 6  # every 6th PFF match (by id) is held out to choose temper
TRAIN_STEP = 10  # neighbouring frames are near-duplicates; every 10th is plenty to train on
METRICA_STEP = 5  # two Metrica games against 64 PFF ones: sample them more densely
MASK_FRACTION = 0.4  # share of Metrica's live play whose ball is hidden in the masked copy
SEED = 0
SWAP_LABEL = {"H": "A", "A": "H", "D": "D", "X": "X"}


def labelled_blocks(labels: np.ndarray) -> list:
    """(first, last) of each contiguous non-X stretch -- the periods of a labelled match."""
    graded = np.r_[False, labels != "X", False]
    edges = np.flatnonzero(graded[1:] != graded[:-1])
    return [(a, b - 1) for a, b in zip(edges[0::2], edges[1::2])]


def train(pff_ids: list) -> tuple:
    """(classifier, class prior, transitions) from the given PFF matches plus both Metrica games.

    Metrica's sample games are used as recorded and with the ball hidden on 40% of live play, as
    their ball is tracked far more often than the competition tracker's. PFF's broadcast tracking
    already loses the ball about as often, so it is used as is.
    """
    rng = np.random.default_rng(SEED)
    matches = []
    for game in pff_ids:
        track, labels, pitch = pff_games.load(game)
        matches.append((track, labels, pitch, TRAIN_STEP))
    for game in (1, 2):
        track, labels = sample_games.load(game)
        masked = features.mask_ball(track, labels, MASK_FRACTION, rng)
        matches += [(track, labels, sample_games.PITCH, METRICA_STEP),
                    (masked, labels, sample_games.PITCH, METRICA_STEP)]

    rows, targets, transitions = [], [], []
    for track, labels, pitch, step in matches:
        frame = features.build(track, labelled_blocks(labels), pitch).iloc[::step]
        y = labels[frame.index]
        rows += [frame.to_numpy(), features.swap_teams(frame).to_numpy()]
        targets += [y, np.array([SWAP_LABEL[v] for v in y])]
        transitions.append(np.append(labels, "X"))

    x, y = np.vstack(rows), np.concatenate(targets)
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                           random_state=SEED).fit(x, y)
    prior = np.array([np.mean(y == s) for s in hmm.STATES])
    return model, prior, hmm.count_transitions(np.concatenate(transitions))


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


def evidence(model, prior: np.ndarray, track: dict, labels: np.ndarray, pitch: tuple) -> tuple:
    """Everything decode needs for one labelled match, computed once and reused per temper."""
    blocks = labelled_blocks(labels)
    frame = features.build(track, blocks, pitch)
    return log_likelihoods(model, prior, frame), frame.index.to_numpy(), blocks, labels


def score(cases: list, trans: np.ndarray, temper: float) -> list:
    return [spell_f1(truth, decode(ll, index, blocks, len(truth), trans, temper))
            for ll, index, blocks, truth in cases]


if __name__ == "__main__":
    games = pff_games.game_ids()
    holdout = games[::HOLDOUT_EVERY]
    training = [g for g in games if g not in holdout]
    print("training on {} PFF matches + 2 Metrica, {} PFF held out".format(len(training), len(holdout)), flush=True)
    model, prior, trans = train(training)

    held = [evidence(model, prior, *pff_games.load(g)) for g in holdout]
    print("\nspellF1 on held-out PFF matches")
    best, best_score = None, -1.0
    for temper in TEMPERS:
        mean = float(np.mean(score(held, trans, temper)))
        print("  temper {:>3}   mean {:.4f}".format(temper, mean))
        if mean > best_score:
            best, best_score = temper, mean
    temper = best
    print("  -> temper {}\n".format(temper), flush=True)

    model, prior, trans = train(games)
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
