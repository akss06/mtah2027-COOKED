"""Metrica's public sample games 1 and 2 as (track, labels) pairs, for training and checking.

These are other matches, with their own event data, so learning from them is allowed (RULES.md).
Download once, outside this repository:

    git clone --depth 1 https://github.com/metrica-sports/sample-data.git

and point METRICA_SAMPLE_DIR at its data/ folder (default: ../external/metrica-sample-data/data
next to this repository).

Labels are derived mechanically from the event log -- no human decided any frame:
  * alive frames belong to the team of the most recent possession-asserting event
    (pass, set piece, shot, recovery, ball lost, ball out). Challenges are skipped: both teams
    appear in them and a challenge alone does not change possession (CHALLENGE.md);
  * a frame is dead from the end of the last event before a set piece until the set piece starts;
  * frames outside the first-to-last event of each period are X (not graded).
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SAMPLE_DIR = os.environ.get("METRICA_SAMPLE_DIR") or os.path.join(
    ROOT, "..", "external", "metrica-sample-data", "data")
PITCH = (105.0, 68.0)

ASSERTING = {"PASS", "SET PIECE", "SHOT", "RECOVERY", "BALL LOST", "BALL OUT"}
H, A, D, X = "H", "A", "D", "X"


def _tracking_xy(path: str) -> tuple:
    """(players x, players y, ball x, ball y) in metres from one team's raw tracking CSV."""
    frame = pd.read_csv(path, skiprows=2)
    values = frame.iloc[:, 3:].to_numpy(dtype=float)
    players, ball = values[:, :-2], values[:, -2:]
    return (players[:, 0::2] * PITCH[0], players[:, 1::2] * PITCH[1],
            ball[:, 0] * PITCH[0], ball[:, 1] * PITCH[1])


def track(game: int) -> dict:
    """{'home': (n, p, 2), 'away': (n, q, 2), 'ball': (n, 2)} in metres, NaN where missing."""
    folder = os.path.join(SAMPLE_DIR, "Sample_Game_{}".format(game))
    stem = os.path.join(folder, "Sample_Game_{}_RawTrackingData_{{}}_Team.csv".format(game))
    hx, hy, bx, by = _tracking_xy(stem.format("Home"))
    ax, ay, _, _ = _tracking_xy(stem.format("Away"))
    return {"home": np.stack([hx, hy], axis=2), "away": np.stack([ax, ay], axis=2),
            "ball": np.stack([bx, by], axis=1)}


def labels(game: int, n_frames: int) -> np.ndarray:
    """Per-frame H / A / D / X from the event log. Frame f of the tracking is index f - 1."""
    path = os.path.join(SAMPLE_DIR, "Sample_Game_{0}".format(game),
                        "Sample_Game_{0}_RawEventsData.csv".format(game))
    events = pd.read_csv(path)
    events = events[events["Type"] != "CARD"].sort_values("Start Frame", kind="stable")
    start = events["Start Frame"].to_numpy() - 1
    end = np.maximum(events["End Frame"].to_numpy() - 1, start)
    team = np.where(events["Team"].to_numpy() == "Home", H, A)
    kind = events["Type"].to_numpy()

    out = np.full(n_frames, X, dtype="<U1")
    for period in events["Period"].unique():
        rows = np.flatnonzero(events["Period"].to_numpy() == period)
        first, last = start[rows[0]], end[rows].max()
        asserting = rows[np.isin(kind[rows], list(ASSERTING))]
        frames = np.arange(first, last + 1)
        owner = np.searchsorted(start[asserting], frames, side="right") - 1
        out[frames] = team[asserting][np.maximum(owner, 0)]

        for i in rows[1:]:
            if kind[i] == "SET PIECE":
                previous = rows[rows < i][-1] if (rows < i).any() else None
                if previous is not None:
                    out[end[previous] + 1:start[i]] = D
    return out


def load(game: int) -> tuple:
    """(track, labels) for one sample game."""
    positions = track(game)
    return positions, labels(game, len(positions["ball"]))
