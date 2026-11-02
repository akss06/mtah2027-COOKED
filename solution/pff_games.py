"""PFF FC's 2022 World Cup data (64 matches) as (track, labels) pairs at 25 fps, for training.

Other matches with their own event data, so learning from them is allowed (RULES.md). Request the
dataset from PFF FC, unpack it outside this repository and point PFF_DIR at the folder that holds
"Tracking Data", "Event Data" and "Metadata". Parsing is slow (~1 min per match), so

    python solution/pff_games.py          # converts every match once into PFF_CACHE_DIR

What is used, and what is deliberately not:
  * raw player positions, VISIBLE and ESTIMATED -- off-screen players are estimated by the tracker
    from context, which is closer to the competition's full-pitch view than leaving them out;
  * the raw ball only where it is VISIBLE. ESTIMATED balls are treated as missing;
  * never the "Smoothed" positions: PFF builds those from the event data, so they would leak the
    labels into the features.

Labels, by the same rules as sample_games.py: live frames belong to the team PFF marks as owning
the possession sequence (home_ball) at the latest on-the-ball event; a frame is dead from the end
of the event before a restart (kick-off, throw-in, goal kick, corner, free kick) until the restart.
"""

import bz2
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PFF_DIR = os.environ.get("PFF_DIR") or os.path.join(
    ROOT, "..", "external", "pff-wc2022", "FIFA World Cup 2022")
CACHE_DIR = os.environ.get("PFF_CACHE_DIR") or os.path.join(ROOT, "..", "external", "pff-cache")

FPS = 25
MAX_PLAYERS = 14  # per team per frame; more only happens with tracker duplicates
RESTARTS = {"K", "T", "G", "C", "F"}


def game_ids() -> list:
    names = os.listdir(os.path.join(PFF_DIR, "Tracking Data"))
    return sorted(n.split(".")[0] for n in names if n.endswith(".jsonl.bz2"))


def _positions(players: list) -> np.ndarray:
    out = np.full((MAX_PLAYERS, 2), np.nan)
    for i, p in enumerate((players or [])[:MAX_PLAYERS]):
        out[i] = p["x"], p["y"]
    return out


def _read(game: str) -> tuple:
    """Native-rate arrays plus every game event seen on the frames."""
    frame_nums, times, periods, home, away, ball, events = [], [], [], [], [], [], {}
    path = os.path.join(PFF_DIR, "Tracking Data", game + ".jsonl.bz2")
    with bz2.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            frame = json.loads(line)
            frame_nums.append(frame["frameNum"])
            times.append(frame["videoTimeMs"])
            periods.append(frame["period"])
            home.append(_positions(frame["homePlayers"]))
            away.append(_positions(frame["awayPlayers"]))
            balls = frame.get("balls") or []
            visible = balls and balls[0].get("visibility") == "VISIBLE"
            ball.append((balls[0]["x"], balls[0]["y"]) if visible else (np.nan, np.nan))
            event = frame.get("game_event")
            if event and frame["game_event_id"] not in events:
                events[frame["game_event_id"]] = event
    return (np.array(frame_nums), np.array(times), np.array(periods), np.array(home),
            np.array(away), np.array(ball), events)


def _labels(frame_nums: np.ndarray, periods: np.ndarray, events: dict,
            setpieces: dict) -> np.ndarray:
    """H / A / D / X per native frame. Event start/end are frameNum values; rows are file order."""
    out = np.full(len(frame_nums), "X", dtype="<U1")
    row = lambda frame_num: int(np.clip(np.searchsorted(frame_nums, frame_num), 0,  # noqa: E731
                                        len(frame_nums) - 1))
    ordered = sorted(events.items(), key=lambda kv: kv[1]["start_frame"])

    for period in np.unique(periods):
        in_period = [(eid, e) for eid, e in ordered if periods[row(e["start_frame"])] == period]
        owners = [(row(e["start_frame"]), "H" if e["home_ball"] else "A") for _, e in in_period
                  if e["game_event_type"] == "OTB" and e.get("home_ball") is not None]
        if not owners:
            continue
        starts = [row(e["start_frame"]) for _, e in in_period]
        ends = [row(e["end_frame"]) for _, e in in_period]
        rows = np.arange(starts[0], max(ends) + 1)
        rows = rows[periods[rows] == period]
        first = np.array([s for s, _ in owners])
        team = np.array([t for _, t in owners])
        out[rows] = team[np.maximum(np.searchsorted(first, rows, side="right") - 1, 0)]

        for i in range(1, len(in_period)):
            if setpieces.get(in_period[i][0]) in RESTARTS:
                out[ends[i - 1] + 1:starts[i]] = "D"
    return out


def convert(game: str) -> str:
    """Parse one match, resample to 25 fps by nearest video time, and cache it as .npz."""
    path = os.path.join(CACHE_DIR, game + ".npz")
    if os.path.isfile(path):
        return path
    with open(os.path.join(PFF_DIR, "Event Data", game + ".json"), encoding="utf-8") as handle:
        setpieces = {e["gameEventId"]: (e.get("gameEvents") or {}).get("setpieceType")
                     for e in json.load(handle)}
    with open(os.path.join(PFF_DIR, "Metadata", game + ".json"), encoding="utf-8") as handle:
        meta = json.load(handle)
    meta = meta[0] if isinstance(meta, list) else meta
    pitch = meta["stadium"]["pitches"][0]
    length, width = float(pitch["length"]), float(pitch["width"])

    frame_nums, times, periods, home, away, ball, events = _read(game)
    labels = _labels(frame_nums, periods, events, setpieces)

    # Resample each period onto a 40 ms grid. A single X frame between periods keeps them apart.
    rows = []
    for period in np.unique(periods):
        span = times[periods == period]
        grid = np.arange(span[0], span[-1], 1000.0 / FPS)
        nearest = np.clip(np.searchsorted(times, grid), 0, len(times) - 1)
        rows.append(nearest)
        rows.append(np.array([-1]))
    rows = np.concatenate(rows[:-1])
    gap = rows < 0
    rows = np.where(gap, 0, rows)

    # PFF's origin is the pitch centre; ours is a corner.
    shift = np.array([length / 2, width / 2])
    home, away, ball = home[rows] + shift, away[rows] + shift, ball[rows] + shift
    labels = np.where(gap, "X", labels[rows])
    home[gap] = away[gap] = np.nan
    ball[gap] = np.nan

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(path, home=home.astype(np.float32), away=away.astype(np.float32),
                        ball=ball.astype(np.float32), labels=labels,
                        pitch=np.array([length, width]))
    return path


def load(game: str) -> tuple:
    """(track, labels, pitch) for one converted match."""
    data = np.load(os.path.join(CACHE_DIR, game + ".npz"))
    track = {"home": data["home"].astype(float), "away": data["away"].astype(float),
             "ball": data["ball"].astype(float)}
    return track, data["labels"], tuple(data["pitch"])


if __name__ == "__main__":
    games = game_ids()
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else os.cpu_count()) as pool:
        for done, path in enumerate(pool.imap_unordered(convert, games), 1):
            print("{:>2}/{}  {}".format(done, len(games), os.path.basename(path)), flush=True)
