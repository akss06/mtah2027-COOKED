"""IDSSE open data (7 Bundesliga / 2. Bundesliga 2022/23 matches) as (track, labels) at 25 fps.

Bassek, M. et al. (2025). An integrated dataset of spatiotemporal and event data in elite soccer.
Scientific Data 12, 195. https://doi.org/10.1038/s41597-025-04505-y -- licensed CC BY 4.0.
Data owner: Deutsche Fussball Liga (DFL); tracking by Sportec Solutions. Hugging Face mirror:
https://huggingface.co/datasets/pysport/idsse-data. Download it outside this repository and point
IDSSE_DIR at the folder, then

    python solution/idsse_games.py        # converts every match once into IDSSE_CACHE_DIR

Labels come straight from the official per-frame ball flags in the position data, the same kind
of flags the competition's ground truth is derived from: BallStatus 0 -> D; otherwise
BallPossession 1 -> H (home), 2 -> A (guest).
"""

import glob
import os
import re
import sys
import xml.etree.ElementTree as ET
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IDSSE_DIR = os.environ.get("IDSSE_DIR") or os.path.join(ROOT, "..", "external", "idsse")
CACHE_DIR = os.environ.get("IDSSE_CACHE_DIR") or os.path.join(ROOT, "..", "external", "idsse-cache")

MAX_PLAYERS = 20  # per team per half, substitutes included: each player is one slot
SECTIONS = ("firstHalf", "secondHalf")


def match_ids() -> list:
    names = glob.glob(os.path.join(IDSSE_DIR, "DFL_04_03_positions_raw_observed_*.xml"))
    return sorted(re.search(r"DFL-MAT-(\w+)\.xml$", n).group(1) for n in names)


def _file(kind: str, match: str) -> str:
    return glob.glob(os.path.join(IDSSE_DIR, "DFL_{}_*DFL-MAT-{}.xml".format(kind, match)))[0]


def convert(match: str) -> str:
    """Parse one match's positions and cache it as .npz."""
    path = os.path.join(CACHE_DIR, match + ".npz")
    if os.path.isfile(path):
        return path
    info = ET.parse(_file("02_01_matchinformation", match)).getroot().find(".//General")
    home_id, guest_id = info.get("HomeTeamId"), info.get("GuestTeamId")

    # section -> {"BALL": {N: (x, y, status, possession)}, team: [ {N: (x, y)} per player ]}
    data = {s: {"BALL": {}, home_id: [], guest_id: []} for s in SECTIONS}
    pitch = (105.0, 68.0)
    current = None
    # "start" events expose each FrameSet's team before its frames arrive.
    for event, element in ET.iterparse(_file("04_03_positions_raw_observed", match),
                                       events=("start", "end")):
        if event == "start" and element.tag == "PitchSize":
            pitch = (float(element.get("X")), float(element.get("Y")))
        elif event == "start" and element.tag == "FrameSet":
            section, team = element.get("GameSection"), element.get("TeamId")
            if team == "BALL":
                current = data[section]["BALL"]
            elif team in (home_id, guest_id):
                current = {}
                data[section][team].append(current)
            else:
                current = None
        elif event == "end" and element.tag == "Frame":
            if current is not None:
                n = int(element.get("N"))
                if "BallStatus" in element.attrib:
                    current[n] = (float(element.get("X")), float(element.get("Y")),
                                  int(element.get("BallStatus")), int(element.get("BallPossession")))
                else:
                    current[n] = (float(element.get("X")), float(element.get("Y")))
            element.clear()
        elif event == "end" and element.tag == "FrameSet":
            element.clear()

    shift = np.array(pitch) / 2  # DFL's origin is the pitch centre; ours is a corner
    parts = {"home": [], "away": [], "ball": [], "labels": []}
    for i, section in enumerate(SECTIONS):
        ball = data[section]["BALL"]
        frames = np.arange(min(ball), max(ball) + 1)
        index = {n: k for k, n in enumerate(frames)}
        for key, team in (("home", home_id), ("away", guest_id)):
            block = np.full((len(frames), MAX_PLAYERS, 2), np.nan)
            for slot, player in enumerate(data[section][team][:MAX_PLAYERS]):
                for n, (x, y) in player.items():
                    if n in index:
                        block[index[n], slot] = x, y
            parts[key].append(block + shift)
        xy = np.full((len(frames), 2), np.nan)
        labels = np.full(len(frames), "X", dtype="<U1")
        for n, (x, y, status, possession) in ball.items():
            xy[index[n]] = x, y
            labels[index[n]] = "D" if status == 0 else ("H" if possession == 1 else "A")
        parts["ball"].append(xy + shift)
        parts["labels"].append(labels)
        if i < len(SECTIONS) - 1:  # one X frame keeps the halves apart
            parts["home"].append(np.full((1, MAX_PLAYERS, 2), np.nan))
            parts["away"].append(np.full((1, MAX_PLAYERS, 2), np.nan))
            parts["ball"].append(np.full((1, 2), np.nan))
            parts["labels"].append(np.array(["X"]))

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(path, home=np.concatenate(parts["home"]).astype(np.float32),
                        away=np.concatenate(parts["away"]).astype(np.float32),
                        ball=np.concatenate(parts["ball"]).astype(np.float32),
                        labels=np.concatenate(parts["labels"]), pitch=np.array(pitch))
    return path


def load(match: str) -> tuple:
    """(track, labels, pitch) for one converted match."""
    data = np.load(os.path.join(CACHE_DIR, match + ".npz"))
    track = {"home": data["home"].astype(float), "away": data["away"].astype(float),
             "ball": data["ball"].astype(float)}
    return track, data["labels"], tuple(data["pitch"])


if __name__ == "__main__":
    matches = match_ids()
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else len(matches)) as pool:
        for done, path in enumerate(pool.imap_unordered(convert, matches), 1):
            print("{}/{}  {}".format(done, len(matches), os.path.basename(path)), flush=True)
