"""How much smoothing and jitter makes a training ball look like the competition tracker's?

    python solution/calibrate_ball.py   # ~2 min; prints the setting used in run.py

Compares three aggregates of the ball's acceleration while it is seen: touch-like peaks per
minute over 30 and over 60 m/s^2, and the median (noise floor). The competition side is measured
on its ATD input inside the two halves; no labels or video are involved. The training side is two
IDSSE matches, smoothed and degraded as in run.py, over a grid of windows and jitters.
"""

import numpy as np

import idsse_games
import run
from baseline import team_of_each_track
from features import FPS, parse_metrica_atd
from games import GAME, GAMES, half_video_blocks, require

WINDOWS = (1, 5, 9, 15, 25, 37, 51)  # frames
JITTERS_CM = (0, 0.5, 1, 1.5, 2, 3, 5)
CALIB_MATCHES = ("J03WMX", "J03WOH")


def ball_stats(ball: np.ndarray, blocks: list) -> dict:
    acc, seen, p30, p60 = [], 0, 0, 0
    for a, b in blocks:
        velocity = np.gradient(ball[a:b + 1], axis=0) * FPS
        x = np.linalg.norm(np.gradient(velocity, axis=0), axis=1) * FPS
        ok = ~np.isnan(x)
        seen += ok.sum()
        acc.append(x[ok])
        peak = np.r_[False, (x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]), False]
        p30 += np.sum(peak & (x > 30))
        p60 += np.sum(peak & (x > 60))
    minutes = seen / FPS / 60
    return {"p30": p30 / minutes, "p60": p60 / minutes, "floor": float(np.median(np.concatenate(acc)))}


def distance(stats: dict, target: dict) -> float:
    return sum((np.log(stats[k] + 0.05) - np.log(target[k] + 0.05)) ** 2 for k in target)


if __name__ == "__main__":
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    pitch = GAMES[GAME]["pitch"]
    track = parse_metrica_atd(raw_path, team_of_each_track(meta_path), pitch)
    target = ball_stats(track["ball"], [(a, b) for _, a, b in half_video_blocks(GAME)])
    print("competition ball: peaks>30 {p30:.1f}/min  >60 {p60:.1f}/min  floor {floor:.1f} m/s2".format(**target))

    data = []
    for i, m in enumerate(idsse_games.match_ids()):
        if m in CALIB_MATCHES:
            track, labels, _ = idsse_games.load(m)
            data.append((track, labels, run.labelled_blocks(labels), i))
    results = []
    for window in WINDOWS:
        for jitter in JITTERS_CM:
            stats = []
            for track, labels, blocks, i in data:
                smooth = {**track, "ball": run.smooth_ball(track["ball"], window, jitter,
                                                           np.random.default_rng(i))}
                degraded = run.degrade(smooth, labels, np.random.default_rng(run.SEED + i))
                stats.append(ball_stats(degraded["ball"], blocks))
            mean = {k: float(np.mean([s[k] for s in stats])) for k in target}
            results.append((distance(mean, target), window, jitter, mean))
            print("  window {:>2}  jitter {:>3} cm  peaks>30 {:6.1f}  >60 {:6.1f}  floor {:5.1f}".format(
                window, jitter, mean["p30"], mean["p60"], mean["floor"]), flush=True)
    best = min(results, key=lambda r: r[0])
    print("\nbest: window {} frames, jitter {} cm (run.py uses {}, {})".format(
        best[1], best[2], run.BALL_WINDOW, run.BALL_JITTER_CM))
