"""Per-frame features from player and ball positions, the same for every match.

A Track is three arrays in metres: home (n, p, 2), away (n, q, 2), ball (n, 2), NaN where missing.
Everything is computed per block (a half or period), so nothing leaks across halftime.

What each group of features is for:
  * ball vs players   -- who is nearest the ball and by how much (the baseline's signal, softened)
  * player speed      -- stoppages are slow: everyone walks while a throw-in is taken
  * team shape        -- the team in possession spreads out and pushes up, the other compacts,
                         which still tells us something on the 41% of frames with no ball
  * ball near a line  -- where the ball was last seen, and how long ago
Short and long rolling means of each give the classifier context around the frame.
"""

import warnings

import numpy as np
import pandas as pd

FPS = 25
WINDOWS = (25, 125)  # 1 s and 5 s centred rolling means


def parse_metrica_atd(raw_path: str, team_of_track: np.ndarray, pitch: tuple) -> dict:
    """The competition ATD: every line is 'frame:x,y,s;...(46 slots):x,y,z'."""
    with open(raw_path) as handle:
        text = handle.read().replace(":", ",").replace(";", ",").replace("\n", ",")
    n_slots = len(team_of_track)
    values = np.fromstring(text, sep=",").reshape(-1, 1 + 3 * n_slots + 3)
    order = np.argsort(values[:, 0])
    values = values[order]
    scale = np.array(pitch, dtype=float)
    players = values[:, 1:1 + 3 * n_slots].reshape(-1, n_slots, 3)[:, :, :2] * scale
    ball = values[:, -3:-1] * scale
    return {"home": players[:, team_of_track == 0], "away": players[:, team_of_track == 1],
            "ball": ball}


def _centred_mean(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window, center=True, min_periods=1).mean().to_numpy()


def _block_features(home: np.ndarray, away: np.ndarray, ball: np.ndarray, length: float,
                    width: float) -> pd.DataFrame:
    # Empty frames (no players, no ball) are expected and come out as NaN; silence numpy about it.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        # Orient each team so it attacks +x in this block: the side its players stand on on
        # average is the side it defends.
        home_sign = 1.0 if np.nanmean(home[:, :, 0]) < length / 2 else -1.0

        def team(players: np.ndarray, sign: float, prefix: str) -> dict:
            x = sign * (players[:, :, 0] - length / 2)
            speed = np.linalg.norm(np.gradient(players, axis=0), axis=2) * FPS
            speed[speed > 12] = np.nan  # identity switches teleport a slot; not a real speed
            ball_dist = np.linalg.norm(players - ball[:, None, :], axis=2)
            return {
                prefix + "x_mean": np.nanmean(x, axis=1),
                prefix + "x_front": np.nanmax(x, axis=1),
                prefix + "x_spread": np.nanstd(x, axis=1),
                prefix + "y_spread": np.nanstd(players[:, :, 1], axis=1),
                prefix + "speed": np.nanmean(speed, axis=1),
                prefix + "ball_dist": np.nanmin(ball_dist, axis=1),
                prefix + "n_near_ball": np.sum(ball_dist < 5, axis=1).astype(float),
                prefix + "count": np.sum(~np.isnan(players[:, :, 0]), axis=1).astype(float),
            }

        cols = {**team(home, home_sign, "home_"), **team(away, -home_sign, "away_")}
        cols["home_x_velocity"] = np.gradient(cols["home_x_mean"]) * FPS
        cols["away_x_velocity"] = np.gradient(cols["away_x_mean"]) * FPS

        seen = ~np.isnan(ball[:, 0])
        cols["ball_seen"] = seen.astype(float)
        cols["ball_dist_diff"] = cols["home_ball_dist"] - cols["away_ball_dist"]
        last = pd.DataFrame(ball).ffill().to_numpy()
        cols["ball_to_touchline"] = np.minimum(last[:, 1], width - last[:, 1])
        cols["ball_to_goalline"] = np.minimum(last[:, 0], length - last[:, 0])
        cols["ball_x_home_attack"] = home_sign * (last[:, 0] - length / 2)
        since = np.arange(len(ball)) - pd.Series(np.where(seen, np.arange(len(ball)), np.nan)).ffill().to_numpy()
        cols["frames_since_ball"] = since

    frame = pd.DataFrame(cols)
    for window in WINDOWS:
        for name in ("home_speed", "away_speed", "ball_seen", "ball_dist_diff", "home_x_velocity",
                     "away_x_velocity", "home_x_spread", "away_x_spread"):
            frame["{}_m{}".format(name, window)] = _centred_mean(frame[name].to_numpy(), window)
    return frame


def build(track: dict, blocks: list, pitch: tuple) -> pd.DataFrame:
    """Feature rows for every frame inside the (first, last) blocks, indexed by frame."""
    parts = []
    for a, b in blocks:
        part = _block_features(track["home"][a:b + 1], track["away"][a:b + 1],
                               track["ball"][a:b + 1], *pitch)
        part.index = np.arange(a, b + 1)
        parts.append(part)
    return pd.concat(parts)


def mask_ball(track: dict, labels: np.ndarray, fraction: float, rng: np.random.Generator) -> dict:
    """Copy of `track` with the ball hidden on ~`fraction` of live frames, in 0.5-4 s gaps.

    The sample games' ball is tracked on ~87% of live play; the competition tracker loses it far
    more often. Training on both versions teaches the model to cope without the ball.
    """
    ball = track["ball"].copy()
    alive = np.flatnonzero(np.isin(labels, ["H", "A"]))
    target, hidden = int(fraction * len(alive)), 0
    while hidden < target:
        start = rng.choice(alive)
        gap = slice(start, start + rng.integers(FPS // 2, 4 * FPS))
        hidden += int(np.sum(~np.isnan(ball[gap, 0]) & np.isin(labels[gap], ["H", "A"])))
        ball[gap] = np.nan
    return {**track, "ball": ball}


def swap_teams(frame: pd.DataFrame) -> pd.DataFrame:
    """The same frames with home and away exchanged, so the model cannot learn a team bias."""
    renamed = {}
    for name in frame.columns:
        if name.startswith("home_"):
            renamed[name] = "away_" + name[5:]
        elif name.startswith("away_"):
            renamed[name] = "home_" + name[5:]
    swapped = frame.rename(columns=renamed)[frame.columns].copy()
    swapped["ball_dist_diff"] = -frame["ball_dist_diff"]
    swapped["ball_x_home_attack"] = -frame["ball_x_home_attack"]
    for window in WINDOWS:
        name = "ball_dist_diff_m{}".format(window)
        swapped[name] = -frame[name]
    return swapped
