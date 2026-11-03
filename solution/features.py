"""Per-frame features from player and ball positions, the same for every match.

A Track is three arrays in metres: home (n, p, 2), away (n, q, 2), ball (n, 2), NaN where missing.
Everything is computed per block (a half or period), so nothing leaks across halftime.

What each group of features is for:
  * ball vs players   -- who is nearest the ball and by how much (the baseline's signal, softened)
  * player speed      -- stoppages are slow: everyone walks while a throw-in is taken
  * team shape        -- the team in possession spreads out and pushes up, the other compacts,
                         which still tells us something on the 41% of frames with no ball
  * ball near a line  -- where the ball was last seen, and how long ago
  * across ball gaps  -- each team's distance to the ball when it was last / next seen, and in
                         the 2 s before vs after the frame
Short and long rolling means of each give the classifier context around the frame.
"""

import warnings

import numpy as np
import pandas as pd

FPS = 25
WINDOWS = (25, 125)  # 1 s and 5 s centred rolling means
LONG_WINDOW = 250  # 10 s, for speed and ball visibility only
PAST_FUTURE = 50  # 2 s one-sided means of each team's distance to the ball


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

        # The ball is missing on ~41% of competition frames. Who was nearest when it was last
        # seen, who is nearest when it reappears, and how each team's distance to it differs
        # just before vs just after the frame carry the possession across the gap -- and a
        # hand-over shows up as "close before, far after".
        for prefix in ("home_", "away_"):
            dist = pd.Series(np.where(seen, cols[prefix + "ball_dist"], np.nan))
            cols[prefix + "ball_dist_last"] = dist.ffill().to_numpy()
            cols[prefix + "ball_dist_next"] = dist.bfill().to_numpy()
            cols[prefix + "ball_dist_past"] = dist.rolling(PAST_FUTURE, min_periods=1).mean().to_numpy()
            cols[prefix + "ball_dist_future"] = (
                dist[::-1].rolling(PAST_FUTURE, min_periods=1).mean()[::-1].to_numpy())
        upcoming = pd.Series(np.where(seen, np.arange(len(ball)), np.nan)).bfill().to_numpy()
        cols["frames_until_ball"] = upcoming - np.arange(len(ball))

    frame = pd.DataFrame(cols)
    for window in WINDOWS:
        for name in ("home_speed", "away_speed", "ball_seen", "ball_dist_diff", "home_x_velocity",
                     "away_x_velocity", "home_x_spread", "away_x_spread"):
            frame["{}_m{}".format(name, window)] = _centred_mean(frame[name].to_numpy(), window)
    for name in ("home_speed", "away_speed", "ball_seen"):
        frame["{}_m{}".format(name, LONG_WINDOW)] = _centred_mean(frame[name].to_numpy(), LONG_WINDOW)
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


def mask_ball(track: dict, labels: np.ndarray, fraction: float, rng: np.random.Generator,
              states: tuple = ("H", "A")) -> dict:
    """Copy of `track` with the ball hidden on ~`fraction` of `states` frames, in 0.5-4 s gaps.

    The training matches track the ball far more often than the competition tracker does.
    Training on both versions teaches the model to cope without the ball.
    """
    ball = track["ball"].copy()
    chosen = np.flatnonzero(np.isin(labels, states))
    if not len(chosen):
        return {**track, "ball": ball}
    target = int(fraction * len(chosen))
    hidden = int(np.sum(np.isnan(ball[chosen, 0])))
    while hidden < target:
        start = rng.choice(chosen)
        gap = slice(start, start + rng.integers(FPS // 2, 4 * FPS))
        hidden += int(np.sum(~np.isnan(ball[gap, 0]) & np.isin(labels[gap], states)))
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
