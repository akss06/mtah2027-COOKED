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
  * ball touches      -- sudden changes of the ball's velocity, credited to the team nearest the
                         ball, summed over the past and next 2 s and 5 s
  * kick direction    -- the ball's path 0.8-1.6 s before and after each frame, outside the blur
                         of a touch at that frame: where the two meet, how hard and which way the
                         ball was pushed, and whether each team's nearest player fits that push
  * de-blurred ball   -- two gently sharpened ball positions that are safe for any blur shape
                         seen: each team's distance to them and the ball's turn credited to it
Short and long rolling means of each give the classifier context around the frame.
"""

import warnings

import numpy as np
import pandas as pd

FPS = 25
WINDOWS = (25, 125)  # 1 s and 5 s centred rolling means
LONG_WINDOW = 250  # 10 s, for speed and ball visibility only
PAST_FUTURE = 50  # 2 s one-sided means of each team's distance to the ball
# A nearest player further than this gets no credit for a touch. On IDSSE the gaining team's
# nearest player is a median 0.62 m (p90 1.13 m) from the ball when the official flag switches.
REACH = 2.0
TURN = 12  # frames: the ball's velocity over the past vs the next ~0.5 s
# The competition ball is blurred over ~1.5 s (0.75 s either side), so straight-line fits to the
# ball 0.8-1.6 s before and after a frame are untouched by a touch at that frame.
APEX_NEAR, APEX_FAR = 20, 40  # frames
APEX_MIN_SEEN = 6  # visible ball frames needed in each window
APEX_REACH = 2.5  # m
# De-blur z = y - c ((y(t-D) + y(t+D)) / 2 - y) sharpens a kick iff 0 < c < 2 c_kick(kernel).
# Over boxcar 37-75, Gaussian, two-sided exponential and Savitzky-Golay 51 / 75 / 69 blurs,
# c_kick(19) is 0.31-3.9 and c_kick(11) 0.78-11.6, so these two never overshoot.
GENTLE = ((11, 0.90, "pn"), (19, 0.45, "pw"))  # (D frames, c, column tag)
GENTLE_REACH = 3.0  # m: a partly de-blurred corner still gets credit
GENTLE_MIN = 12  # frames: rolling-min window on each side


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


def _one_sided_sum(values: np.ndarray, window: int, future: bool) -> np.ndarray:
    s = pd.Series(values[::-1] if future else values).rolling(window, min_periods=1).sum()
    return s.to_numpy()[::-1] if future else s.to_numpy()


def _touch_features(frame: pd.DataFrame, ball: np.ndarray) -> None:
    """A touch shows up as a change in the ball's velocity: frame-to-frame (acceleration) and over
    ~0.5 s either side (turn), which survives the competition tracker's smoothing better. Each is
    credited to the team nearest the ball, scaled down with that player's distance; one-sided
    sums tell each frame which team touched the ball recently and which touches it next."""
    with np.errstate(invalid="ignore"):
        velocity = np.gradient(ball, axis=0) * FPS  # NaN next to ball gaps
        accel = np.nan_to_num(np.linalg.norm(np.gradient(velocity, axis=0), axis=1) * FPS)
        past = np.full_like(ball, np.nan)
        nxt = np.full_like(ball, np.nan)
        past[TURN:] = (ball[TURN:] - ball[:-TURN]) / (TURN / FPS)
        nxt[:-TURN] = (ball[TURN:] - ball[:-TURN]) / (TURN / FPS)
        turn = np.nan_to_num(np.linalg.norm(nxt - past, axis=1))
        frame["ball_speed"] = np.linalg.norm(velocity, axis=1)
        frame["ball_accel"] = accel
        frame["ball_accel_m25"] = _centred_mean(accel, 25)
        for own, other in (("home_", "away_"), ("away_", "home_")):
            _credit_to_team(frame, own, other, accel, "touch")
            # Negative while the ball travels towards this team's nearest player.
            frame[own + "ball_dist_rate"] = np.gradient(frame[own + "ball_dist"].to_numpy()) * FPS
        frame["ball_turn"] = turn
        for own, other in (("home_", "away_"), ("away_", "home_")):
            _credit_to_team(frame, own, other, turn, "turn")


def _credit_to_team(frame: pd.DataFrame, own: str, other: str, signal: np.ndarray, name: str) -> None:
    dist = frame[own + "ball_dist"].to_numpy()
    nearest = dist < frame[other + "ball_dist"].to_numpy()
    credit = np.nan_to_num(np.where(nearest, signal * np.clip(1 - dist / REACH, 0, 1), 0))
    frame[own + name] = credit
    for window in (PAST_FUTURE, WINDOWS[1]):
        frame["{}{}_past{}".format(own, name, window)] = _one_sided_sum(credit, window, False)
        frame["{}{}_future{}".format(own, name, window)] = _one_sided_sum(credit, window, True)


def _window_line(ball: np.ndarray, start: int, stop: int) -> tuple:
    """Least-squares line through the visible ball in frames [t+start, t+stop) for every t,
    evaluated at t: (position, velocity in m/s), NaN where too few samples."""
    n = len(ball)
    seen = ~np.isnan(ball[:, 0])
    lo = np.clip(np.arange(n) + start, 0, n)
    hi = np.clip(np.arange(n) + stop, 0, n)

    def wsum(v: np.ndarray) -> np.ndarray:
        c = np.r_[0.0, np.cumsum(v)]
        return c[hi] - c[lo]

    tt = np.arange(n, dtype=float)
    m = seen.astype(float)
    s0, s1, s2 = wsum(m), wsum(m * tt), wsum(m * tt * tt)
    tbar = s1 / s0
    var = s2 / s0 - tbar ** 2
    pos, vel = np.full((n, 2), np.nan), np.full((n, 2), np.nan)
    for d in range(2):
        x = np.where(seen, ball[:, d], 0.0)
        sx, stx = wsum(x), wsum(x * tt)
        slope = (stx / s0 - tbar * sx / s0) / var
        pos[:, d] = sx / s0 + slope * (tt - tbar)
        vel[:, d] = slope * FPS
    bad = (s0 < APEX_MIN_SEEN) | (var < 1)
    pos[bad], vel[bad] = np.nan, np.nan
    return pos, vel


def _apex_features(frame: pd.DataFrame, home: np.ndarray, away: np.ndarray, ball: np.ndarray) -> None:
    """Extrapolate the lines before and after each frame to the frame: their midpoint (apex) and
    gap, and the push v_after - v_before. Per team: its nearest player's distance to the apex,
    the push credited to it within APEX_REACH (summed over the past and next 2 s and 5 s), that
    player's speed along the push, and whether it stands on the side the push came from."""
    p_in, v_in = _window_line(ball, -APEX_FAR, -APEX_NEAR)
    p_out, v_out = _window_line(ball, APEX_NEAR + 1, APEX_FAR + 1)
    apex = (p_in + p_out) / 2
    dv = v_out - v_in
    dvn = np.linalg.norm(dv, axis=1)
    unit = dv / (dvn[:, None] + 1e-6)
    frame["apex_gap"] = np.linalg.norm(p_in - p_out, axis=1)
    frame["apex_dv"] = dvn
    rows = np.arange(len(ball))
    for prefix, players in (("home_", home), ("away_", away)):
        vel = np.gradient(players, axis=0) * FPS
        rel = apex[:, None, :] - players
        dist = np.linalg.norm(rel, axis=2)
        nearest = np.nanargmin(np.where(np.isnan(dist), np.inf, dist), axis=1)
        d, v, r = dist[rows, nearest], vel[rows, nearest], rel[rows, nearest]
        credit = np.nan_to_num(dvn * np.clip(1 - d / APEX_REACH, 0, 1))
        frame[prefix + "apex_dist"] = d
        frame[prefix + "apex_touch"] = credit
        frame[prefix + "apex_align"] = np.sum(v * unit, axis=1)
        frame[prefix + "apex_side"] = np.sum(r * unit, axis=1) / (np.linalg.norm(r, axis=1) + 1e-6)
        for window in (PAST_FUTURE, WINDOWS[1]):
            frame["{}apex_touch_past{}".format(prefix, window)] = _one_sided_sum(credit, window, False)
            frame["{}apex_touch_future{}".format(prefix, window)] = _one_sided_sum(credit, window, True)


def _gentle_features(frame: pd.DataFrame, home: np.ndarray, away: np.ndarray, ball: np.ndarray) -> None:
    """Two kernel-safe de-blurred ball positions (see GENTLE). Per team: nearest player's distance
    to each, held across ball gaps and as a rolling minimum either side, and the turn credited to
    it within GENTLE_REACH, summed over the past and next 2 s and 5 s."""
    n = len(ball)
    turn = frame["ball_turn"].to_numpy()
    moving = turn > 0
    # Per-block 90th percentile: a wider blur shrinks every turn by the same factor.
    p90 = np.percentile(turn[moving], 90) if moving.sum() > 100 else 0.0
    turn = turn / p90 if p90 > 0 else turn
    frame["ball_turn_norm"] = turn
    for d, c, tag in GENTLE:
        z = np.full_like(ball, np.nan)
        if n > 2 * d:
            mid = ball[d:-d]
            z[d:-d] = mid - c * ((ball[:-2 * d] + ball[2 * d:]) / 2 - mid)
        seen = ~np.isnan(z[:, 0])
        frame[tag + "_seen"] = seen.astype(float)
        for prefix, players in (("home_", home), ("away_", away)):
            dist = np.nanmin(np.linalg.norm(players - z[:, None, :], axis=2), axis=1)
            s = pd.Series(np.where(seen, dist, np.nan))
            frame[prefix + tag + "_dist"] = dist
            frame[prefix + tag + "_dist_last"] = s.ffill().to_numpy()
            frame[prefix + tag + "_dist_next"] = s.bfill().to_numpy()
            frame[prefix + tag + "_min_past"] = s.rolling(GENTLE_MIN + 1, min_periods=1).min().to_numpy()
            frame[prefix + tag + "_min_future"] = s[::-1].rolling(GENTLE_MIN + 1, min_periods=1).min().to_numpy()[::-1]
            credit = np.nan_to_num(turn * np.clip(1 - dist / GENTLE_REACH, 0, 1))
            frame[prefix + tag + "_turn"] = credit
            for window in (PAST_FUTURE, WINDOWS[1]):
                frame["{}{}_turn_past{}".format(prefix, tag, window)] = _one_sided_sum(credit, window, False)
                frame["{}{}_turn_future{}".format(prefix, tag, window)] = _one_sided_sum(credit, window, True)


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
    _touch_features(frame, ball)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _apex_features(frame, home, away, ball)
        _gentle_features(frame, home, away, ball)
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
