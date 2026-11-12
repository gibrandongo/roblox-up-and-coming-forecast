"""Forecast the Roblox Up-and-Coming top 10, place by place.

Reads chart history (history.ndjson plus task.json) from the input
directory and writes one forecast.json to the output directory.

Method, in brief: every game seen on the Up-and-Coming chart gets
recency-weighted rank and player-count trajectories (weighted least
squares against time), extrapolated part-way into the forecast window
with damping. The extrapolated rank, the game's rank volatility, its
presence on the Top Trending / Top Playing Now charts, how recently
it was last seen, and how close it is to the 28-day eligibility edge
(from details.created when present) combine into an expected rank
and an inclusion probability. Each game's probability mass is spread
over places 1..10 with a bell-shaped kernel around its expected
rank, so nearby places get meaningful mass and no single probability
is overconfident. Per-place sums stay below 1 so the unnamed share
covers games the forecast never names.

Standard library only, no network, no randomness, no wall clock:
the same input bytes always produce byte-identical output.
"""

import json
import math
import os
from datetime import datetime, timezone

# --- Model constants. The kernel width, per-place mass target and
# recency half-life were chosen on held-out backtests of the kit
# history (train on early reads, score on later reads): wider kernels
# and a lower named-mass target beat sharper, more confident variants
# on every split, because the chart drifts hard over a 72-hour
# window; the values below sit in the middle of the flat optimum
# rather than at its most extreme end, to avoid overfitting 34 reads.
HALF_LIFE_HOURS = 24.0      # recency weight halves every 24 hours
SLOPE_DAMP_HOURS = 18.0     # trend extrapolation horizon (half the window)
MOMENTUM_GAIN = 120.0       # rank-place shift per unit log-playing slope
MOMENTUM_CLAMP = 2.5        # max rank shift from playing momentum
LOGISTIC_CENTER = 10.5      # rank at which inclusion probability is 0.5
LOGISTIC_SCALE = 3.0        # steepness of the inclusion curve
PLACE_SUM_TARGET = 0.72     # per-place probability mass to name
P_CAP = 0.55                # no single probability above this
P_MIN = 0.002               # rows below this are dropped (mass stays unnamed)
SIGMA_BASE = 3.0            # kernel width around the expected rank
SIGMA_VOL = 0.30            # extra width per unit of rank volatility
SIGMA_MAX = 5.0             # widest kernel allowed
MAX_CANDIDATES = 60         # games carried into the probability table
AGE_FULL_DAYS = 23.0        # full weight while younger than this at mid-window
AGE_OUT_DAYS = 28.0         # eligibility edge of the Up-and-Coming chart
AGE_FLOOR = 0.10            # residual weight for games past the edge
STALE_DECAY = 0.85          # inclusion multiplier per snapshot since last seen
STALE_FLOOR = 0.10          # floor for long-absent games
CROSS_TREND_BONUS = 1.5     # expected-rank improvement for trending presence
CROSS_PLAYING_BONUS = 0.8   # expected-rank improvement for playing-now presence
CROSS_BONUS_MAX = 2.5       # cap on the combined cross-chart improvement
MAX_OUTPUT_BYTES = 1048576  # hard cap on forecast.json


def _parse_time(value):
    """Parse an ISO 8601 UTC timestamp to epoch seconds, or None."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _as_int(value):
    """Coerce a JSON value to int when it is honestly an integer."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text and (text.lstrip("-").isdigit()):
            try:
                return int(text)
            except ValueError:
                return None
    return None


def _as_float(value):
    """Coerce a JSON value to a finite float, or None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _resolve_paths():
    """Input directory and output file, honouring the documented env vars."""
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or os.environ.get(
        "MAKEFUN_INPUT"
    ) or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR")
    output_file = os.environ.get("MAKEFUN_OUTPUT")
    if output_dir:
        out_path = os.path.join(output_dir, "forecast.json")
    elif output_file:
        out_path = output_file
    else:
        out_path = os.path.join("/output", "forecast.json")
    return input_dir, out_path


def _load_task(input_dir):
    """Read places / window / output name from task.json, with defaults."""
    places = 10
    window_mid = None
    output_name = "forecast.json"
    try:
        with open(os.path.join(input_dir, "task.json"), "r", encoding="utf-8") as handle:
            task = json.load(handle)
    except (OSError, ValueError):
        return places, window_mid, output_name
    if not isinstance(task, dict):
        return places, window_mid, output_name
    count = _as_int(task.get("places"))
    if count is not None and 1 <= count <= 50:
        places = count
    window = task.get("window")
    if isinstance(window, dict):
        start = _parse_time(window.get("from"))
        end = _parse_time(window.get("until"))
        if start is not None and end is not None and end > start:
            window_mid = (start + end) / 2.0
    name = task.get("output")
    if isinstance(name, str) and name and "/" not in name and "\\" not in name:
        output_name = name
    return places, window_mid, output_name


def _iter_snapshot_lines(path):
    """Yield parsed snapshot dicts, skipping malformed lines quietly."""
    try:
        handle = open(path, "r", encoding="utf-8")
    except OSError:
        return
    with handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict):
                yield record


def _chart_entries(chart):
    """Normalise one chart list to (universe_id, rank, playing) triples.

    Duplicates keep the best (smallest) rank; entries without a usable
    universe id are dropped; a missing rank falls back to list order.
    """
    if not isinstance(chart, list):
        return []
    best = {}
    for position, entry in enumerate(chart):
        if not isinstance(entry, dict):
            continue
        uid = _as_int(entry.get("universe_id"))
        if uid is None:
            continue
        rank = _as_float(entry.get("rank"))
        if rank is None or rank < 1.0 or rank > 100000.0:
            # A missing or implausible rank falls back to list order,
            # so one corrupt row cannot drag a game's trajectory.
            rank = float(position + 1)
        playing = _as_float(entry.get("playing"))
        previous = best.get(uid)
        if previous is None or rank < previous[0]:
            best[uid] = (rank, playing)
    return [(uid, rank, playing) for uid, (rank, playing) in best.items()]


def _weighted_fit(weight_sum, weight_x, weight_xx, weight_y, weight_xy):
    """Weighted least squares of y on x; returns (intercept, slope)."""
    if weight_sum <= 0:
        return 0.0, 0.0
    denom = weight_sum * weight_xx - weight_x * weight_x
    if abs(denom) < 1e-9:
        return weight_y / weight_sum, 0.0
    slope = (weight_sum * weight_xy - weight_x * weight_y) / denom
    intercept = (weight_y - slope * weight_x) / weight_sum
    return intercept, slope


class _Game(object):
    """Recency-weighted aggregates for one game; O(1) memory per game."""

    __slots__ = (
        "seen", "weight", "rank_w", "rank_wx", "rank_wxx", "rank_wy",
        "rank_wxy", "rank_wy2", "play_w", "play_wx", "play_wxx",
        "play_wy", "play_wxy", "last_rank", "last_index", "last_playing",
        "trend_w", "trend_sum", "playing_w", "playing_sum", "created",
        "first_index",
    )

    def __init__(self, index):
        self.seen = 0
        self.weight = 0.0
        self.rank_w = 0.0
        self.rank_wx = 0.0
        self.rank_wxx = 0.0
        self.rank_wy = 0.0
        self.rank_wxy = 0.0
        self.rank_wy2 = 0.0
        self.play_w = 0.0
        self.play_wx = 0.0
        self.play_wxx = 0.0
        self.play_wy = 0.0
        self.play_wxy = 0.0
        self.last_rank = None
        self.last_index = -1
        self.last_playing = None
        self.trend_w = 0.0
        self.trend_sum = 0.0
        self.playing_w = 0.0
        self.playing_sum = 0.0
        self.created = None
        self.first_index = index


def _collect(input_dir):
    """Stream the history and build per-game aggregates.

    Two streaming passes over the file: the first learns the time span
    (for recency weights), the second accumulates. Memory stays
    proportional to the number of distinct games, never to the file.
    """
    history_path = os.path.join(input_dir, "history.ndjson")
    times = []
    for record in _iter_snapshot_lines(history_path):
        stamp = _parse_time(record.get("at"))
        times.append(stamp)
    total = len(times)
    if total == 0:
        return {}, None, 0, None
    # Fill missing timestamps by interpolation on snapshot order so that
    # every read still gets a sensible age.
    known = [(i, t) for i, t in enumerate(times) if t is not None]
    if known:
        last_time = known[-1][1]
        first_time = known[0][1]
        span = max(0.0, last_time - first_time)
        filled = []
        for i, stamp in enumerate(times):
            if stamp is not None:
                filled.append(stamp)
            elif total > 1:
                filled.append(first_time + span * i / (total - 1))
            else:
                filled.append(first_time)
    else:
        # No usable clock at all: assume six hours between reads.
        filled = [float(i * 6 * 3600) for i in range(total)]
        last_time = filled[-1]
    last_time = filled[-1]

    games = {}
    last_chart = []
    for index, record in enumerate(_iter_snapshot_lines(history_path)):
        if record.get("ok") is False:
            continue
        age_hours = max(0.0, (last_time - filled[index]) / 3600.0)
        weight = 0.5 ** (age_hours / HALF_LIFE_HOURS)
        entries = _chart_entries(record.get("up_and_coming"))
        if index == total - 1:
            last_chart = sorted(entries, key=lambda item: (item[1], item[0]))
        # Cross-chart presence, rank-weighted, for this read.
        trending = {}
        for uid, rank, _playing in _chart_entries(record.get("top_trending")):
            trending[uid] = max(0.0, 1.0 - rank / 50.0)
        playing_now = {}
        for uid, rank, _playing in _chart_entries(record.get("top_playing_now")):
            playing_now[uid] = max(0.0, 1.0 - rank / 50.0)
        # Details: keep each game's creation time when the read carries it.
        details = record.get("details")
        if isinstance(details, dict):
            for key, value in details.items():
                uid = _as_int(key)
                if uid is None or not isinstance(value, dict):
                    continue
                created = _parse_time(value.get("created"))
                if created is None:
                    continue
                game = games.get(uid)
                if game is None:
                    game = _Game(index)
                    games[uid] = game
                if game.created is None:
                    game.created = created
        for uid, rank, playing in entries:
            game = games.get(uid)
            if game is None:
                game = _Game(index)
                games[uid] = game
            game.seen += 1
            game.weight += weight
            game.rank_w += weight
            game.rank_wx += weight * age_hours
            game.rank_wxx += weight * age_hours * age_hours
            game.rank_wy += weight * rank
            game.rank_wxy += weight * age_hours * rank
            game.rank_wy2 += weight * rank * rank
            if playing is not None and playing >= 0:
                log_playing = math.log(playing + 1.0)
                game.play_w += weight
                game.play_wx += weight * age_hours
                game.play_wxx += weight * age_hours * age_hours
                game.play_wy += weight * log_playing
                game.play_wxy += weight * age_hours * log_playing
                game.last_playing = playing
            game.last_rank = rank
            game.last_index = index
            if uid in trending:
                game.trend_w += weight
                game.trend_sum += weight * trending[uid]
            if uid in playing_now:
                game.playing_w += weight
                game.playing_sum += weight * playing_now[uid]
    return games, last_time, total, last_chart


def _age_factor(created, window_mid):
    """Discount for games nearing the 28-day eligibility edge."""
    if created is None or window_mid is None:
        return 1.0
    age_days = (window_mid - created) / 86400.0
    if age_days <= AGE_FULL_DAYS:
        return 1.0
    if age_days >= AGE_OUT_DAYS:
        return AGE_FLOOR
    span = AGE_OUT_DAYS - AGE_FULL_DAYS
    frac = (age_days - AGE_FULL_DAYS) / span
    return 1.0 - frac * (1.0 - AGE_FLOOR)


def _score_games(games, total, window_mid, last_time):
    """Turn aggregates into (expected rank, inclusion q, kernel width)."""
    if window_mid is None and last_time is not None:
        window_mid = last_time + 36.0 * 3600.0
    scored = {}
    for uid, game in games.items():
        if game.seen == 0 or game.rank_w <= 0:
            continue
        mean_rank = game.rank_wy / game.rank_w
        intercept, slope = _weighted_fit(
            game.rank_w, game.rank_wx, game.rank_wxx,
            game.rank_wy, game.rank_wxy,
        )
        # Slope is rank change per hour of age (older reads have larger
        # age). A positive slope means the rank number was worse long
        # ago, i.e. the game has been climbing: extrapolate forward,
        # damped, then blend with the recency-weighted mean rank.
        extrapolated = intercept - SLOPE_DAMP_HOURS * slope
        expected = 0.7 * extrapolated + 0.3 * mean_rank
        # Player-count momentum: a rising log-playing slope (negative
        # slope against age) pulls the expected rank up a little.
        if game.play_w > 0:
            _p_intercept, play_slope = _weighted_fit(
                game.play_w, game.play_wx, game.play_wxx,
                game.play_wy, game.play_wxy,
            )
            growth = -play_slope
            shift = max(-MOMENTUM_CLAMP, min(MOMENTUM_CLAMP, growth * MOMENTUM_GAIN))
            expected -= shift
        # Cross-chart support improves the expected rank, capped.
        trend_presence = game.trend_sum / game.trend_w if game.trend_w > 0 else 0.0
        playing_presence = (
            game.playing_sum / game.playing_w if game.playing_w > 0 else 0.0
        )
        bonus = min(
            CROSS_BONUS_MAX,
            CROSS_TREND_BONUS * trend_presence
            + CROSS_PLAYING_BONUS * playing_presence,
        )
        expected -= bonus
        expected = max(1.0, min(60.0, expected))
        # Volatility from the weighted rank variance.
        variance = game.rank_wy2 / game.rank_w - mean_rank * mean_rank
        spread = math.sqrt(max(0.0, variance))
        sigma = min(SIGMA_MAX, SIGMA_BASE + SIGMA_VOL * spread)
        if game.seen < 3:
            sigma = min(SIGMA_MAX, sigma + 0.4)
        # Inclusion probability: logistic in expected rank, discounted
        # for age and for absence from recent reads.
        age = _age_factor(game.created, window_mid)
        stale_steps = max(0, (total - 1) - game.last_index)
        stale = max(STALE_FLOOR, STALE_DECAY ** stale_steps)
        logit = (LOGISTIC_CENTER - expected) / LOGISTIC_SCALE
        if logit > 20.0:
            base = 1.0
        elif logit < -20.0:
            base = 0.0
        else:
            base = 1.0 / (1.0 + math.exp(-logit))
        inclusion = base * age * stale
        inclusion = min(0.95, inclusion * (1.0 + 0.25 * trend_presence))
        scored[uid] = {
            "expected": expected,
            "inclusion": inclusion,
            "sigma": sigma,
            "age": age,
            "seen": game.seen,
            "weight": game.weight,
            "last_rank": game.last_rank,
        }
    return scored


def _build_forecast(scored, places):
    """Assemble the probability table and the top-10 guess."""
    ranked = sorted(
        scored.items(),
        key=lambda item: (-item[1]["inclusion"], item[1]["expected"], item[0]),
    )
    candidates = []
    for uid, info in ranked:
        if info["inclusion"] >= 0.02 and info["expected"] <= 30.0:
            candidates.append((uid, info))
        if len(candidates) >= MAX_CANDIDATES:
            break
    if len(candidates) < places:
        # Top up with the strongest remaining games by inclusion.
        chosen = {uid for uid, _info in candidates}
        for uid, info in ranked:
            if uid in chosen:
                continue
            candidates.append((uid, info))
            chosen.add(uid)
            if len(candidates) >= max(places, 20):
                break
    # Kernel mass per game, normalised over the places, scaled by the
    # game's inclusion probability so its total mass never exceeds 1.
    per_place = {place: {} for place in range(1, places + 1)}
    for uid, info in candidates:
        centre = info["expected"]
        sigma = info["sigma"]
        weights = []
        total_weight = 0.0
        for place in range(1, places + 1):
            z = (place - centre) / sigma
            w = math.exp(-0.5 * z * z)
            weights.append(w)
            total_weight += w
        if total_weight <= 0:
            continue
        for offset, place in enumerate(range(1, places + 1)):
            mass = info["inclusion"] * weights[offset] / total_weight
            if mass > 0:
                per_place[place][uid] = mass
    rows = []
    for place in range(1, places + 1):
        table = per_place[place]
        if not table:
            continue
        capped = {uid: min(P_CAP, mass) for uid, mass in table.items()}
        total_mass = sum(capped.values())
        if total_mass > PLACE_SUM_TARGET:
            scale = PLACE_SUM_TARGET / total_mass
            capped = {uid: mass * scale for uid, mass in capped.items()}
        for uid, mass in capped.items():
            rounded = round(mass, 6)
            if rounded >= P_MIN:
                rows.append({"place": place, "p": rounded, "universe_id": uid})
    rows.sort(key=lambda row: (row["place"], -row["p"], row["universe_id"]))
    # Top 10: honest expected order, with games ageing out pushed down.
    order = sorted(
        scored.items(),
        key=lambda item: (
            item[1]["expected"] + (1.0 - item[1]["age"]) * 8.0,
            -item[1]["inclusion"],
            item[0],
        ),
    )
    top = [uid for uid, _info in order[:places]]
    if len(top) < places:
        # Fill from the most-seen games, then placeholder ids.
        seen_order = sorted(
            scored.items(), key=lambda item: (-item[1]["weight"], item[0])
        )
        used = set(top)
        for uid, _info in seen_order:
            if uid not in used:
                top.append(uid)
                used.add(uid)
            if len(top) >= places:
                break
    placeholder = 1
    while len(top) < places:
        if placeholder not in set(top):
            top.append(placeholder)
        placeholder += 1
    return rows, top[:places]


def _fallback_forecast(places, scored):
    """A valid forecast when the model has nothing to work with."""
    if scored:
        rows, top = _build_forecast(scored, places)
        return rows, top
    return [], list(range(1, places + 1))


def _serialize(rows, top, places):
    """Compact deterministic JSON, pruned if it ever exceeds the cap."""
    payload = {"probabilities": rows, "top10": top}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    data = text.encode("utf-8")
    if len(data) <= MAX_OUTPUT_BYTES:
        return data
    # Belt and braces: keep fewer rows per place until under the cap.
    for keep in (40, 20, 10, 5):
        by_place = {}
        for row in rows:
            by_place.setdefault(row["place"], []).append(row)
        trimmed = []
        for place in sorted(by_place):
            kept = sorted(
                by_place[place], key=lambda row: (-row["p"], row["universe_id"])
            )[:keep]
            trimmed.extend(kept)
        trimmed.sort(key=lambda row: (row["place"], -row["p"], row["universe_id"]))
        text = json.dumps(
            {"probabilities": trimmed, "top10": top},
            sort_keys=True, separators=(",", ":"),
        )
        data = text.encode("utf-8")
        if len(data) <= MAX_OUTPUT_BYTES:
            return data
    text = json.dumps(
        {"probabilities": [], "top10": top}, sort_keys=True, separators=(",", ":")
    )
    return text.encode("utf-8")


def main():
    input_dir, out_path = _resolve_paths()
    try:
        places, window_mid, output_name = _load_task(input_dir)
        # The task's output name wins when the output came from a
        # directory env var or the default; an explicit output file
        # path (legacy MAKEFUN_OUTPUT) is used exactly as given.
        if os.environ.get("MAKEFUN_OUTPUT_DIR") or (
            not os.environ.get("MAKEFUN_OUTPUT")
        ):
            out_dir = os.path.dirname(out_path) or "."
            out_path = os.path.join(out_dir, output_name)
        games, last_time, total, _last_chart = _collect(input_dir)
        scored = _score_games(games, total, window_mid, last_time)
        if scored:
            rows, top = _build_forecast(scored, places)
        else:
            rows, top = _fallback_forecast(places, scored)
        data = _serialize(rows, top, places)
    except Exception:
        # Never emit a missing or malformed file: fall back to the
        # shape-valid empty forecast.
        try:
            places = 10
            data = _serialize([], list(range(1, places + 1)), places)
        except Exception:
            data = b'{"probabilities":[],"top10":[1,2,3,4,5,6,7,8,9,10]}'
    try:
        directory = os.path.dirname(out_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(out_path, "wb") as handle:
            handle.write(data)
    except OSError:
        # Nowhere left to write; the run cannot be salvaged.
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
