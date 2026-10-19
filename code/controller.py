"""Shared controllers: used by BOTH the offline simulator (P3) and the live pipeline (P2).
Each controller exposes step(...) -> resolution to infer at, or None (skip, carry forward)."""
import math

TIERS = (320, 480, 640)
INTERVAL = {"low": 5, "mid": 2, "high": 1}
ORDER = {"low": 0, "mid": 1, "high": 2}


class TemporalTier:
    """Motion score -> sampling tier, immediate escalation, hysteresis on de-escalation."""
    def __init__(self, t_low, t_high, hysteresis=3):
        self.t_low, self.t_high, self.h = t_low, t_high, hysteresis
        self.tier, self.down = "high", 0

    def update(self, score):
        want = "low" if score < self.t_low else ("mid" if score < self.t_high else "high")
        escalated = False
        if ORDER[want] > ORDER[self.tier]:
            self.tier, self.down, escalated = want, 0, True
        elif ORDER[want] < ORDER[self.tier]:
            self.down += 1
            if self.down >= self.h:
                self.tier, self.down = want, 0
        else:
            self.down = 0
        return self.tier, escalated


class DecoupledController:
    """PROPOSED. Motion (GPU pixel-difference on every frame) sets the sampling interval;
    object scale (smallest box from the last inference) sets the resolution;
    periodic full-resolution refresh prevents getting stuck at low resolution."""
    def __init__(self, t_low, t_high, k_px, refresh, long_side, hysteresis=3):
        self.temporal = TemporalTier(t_low, t_high, hysteresis)
        self.k, self.refresh, self.L = k_px, refresh, long_side
        self.since, self.since_refresh, self.res = 10**9, 10**9, 640

    def _spatial(self, min_h):
        if min_h is None or math.isinf(min_h):
            return TIERS[0]
        for r in TIERS:                      # smallest tier where the smallest object is >= k px
            if min_h * r / self.L >= self.k:
                return r
        return TIERS[-1]

    def step(self, score, last_min_h):
        tier, escalated = self.temporal.update(score)
        self.since += 1
        self.since_refresh += 1
        if self.since_refresh >= self.refresh:
            self.since, self.since_refresh = 0, 0
            return TIERS[-1]                 # refresh at full resolution
        if escalated or self.since >= INTERVAL[tier]:
            self.since = 0
            return self._spatial(last_min_h)
        return None


class CoupledController:
    """SYNOPSIS BASELINE: one motion score drives resolution AND interval together."""
    RES = {"low": 320, "mid": 480, "high": 640}

    def __init__(self, t_low, t_high, hysteresis=3):
        self.temporal = TemporalTier(t_low, t_high, hysteresis)
        self.since = 10**9

    def step(self, score, last_min_h=None):
        tier, escalated = self.temporal.update(score)
        self.since += 1
        if escalated or self.since >= INTERVAL[tier]:
            self.since = 0
            return self.RES[tier]
        return None


class StaticController:
    """Fixed resolution and fixed interval (incl. the full-rate reference 640/1)."""
    def __init__(self, res, interval):
        self.res, self.interval, self.since = res, interval, 10**9

    def step(self, score=None, last_min_h=None):
        self.since += 1
        if self.since >= self.interval:
            self.since = 0
            return self.res
        return None


class KeyframeDiffController:
    """REDUCTO-STYLE BASELINE: full resolution; infer only when the frame differs enough
    from the LAST INFERRED frame. step() receives that difference, not the consecutive one."""
    def __init__(self, t):
        self.t, self.first = t, True

    def step(self, diff_to_key, last_min_h=None):
        if self.first or diff_to_key >= self.t:
            self.first = False
            return 640
        return None


def family(a, q, long_side):
    """One-parameter policy family for calibration. a=0 is the full-rate reference
    (loss 0 by construction); larger a = more aggressive. q(p) = score quantile of the
    DESIGN split (frozen before calibration)."""
    if a <= 0:
        return StaticController(640, 1)
    t_low, t_high = q(0.30 + 0.60 * a), q(0.60 + 0.35 * a)
    k_px = 24 - 12 * a
    refresh = int(round(25 * (1 + 3 * a)))          # 1 s ... 4 s at 25 fps
    return DecoupledController(t_low, t_high, k_px, refresh, long_side)
