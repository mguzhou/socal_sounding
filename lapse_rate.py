"""The lapse-rate side panel: environmental lapse rate, surface
inversion detection, and the parcel-minus-environment overlay."""

import numpy as np


# Environmental lapse rate (degC/km, positive = cooling with height) at
# each level.
#
# The slope is fit (least squares) over a small centered window of
# points rather than taken point-to-point: real reported levels are
# irregularly spaced (as close as 1 hPa apart in this data) and height
# is only reported to metre precision, so a naive point-to-point
# dT/dz is dominated by rounding noise rather than the actual lapse
# rate. Same reasoning as _interp_with_extrap's windowed slope fit.
#
# The window itself is derived from the sounding's own typical level
# spacing (median |delta pressure|) rather than a fixed sample count --
# a modern high-resolution sounding (e.g. station 72572, ~0.2 hPa
# spacing) reports many more levels per hPa than NKX's traditional
# ~2.5 hPa spacing, so a fixed 5-sample window covers a wildly different
# (and much noisier) physical depth there. NKX_BASELINE_WINDOW at
# NKX_BASELINE_DELTA_P is the reference point that was tuned by eye on
# NKX soundings; other stations get a window scaled to cover roughly
# that same pressure depth, just expressed in however many samples that
# takes locally -- the windowing itself still operates on plain sample
# counts (np.polyfit over a sliced index range), not a distance-based
# mask, to keep the implementation simple.
NKX_BASELINE_DELTA_P = 2.5  # hPa
NKX_BASELINE_WINDOW = 8  # samples, at NKX_BASELINE_DELTA_P spacing


def compute_lapse_rate(T_degc, height_km, pressure_hpa, min_window=3):
    delta_p = np.median(np.abs(np.diff(pressure_hpa)))
    if delta_p > 0 and np.isfinite(delta_p):
        window = round(NKX_BASELINE_WINDOW * NKX_BASELINE_DELTA_P / delta_p)
        window = max(min_window, window)
        if window % 2 == 0:
            window += 1  # odd, so it's centered on each point
    else:
        window = NKX_BASELINE_WINDOW

    n = len(T_degc)
    half = window // 2
    lapse_rate = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        # Some stations (e.g. 72572) report a handful of levels with a
        # missing height (temperature/dewpoint present, no GPS height
        # fix) -- np.polyfit chokes on NaN input with a LAPACK-level
        # SVD-non-convergence error rather than a clean exception, so
        # those points need filtering out of the window, not just a
        # short-window count check.
        finite = np.isfinite(height_km[lo:hi]) & np.isfinite(T_degc[lo:hi])
        if finite.sum() < 2:
            lapse_rate[i] = 0.
            continue
        slope = np.polyfit(height_km[lo:hi][finite], T_degc[lo:hi][finite], 1)[0]
        lapse_rate[i] = -slope
    return lapse_rate


def find_surface_inversion_top(T_degc, height_km, pressure_hpa):
    """Pressure at the top of a surface-based inversion (temperature
    increasing with height starting right at the ground), or None if
    the lowest level isn't already inverted.

    Per the Holzworth mixing-height method: a surface inversion caps
    mixing at its top, regardless of where a lifted parcel's dry adiabat
    would otherwise cross the environmental profile -- thermals can't
    mix through an inversion sitting right on the ground no matter how
    hot the forecast high is. Elevated inversions (aloft, not touching
    the surface) aren't handled here; those need their own capping rule
    (use the inversion's base, not top) and weren't asked for."""
    lapse_rate = compute_lapse_rate(T_degc, height_km, pressure_hpa)
    if lapse_rate[0] >= 0:
        return None  # not inverted at the surface

    recovered = np.flatnonzero(lapse_rate >= 0)
    if recovered.size == 0:
        return None  # inversion never ends within the reported profile
    return pressure_hpa[recovered[0]]


def render_lapse_rate_panel(fig, subplot, sounding_df, skew_ax, compare_sounding_df=None,
                            parcel_p=None, parcel_profile=None, parcel_env_T=None):
    """Draw the environmental lapse rate (degC/km) in its own panel next
    to the main Skew-T, sharing skew_ax's y-axis (pressure, log-p, same
    limits) via sharey -- rather than overlaying it on the Skew-T itself
    at some made-up temperature offset, this keeps it a real, correctly
    scaled reading, lined up level-for-level with the main panel just by
    virtue of the shared axis.

    parcel_p/parcel_profile/parcel_env_T (from render_skewt_panel, same
    pressure range) additionally draw the parcel-minus-environment
    temperature delta -- degC, a genuinely different physical quantity
    from the lapse rate's degC/km (a rate vs. a static difference), so
    it gets its own x-axis (twinned at the top) rather than sharing the
    lapse rate's scale -- the two only coincidentally run similar
    numeric ranges for a typical sounding, which doesn't make "5" mean
    the same thing on both. It's shown here anyway (not as a fully
    separate panel) because the two are physically linked and useful to
    read level-for-level together: this is essentially unscaled parcel
    buoyancy (the quantity CAPE/CIN integrate, minus the g/T_v scaling
    and without using virtual temperature), and the lapse rate is often
    *why* it does what it does -- e.g. a near-isothermal layer (lapse
    rate near 0) is exactly where a dry-adiabatically-cooling parcel
    falls behind the environment and buoyancy drops. Positive is where
    the parcel is warmer than its surroundings (buoyant/CAPE), negative
    where it's colder (CIN), crossing zero exactly at the
    parcel/environment intersections the main panel's CAPE/CIN shading
    is built from."""
    ax = fig.add_subplot(subplot, sharey=skew_ax)

    if compare_sounding_df is not None:
        compare_height_km = compare_sounding_df['height'].values / 1000.
        compare_lapse_rate = compute_lapse_rate(compare_sounding_df['temperature'].values, compare_height_km,
                                                compare_sounding_df['pressure'].values)
        ax.plot(compare_lapse_rate, compare_sounding_df['pressure'].values,
               color='purple', linestyle='dashed', alpha=0.3)

    height_km = sounding_df['height'].values / 1000.
    lapse_rate = compute_lapse_rate(sounding_df['temperature'].values, height_km,
                                    sounding_df['pressure'].values)
    ax.plot(lapse_rate, sounding_df['pressure'].values, color='purple', label='Lapse rate')

    ax.axvline(0, color='grey', linestyle='solid', linewidth=1, alpha=0.6)
    ax.axvline(9.8, color='grey', linestyle='solid', linewidth=1, alpha=0.6)
    ax.set_xlabel('Lapse rate (\N{DEGREE CELSIUS}/km)')
    ax.set_xlim(-4, 14)

    if parcel_p is not None:
        # twiny(), not a shared x-axis: an independent x-scale (own
        # xlim, own ticks, drawn at the top of the panel instead of the
        # bottom) for this differently-dimensioned quantity, while still
        # sharing ax's y-axis so the two stay level-for-level comparable.
        ax2 = ax.twiny()
        delta_T = parcel_profile.m - parcel_env_T.m
        ax2.plot(delta_T, parcel_p.m, color='crimson', linewidth=1.2,
                label='Parcel \N{MINUS SIGN} environment')
        ax2.set_xlim(-0.2, 1.2)

        ax2.axvline(0, color='Crimson', linestyle='dotted', linewidth=1, alpha=0.6)
        ax2.set_xlabel('Parcel \N{MINUS SIGN} environment (\N{DEGREE CELSIUS})', color='Black')
        ax2.tick_params(axis='x')
        handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
        labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
        ax.legend(handles, labels, loc='upper right', fontsize=8)

    ax.grid(axis='y', linestyle='dashed', color='gray', alpha=0.3)
    # Pressure/altitude are already labeled on the main panel's own axes;
    # this one just needs to visually line up with it.
    ax.tick_params(labelleft=False)
    ax.yaxis.set_visible(False)

    return ax
