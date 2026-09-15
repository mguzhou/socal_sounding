# Copyright (c) 2015,2016,2017 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""
===============
Simple Sounding
===============

Fetch and plot a Skew-T LogP sounding (plus a comparison run, overlaid)
for a station, using MetPy, siphon, and Open-Meteo.
"""

import argparse
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import reverse_geocoder
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dateutil import parser as dateutil_parser

import metpy.calc as mpcalc
from metpy.plots import SkewT
from metpy.units import units
from scipy.ndimage import median_filter

from siphon.simplewebservice.wyoming import WyomingUpperAir


def describe_location_offline(lat, lon):
    """"City, region" for a lat/lon, via reverse_geocoder's offline
    dataset (a bundled k-d tree of world cities) -- no network call, so
    it can't fail from a network hiccup the way the online option can,
    but it's nearest-city-by-point-distance only: a station that isn't
    itself a city (e.g. a military airfield) can resolve to whichever
    *other* city's center point happens to be geometrically closest,
    even if the station actually sits inside a different city's limits
    (NKX/MCAS Miramar resolves to "La Mesa, California" here, not "San
    Diego", despite being within San Diego city limits -- La Mesa's
    center is ~13 km away, San Diego's ~15 km, and this method has no
    concept of municipal boundaries to prefer the latter).

    mode=1 forces single-process lookup: the default (mode=2) spawns
    worker processes via multiprocessing, which needs a real importable
    __main__ module and breaks when this is driven from something like
    a REPL or a one-off -c invocation. We're only ever looking up one
    point here anyway, so there's no parallelism to gain from mode=2.
    """
    match = reverse_geocoder.search((lat, lon), mode=1)[0]
    region = match['admin1'] or match['cc']
    return f"{match['name']}, {region}"


def describe_location_online(station):
    """"Name, region" for a station, via airportsapi.com's /airports/{code}
    lookup -- keyed on the station's own identifier (ICAO/IATA/local code)
    rather than reverse-geocoded from coordinates, so it names the airport
    itself (e.g. "Miramar Marine Corps Air Station - Mitscher Field,
    California" for NKX) instead of whichever nearby city's center point
    happens to be closest to the launch point -- sidestepping the
    La-Mesa-vs-San-Diego ambiguity describe_location_offline has. No API
    key required. Returns None on any failure (network error, unknown
    station, malformed response, etc.) rather than raising -- labeling
    the plot isn't worth failing the whole run over."""
    try:
        resp = requests.get(f'https://airportsapi.com/api/airports/{station}',
                            params={'include': 'region,country'}, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, ValueError):
        return None

    data = payload.get('data') or {}
    name = data.get('attributes', {}).get('name')
    if not name:
        return None

    included = {(item['type'], item['id']): item for item in payload.get('included', [])}

    def related_name(rel_type):
        ref = data.get('relationships', {}).get(rel_type, {}).get('data')
        if not ref:
            return None
        return included.get((ref['type'], ref['id']), {}).get('attributes', {}).get('name')

    region = related_name('region') or related_name('country')
    return f'{name}, {region}' if region else name


def describe_location(station, lat, lon, method='online'):
    if method == 'online':
        return describe_location_online(station) or describe_location_offline(lat, lon)
    return describe_location_offline(lat, lon)

###########################################


def parse_datetime(value):
    """Parse pretty much any common date/time string (ISO 8601, "Jan 1
    2025 12:00", "2025/01/01 12:00", with or without seconds, etc.) via
    dateutil rather than matching against a fixed list of strptime
    formats -- that list only ever covered the formats someone had
    already thought to add. A bare date/time with no timezone is
    assumed UTC (matching what --datetime has always meant here); one
    that does specify a timezone gets converted to UTC instead of
    having it overwritten.
    """
    try:
        dt = dateutil_parser.parse(value)
    except (ValueError, OverflowError) as e:
        raise argparse.ArgumentTypeError(f'Invalid date/time: {value!r}') from e
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def parse_args():
    parser = argparse.ArgumentParser(description='Fetch and plot a Skew-T sounding.')
    parser.add_argument('--station', default='NKX',
                        help='Station identifier, e.g. NKX (default: %(default)s)')
    parser.add_argument('--altitude-unit', choices=['km', 'kft'], default='km',
                        help='Unit for the MSL altitude axis (default: %(default)s)')
    parser.add_argument('--datetime', type=parse_datetime, default=None,
                        help='UTC date/time of the sounding run to plot, e.g. '
                             '"2025-01-01T12" (default: latest available 00Z/12Z run)')
    parser.add_argument('--geocoder', choices=['offline', 'online'], default='online',
                        help='How to turn the station into a place name for the title: '
                             '"online" (airportsapi.com, looked up by station code -- '
                             'names the airport itself, falls back to "offline" on any '
                             'error) or "offline" (nearest city to the coordinates by '
                             'point distance, no network call) (default: %(default)s)')
    parser.add_argument('--compare', type=parse_datetime, default=None,
                        help='UTC date/time of a specific sounding run to overlay for '
                             'comparison, e.g. "2025-01-01T00" (default: the previous '
                             'synoptic run, 12h earlier, same station)')
    parser.add_argument('--compare-station', default=None,
                        help='Station for the --compare sounding, if different from '
                             '--station (default: same station)')
    return parser.parse_args()


args = parse_args()
station = args.station

# Toggle: overlay the previous synoptic sounding (12h earlier) on top of
# the current one, faded and dashed, for comparison.
SHOW_PREVIOUS = True

# Unit for the MSL altitude axis: 'km' (1 km ticks) or 'kft' (thousands
# of feet, 3 kft ticks).
ALTITUDE_UNIT = args.altitude_unit
KM_TO_KFT = 3.280839895

# Cache fetched soundings locally so re-running the script (e.g. while
# tweaking the plot) doesn't have to hit the Wyoming archive every time.
CACHE_DIR = Path(__file__).parent / 'cache'
CACHE_DIR.mkdir(exist_ok=True)


def load_sounding(date, station):
    cache_file = CACHE_DIR / f'{station}_{date:%Y%m%d_%HZ}.csv'
    if cache_file.exists():
        return pd.read_csv(cache_file, parse_dates=['time'])
    df = WyomingUpperAir.request_data(date.replace(tzinfo=None), station)
    df.to_csv(cache_file, index=False)
    return df


def fetch_recent_sounding(start_date, station, max_tries=4):
    """Load a sounding at start_date, falling back 12h at a time if it
    hasn't posted yet (or doesn't exist for that run)."""
    date = start_date
    for _ in range(max_tries):
        try:
            df = load_sounding(date, station)
            df = df.dropna(subset=('temperature', 'dewpoint', 'direction', 'speed'),
                           how='all')
            # High-resolution modern soundings (e.g. station 72572) can
            # report the same rounded pressure across several consecutive
            # rows -- the pressure sensor's 0.1 hPa resolution is coarser
            # than the actual sampling rate, so height/temperature keep
            # changing while pressure doesn't. That breaks anything
            # (interpolation, the lapse-rate window) that needs pressure
            # to be a valid unique coordinate, so keep only the first row
            # per distinct pressure value.
            df = df.drop_duplicates(subset='pressure').reset_index(drop=True)
            return df, date
        except ValueError:
            date -= timedelta(hours=12)
    raise RuntimeError(f'No recent sounding data available for {station}')


def latest_synoptic_time(now=None):
    """Round down to the most recent 00Z/12Z sounding launch time."""
    now = now or datetime.now(timezone.utc)
    synoptic_hour = 12 if now.hour >= 12 else 0
    return now.replace(hour=synoptic_hour, minute=0, second=0, microsecond=0)


date = args.datetime or latest_synoptic_time()
df, date = fetch_recent_sounding(date, station)

# The comparison sounding defaults to the previous synoptic run (12h
# earlier) for the same station, but --compare/--compare-station let it
# be any run for any station instead.
compare_station = args.compare_station or station
if SHOW_PREVIOUS:
    compare_start = args.compare or (date - timedelta(hours=12))
    compare_df, compare_date = fetch_recent_sounding(compare_start, compare_station)

###########################################
# Special lines. Dry adiabats are clipped so each one starts at the bottom
# of the plot (the x axis, i.e. the highest plotted pressure) and stops at
# the first point it crosses the environmental temperature profile going
# up, instead of continuing on past it.
def plot_clipped_dry_adiabats(skew, t0_values, env_p, env_T, **kwargs):
    bottom, top = skew.ax.get_ylim()
    p_grid = np.linspace(bottom, top, 400) * units.hPa

    order = np.argsort(env_p.m)
    env_T_grid = np.interp(p_grid.m, env_p.m[order], env_T.m[order]) * units.degC

    kwargs.setdefault('color', 'r')
    kwargs.setdefault('linestyle', 'dashed')
    kwargs.setdefault('alpha', 0.5)

    lines = []
    for t0 in t0_values:
        adiabat_T = mpcalc.dry_lapse(p_grid, t0, units.Quantity(1000., 'hPa')).to('degC')
        diff = (adiabat_T - env_T_grid).m

        # Index of the crossing nearest the bottom of the plot (first one
        # hit when scanning up from the x axis); keep only up to there.
        change_idx = np.where(np.diff(np.sign(diff)) != 0)[0]
        cut = change_idx.min() + 1 if change_idx.size else len(diff)
        keep = slice(0, cut)

        line, = skew.ax.plot(adiabat_T[keep].m, p_grid[keep].m, **kwargs)
        lines.append(line)
    return lines


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


def render_lapse_rate_panel(fig, subplot, sounding_df, skew_ax, compare_sounding_df=None):
    """Draw the environmental lapse rate (degC/km) in its own panel next
    to the main Skew-T, sharing skew_ax's y-axis (pressure, log-p, same
    limits) via sharey -- rather than overlaying it on the Skew-T itself
    at some made-up temperature offset, this keeps it a real, correctly
    scaled reading, lined up level-for-level with the main panel just by
    virtue of the shared axis."""
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
    ax.plot(lapse_rate, sounding_df['pressure'].values, color='purple')

    ax.axvline(0, color='grey', linestyle='solid', linewidth=1, alpha=0.6)
    ax.axvline(9.8, color='grey', linestyle='solid', linewidth=1, alpha=0.6)
    
    ax.grid(axis='y', linestyle='dashed', color='gray', alpha=0.3)
    ax.set_xlabel('Lapse rate\n(\N{DEGREE CELSIUS}/km)')
    ax.set_xlim(-4, 14)
    # Pressure/altitude are already labeled on the main panel's own axes;
    # this one just needs to visually line up with it.
    ax.tick_params(labelleft=False)
    ax.yaxis.set_visible(False)

    return ax


def _interp_with_extrap(x, xp, fp, n_fit=5):
    """Like np.interp, but linearly extrapolates past the ends of xp
    instead of clamping to the boundary value, and (unlike
    metpy.interpolate's 1D functions) preserves x's input shape rather
    than always flattening to 1D -- matplotlib's FuncTransform machinery
    calls the forward/inverse functions on a 2D (N, 1) array and expects
    a same-shaped array back, which a flattening implementation breaks.

    The extrapolation slope is fit (least squares) from the nearest
    n_fit points, not just the two closest ones -- adjacent reported
    levels are sometimes only 1 hPa apart, and since height is only
    reported to metre precision, a two-point slope from that close a
    pair is noisy enough to occasionally come out the wrong sign."""
    x = np.asarray(x, dtype=float)
    y = np.interp(x, xp, fp)
    n = min(n_fit, len(xp))

    below = x < xp[0]
    if np.any(below):
        slope, intercept = np.polyfit(xp[:n], fp[:n], 1)
        y[below] = intercept + slope * x[below]

    above = x > xp[-1]
    if np.any(above):
        slope, intercept = np.polyfit(xp[-n:], fp[-n:], 1)
        y[above] = intercept + slope * x[above]

    return y


# Pressure <-> MSL altitude conversion for the secondary y-axis, built by
# interpolating the sounding's own reported geopotential heights (which
# are already referenced to mean sea level) rather than assuming a
# standard-atmosphere formula -- this reflects the atmosphere that was
# actually observed, not a theoretical one. `unit` picks the displayed
# altitude unit ('km' or 'kft'); the underlying pressure data is
# unaffected either way.
#
# Interpolation is done against log(pressure), not raw pressure -- by the
# hypsometric equation, height is approximately linear in ln(p), not in p
# itself (the same reason a skew-T's y-axis is log-p rather than linear).
def make_msl_height_functions(sounding_df, unit='km'):
    scale = KM_TO_KFT if unit == 'kft' else 1.0
    order = np.argsort(sounding_df['pressure'].values)
    logp_sorted = np.log(sounding_df['pressure'].values[order])
    h_sorted = sounding_df['height'].values[order] / 1000. * scale

    def pressure_to_height(pressure_mb):
        pressure_mb = np.atleast_1d(pressure_mb)
        return _interp_with_extrap(np.log(pressure_mb), logp_sorted, h_sorted)

    def height_to_pressure_mb(height_val):
        height_val = np.atleast_1d(height_val)
        h_order = np.argsort(h_sorted)
        logp = _interp_with_extrap(height_val, h_sorted[h_order], logp_sorted[h_order])
        return np.exp(logp)

    return pressure_to_height, height_to_pressure_mb


def render_skewt_panel(fig, subplot, sounding_df, sounding_date, tz_offset, tz_abbr,
                       forecast_high=None, is_forecast=True, location_desc=None):
    """Draw one full Skew-T panel (data, barbs, adiabats, altitude axis,
    title) into the given subplot position of fig."""
    p = sounding_df['pressure'].values * units.hPa
    T = sounding_df['temperature'].values * units.degC
    Td = sounding_df['dewpoint'].values * units.degC
    wind_speed = sounding_df['speed'].values * units.knots
    wind_dir = sounding_df['direction'].values * units.degrees
    u, v = mpcalc.wind_components(wind_speed, wind_dir)

    skew = SkewT(fig, aspect='auto', subplot=subplot, rotation=30)
    
    # SkewT.__init__ turns on a full grid by default, including solid
    # horizontal isobar lines; drop just those (keep the diagonal
    # temperature gridlines, and keep both the pressure and altitude
    # axes with their own ticks/labels).
    skew.ax.yaxis.grid(False)

    # Plot the data using normal plotting functions, in this case using
    # log scaling in Y, as dictated by the typical meteorological plot.
    # The dewpoint trace is median-filtered before plotting -- NKX's
    # dewpoint reports in particular have sharp single-point discontinuities
    # (likely RH sensor lag/noise, not real atmospheric structure), and a
    # median filter kills those spikes without smearing genuine features
    # the way a mean/rolling-average filter would. Td itself (unfiltered)
    # is kept for the parcel profile anchor below -- that's the sounding's
    # actual reported surface value, not something to smooth away.
    skew.plot(p, T, 'r')
    skew.plot(p, median_filter(Td.m, size=5, mode='nearest') * units.degC, 'g')

    skew.ax.axvline(0, color='grey', linestyle='solid', linewidth=1.3, alpha=0.8)
    skew.ax.set_xlabel('Temperature (\N{DEGREE CELSIUS})')

    # Set spacing interval--Every 10 mb from 1000 to 100 mb
    my_interval = np.arange(500, 1010, 10) * units('mbar')

    # Get indexes of values closest to defined interval
    ix = mpcalc.resample_nn_1d(p, my_interval)

    # Plot only values nearest to defined interval values
    skew.plot_barbs(p[ix], u[ix], v[ix])

    # plot_clipped_dry_adiabats(
    #     skew, units.Quantity(np.arange(24, 40, 2.5), 'degC'), p, T,
    #     color='orange',
    #     linestyle='solid',
    #     linewidth=1.5,
    #     alpha=0.7)

    # The underlying SkewT axes are always pressure/log-p internally (that's
    # baked into MetPy's projection), so the 6 km ylim has to be converted
    # to the equivalent pressure bound using this sounding's own data.
    pressure_to_alt, alt_to_pressure_mb = make_msl_height_functions(sounding_df, ALTITUDE_UNIT)
    top_alt = 6.0 * KM_TO_KFT if ALTITUDE_UNIT == 'kft' else 6.0
    top_pressure = alt_to_pressure_mb(top_alt)[0]
    # Bottom of the plot is always 0 MSL (sea level), not the sounding's
    # own lowest reported level -- so every plot spans the same altitude
    # range and is visually comparable run-to-run. This needs the
    # pressure-to-altitude mapping extrapolated a bit past the real data
    # (see _interp_with_extrap), but only down to sea level -- a much
    # smaller gap than a fixed-pressure bottom would need, and stable now
    # that the extrapolation fits a slope from several nearby points.
    #
    # Queried a hair below exactly 0 (not 0.0 itself): the round-trip
    # through the secondary axis's own transform doesn't land back on
    # bit-for-bit 0, so a "0" tick placed exactly on the view boundary
    # was ending up just outside it and getting silently dropped.
    bottom_margin = 0.02 * KM_TO_KFT if ALTITUDE_UNIT == 'kft' else 0.02
    bottom_pressure = alt_to_pressure_mb(-bottom_margin)[0]
    skew.ax.set_ylim(bottom_pressure, top_pressure)
    skew.ax.set_xlim(-5, 50)

    # The high-temp dry adiabat (the "convective temperature" technique)
    # is only drawn when a value is passed in (i.e. the current sounding's
    # panel, not a previous run's). For a recent date this is Open-Meteo's
    # forecast high; for a historical date it's that day's actual observed
    # high, pulled from Open-Meteo's archive instead.
    if forecast_high is not None:
        high_label = 'Forecast high' if is_forecast else 'Observed high'
        forecast_uncertainty = 1.0
        # Full parcel ascent (dry below the LCL, moist above) starting from
        # the high temp at the surface, using the sounding's own surface
        # dewpoint -- shows any CAPE a parcel heated to that day's high
        # would actually have.
        forecast_profile = mpcalc.parcel_profile(
            p, units.Quantity(forecast_high, 'degC'), Td[0]).to('degC')
        forecast_profile_minus = mpcalc.parcel_profile(
            p, units.Quantity(forecast_high - forecast_uncertainty, 'degC'), Td[0]).to('degC')
        forecast_profile_plus = mpcalc.parcel_profile(
            p, units.Quantity(forecast_high + forecast_uncertainty, 'degC'), Td[0]).to('degC')
        skew.plot(p, forecast_profile, color='black', linewidth=1.2, linestyle='solid',
                 label=f'{high_label} parcel ({forecast_high:.1f} \N{PLUS-MINUS SIGN} {forecast_uncertainty:.1f}) \N{DEGREE CELSIUS} - OpenMeteo')
        #)

        # Lifted condensation level for that same forecast-high parcel --
        # marks where it would saturate, i.e. the base of any clouds that
        # afternoon's heating would produce.
        lcl_pressure, lcl_temperature = mpcalc.lcl(
            p[0], units.Quantity(forecast_high, 'degC'), Td[0])
        #skew.plot(lcl_pressure, lcl_temperature, 'ko', markerfacecolor='black',
        #         label='Lifted condensation level - forecast parcel')

        # Convective condensation level and convective temperature: unlike
        # the LCL above (a specific, externally-given parcel lifted
        # mechanically), the CCL asks whether surface heating alone (no
        # forced lift) would trigger convection -- it's where the same
        # surface mixing-ratio line crosses the actual environmental
        # temperature profile, rather than a lifted parcel's dry adiabat.
        # Convective temperature is the surface temperature that would
        # have to be reached for that to happen; the dry adiabat from
        # that temperature up to the CCL is the heating path itself.
        #
        # which='bottom' (nearest-surface crossing), not the default
        # 'top': the mixing-ratio line can cross the environmental
        # profile more than once, and the highest-altitude crossing is
        # sometimes a spurious one near the top of the data (e.g. a
        # profile reaching ~7 hPa can re-cross there by coincidence),
        # giving a nonsensical convective temperature -- 663 degC on one
        # NKX sounding -- instead of the physically meaningful one right
        # above the surface.
        #
        # T/Td are median-filtered before this call specifically (not
        # just Td, and not the raw arrays used elsewhere) because the
        # same small-scale near-surface noise that motivated smoothing
        # the plotted dewpoint trace can *also* fabricate a trivial
        # crossing just a few hPa above the surface -- which='bottom'
        # would then grab that spurious wobble instead of the real CCL,
        # giving a convective temperature barely above actual surface T.
        T_for_ccl = median_filter(T.m, size=5, mode='nearest') * units.degC
        Td_for_ccl = median_filter(Td.m, size=5, mode='nearest') * units.degC
        ccl_pressure, ccl_temperature, convective_temp = mpcalc.ccl(
            p, T_for_ccl, Td_for_ccl, which='bottom')
        skew.plot(ccl_pressure, ccl_temperature, marker='^', color='black',
                 markerfacecolor='none', markersize=9, linestyle='none',
                 label='Convective condensation level')
        conv_path_pressure = units.Quantity(np.linspace(p[0].m, ccl_pressure.m, 50), 'hPa')
        conv_path_temp = mpcalc.dry_lapse(conv_path_pressure, convective_temp).to('degC')
        skew.plot(conv_path_pressure, conv_path_temp, color='darkorange', linewidth=1.2,
                 linestyle='solid', alpha=0.8,
                 label=f'Convective temperature ({convective_temp.m:.1f}\N{DEGREE CELSIUS})')

        # The mixing line (constant mixing ratio) from the surface dewpoint
        # up to the higher of the LCL/CCL -- the classic graphical
        # technique for finding both is where this line (constant
        # moisture) crosses the relevant temperature curve: the lifted
        # parcel's dry adiabat for the LCL, the environmental profile
        # itself for the CCL. Both land on this same line since both are
        # built from the same starting (surface) dewpoint.
        surface_mixing_ratio = mpcalc.saturation_mixing_ratio(p[0], Td[0])
        mixing_line_top = min(lcl_pressure.m, ccl_pressure.m)
        skew.plot_mixing_lines(
            mixing_ratio=np.atleast_1d(surface_mixing_ratio.m),
            pressure=units.Quantity(np.linspace(p[0].m, mixing_line_top, 50), 'hPa'),
            colors='black', linestyles='dashed', linewidths=1, alpha=0.6)
        
        # Meters specifically (not ALTITUDE_UNIT), via a fresh unscaled
        # pressure_to_height so this doesn't depend on whatever display
        # unit the plot happens to be using.
        pressure_to_height_km, _ = make_msl_height_functions(sounding_df, 'km')

        # A surface-based inversion caps mixing at its own top regardless
        # of where a lifted parcel's dry adiabat would otherwise cross
        # the environment (Holzworth's method) -- higher pressure than a
        # crossing means lower altitude, i.e. more restrictive, so the
        # cap is applied as max(crossing_pressure, inversion_top).
        surface_inversion_top = find_surface_inversion_top(
            T.m, sounding_df['height'].values / 1000., p.m)

        # A thermal can't mix past its own LCL -- once it saturates it's
        # ascending as a cloud, not doing dry boundary-layer mixing
        # anymore, so the dry-adiabat/environment crossing (which the
        # full dry+moist parcel_profile can push past the LCL, if the
        # moist adiabat stays warmer than the environment for a while)
        # is capped at whichever of the two is lower/more restrictive.
        def capped_mixing_top(profile, lcl_top_pressure):
            cross_pressure, _ = mpcalc.find_intersections(
                p, profile, T, direction='decreasing', log_x=True)
            top = cross_pressure[0].m if len(cross_pressure) > 0 else lcl_top_pressure
            top = max(top, lcl_top_pressure)
            #if surface_inversion_top is not None:
            #    top = max(top, surface_inversion_top)
            # LCL was the binding constraint (parcel saturates before it
            # runs out of buoyancy) -- this is a cloud base, not a dry
            # thermal top.
            is_cloud_base = np.isclose(top, lcl_top_pressure)
            return top, is_cloud_base

        # Where the *central* forecast-high parcel (no uncertainty offset)
        # first drops back below the environmental temperature -- a
        # subtle single-line best estimate of the mixing top, sitting
        # inside the uncertainty band below.
        mid_top, mid_is_cloud_base = capped_mixing_top(forecast_profile, lcl_pressure.m)
        if mid_top is not None:
            mixing_height_m = pressure_to_height_km(mid_top)[0] * 1000
            mid_label = 'Cloud base' if mid_is_cloud_base else 'Thermal tops'
            skew.ax.axhline(mid_top, color='indigo', linestyle='dotted',
                            linewidth=1, alpha=0.6,
                            label=f'{mid_label} (forecast high, {mixing_height_m:,.0f} m)')

        # Where each bound of the forecast-high uncertainty range first
        # drops back below the environmental temperature -- the top of
        # the layer that parcel would actively mix through. This is the
        # *first* (lowest) parcel/environment crossing, not the *last*
        # one mpcalc.el() finds (the true equilibrium level, generally
        # much higher up on a convective sounding) -- see the earlier
        # discussion on why el() doesn't fit this use.
        mixing_top_pressures = []
        for profile, bound_temp in ((forecast_profile_plus, forecast_high + forecast_uncertainty),
                                    (forecast_profile_minus, forecast_high - forecast_uncertainty)):
            bound_lcl_pressure, _ = mpcalc.lcl(p[0], units.Quantity(bound_temp, 'degC'), Td[0])
            top, _ = capped_mixing_top(profile, bound_lcl_pressure.m)
            mixing_top_pressures.append(top)
            skew.ax.axhline(top, color='steelblue', linestyle='dotted',
                            linewidth=1, alpha=0.7)

        # Shade the pressure band between the two bounds' mixing tops --
        # the range of possible mixing heights given the forecast
        # uncertainty, not just the two edge cases. Labeled with the
        # actual height difference the ±uncertainty translates to, since
        # the same temperature spread can mean very different height
        # spreads depending on how steep the local lapse rate is there.
        if len(mixing_top_pressures) == 2:
            mixing_heights_m = sorted(pressure_to_height_km(pr)[0] * 1000
                                      for pr in mixing_top_pressures)
            mixing_height_range_m = mixing_heights_m[1] - mixing_heights_m[0]
            skew.ax.axhspan(min(mixing_top_pressures), max(mixing_top_pressures),
                            color='steelblue', alpha=0.12,
                            label=f'Thermal top uncertainty ({mixing_height_range_m:,.0f} m)')


        skew.shade_cape(p, T, forecast_profile_plus, alpha = 0.1)
        skew.shade_cin(p, T, forecast_profile_minus, alpha = 0.05)
        skew.plot(p, forecast_profile_minus, color='steelblue', linewidth=.5, linestyle='dashed',alpha=.8)
        skew.plot(p, forecast_profile_plus, color='steelblue', linewidth=.5, linestyle='dashed',alpha=.8)
        skew.ax.fill_betweenx(p, forecast_profile_minus, forecast_profile_plus,
                              color='steelblue', alpha=0.15,
        #                        label=f'{high_label} parcel ({forecast_high:.0f}\N{DEGREE CELSIUS}) - OpenMeteo')
        #                     label=f'\N{PLUS-MINUS SIGN}{forecast_uncertainty:.0f}\N{DEGREE CELSIUS} forecast uncertainty')
        )

        skew.ax.legend(loc='upper left')

    # MSL altitude is the primary axis: it sits at the main (0-offset)
    # position, while the underlying pressure values -- still what actually
    # drives the plot's log-p y-coordinate -- are pushed out to a secondary
    # axis instead.
    skew.ax.tick_params(labelleft=False)
    skew.ax.set_ylabel('')

    # The sounding's own lowest reported level (highest pressure, i.e. the
    # bottom of the plot) gets its own tick showing its actual reported
    # geopotential height -- not assumed to be the station's official
    # elevation (that first level is occasionally a bit above true ground,
    # e.g. if a near-surface reading was dropped), just whatever height
    # this specific sounding reported there.
    scale = KM_TO_KFT if ALTITUDE_UNIT == 'kft' else 1.0
    bottom_level_alt = sounding_df.loc[sounding_df['pressure'].idxmax(), 'height'] / 1000. * scale

    alt_ax = skew.ax.secondary_yaxis(0, functions=(pressure_to_alt, alt_to_pressure_mb))
    alt_ax.set_ylabel(f'MSL Altitude ({ALTITUDE_UNIT})', labelpad=0)
    alt_tick_step = 3 if ALTITUDE_UNIT == 'kft' else 1
    fine_step = alt_tick_step / 10
    # Finer ticks (10 subdivisions) fill in whichever coarse interval
    # actually contains the plot's bottom edge -- not always [0, first
    # tick), since a high-elevation station's own ground can already
    # start above the first regular tick (e.g. Denver, at ~1.6 km, is
    # already past a 1 km first tick).
    fine_start = np.floor(bottom_level_alt / alt_tick_step) * alt_tick_step
    fine_ticks = np.arange(fine_start + fine_step, fine_start + alt_tick_step, fine_step)
    # Starts at fine_start itself (not fine_start + alt_tick_step) so that
    # boundary shows up as a normal round-number tick too -- e.g. "0" at
    # the very bottom of a sea-level plot, where it's otherwise never
    # generated (fine_ticks excludes it, and it only coincides with
    # bottom_level_alt when the ground itself lands exactly on the step).
    coarse_ticks = np.arange(fine_start, top_alt + 0.1, alt_tick_step)
    regular_ticks = np.concatenate([fine_ticks, coarse_ticks])
    alt_ax.set_yticks(np.union1d(regular_ticks, [bottom_level_alt]))
    alt_ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda val, pos: (f'{val:,.2f}' if np.isclose(val, bottom_level_alt)
                          else '' if np.any(np.isclose(val, fine_ticks))
                          else f'{val:,.0f}')))
    alt_ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    # alt_ax.grid() doesn't actually render here -- matplotlib's secondary
    # axis gridline transform gets confused by the inverted range (low
    # pressure/high altitude at the top), so the dashed altitude grid is
    # drawn directly on skew.ax instead. Recomputing the pressure with
    # alt_to_pressure_mb() doesn't reliably land exactly on the tick
    # (the secondary axis's internal transform and a fresh call to our
    # interpolation function don't perfectly round-trip), so instead we
    # read back the tick's actual rendered position and convert that.
    # The observation-start tick gets a solid maroon line instead of the
    # regular dashed gray reference-grid styling, since it marks where the
    # sounding's actual reported data begins rather than an arbitrary
    # round-number reference level -- the plot now extends down to 0 MSL
    # regardless, so this is the only marker for the real ground/balloon
    # launch point.
    fig.canvas.draw()
    for alt_val in alt_ax.get_yticks():
        disp_y = alt_ax.transData.transform((0, alt_val))[1]
        pressure_at_tick = skew.ax.transData.inverted().transform((0, disp_y))[1]
        if np.isclose(alt_val, bottom_level_alt):
            skew.ax.axhline(y=pressure_at_tick, linestyle='solid', alpha=0.7,
                            color='maroon', linewidth=1.2)
        else:
            skew.ax.axhline(
                y=pressure_at_tick,
                linestyle='dashed', alpha=0.5, color='gray', linewidth=0.8)

    pres_ax = skew.ax.secondary_yaxis(-0.15)
    pres_ax.set_ylabel('Pressure (hPa)', labelpad=0)
    pres_ax.set_yticks(np.arange(500, 1001, 100))
    pres_ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda val, pos: f'{val:,.0f}'))
    pres_ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    # Title with the station's local time (converted from the sounding's
    # UTC launch time using the station's timezone offset/abbreviation).
    local_dt = sounding_date.replace(tzinfo=None) + tz_offset
    title_left = f'{station} Observed Sounding'
    title_right = f'{local_dt:%Y-%m-%d %H:%M} {tz_abbr}'
    if location_desc:
        # Give both titles a matching second line -- the location
        # descriptor on the left, blank on the right -- so they stack row
        # for row instead of the left title's second line (which can run
        # long, e.g. a full airport name) colliding with the right
        # title's single line sharing that row.
        title_left += f'\n{location_desc}'
        title_right += '\n'
    skew.ax.set_title(
        title_left,
        fontsize=14,
        loc='left'
    )
    skew.ax.set_title(
        title_right,
        fontsize=14,
        loc='right'
    )

    return skew


# Pull that day's high temp for the station's own coordinates from
# Open-Meteo -- the live forecast endpoint for a recent/upcoming date, or
# the historical archive endpoint for a date outside the forecast
# endpoint's ~3-month window (e.g. the 2025 batch runs below). This also
# gives us the station's local-time offset, used for both panels' titles.
station_lat = df['latitude'].iloc[0]
station_lon = df['longitude'].iloc[0]
location_desc = describe_location(station, station_lat, station_lon, method=args.geocoder)
is_forecast = (datetime.now(timezone.utc) - date) < timedelta(hours=12)
weather_url = ('https://api.open-meteo.com/v1/forecast' if is_forecast
              else 'https://archive-api.open-meteo.com/v1/archive')
target_date = date.strftime('%Y-%m-%d')
forecast = requests.get(
    weather_url,
    params={
        'latitude': station_lat,
        'longitude': station_lon,
        'daily': 'temperature_2m_max',
        'timezone': 'auto',
        'start_date': target_date,
        'end_date': target_date,
    },
).json()
forecast_high = forecast['daily']['temperature_2m_max'][0]
tz_offset = timedelta(seconds=forecast.get('utc_offset_seconds', 0))
tz_abbr = forecast.get('timezone_abbreviation', '')

fig = plt.figure(figsize=(9.5, 10))
gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.14)
skew = render_skewt_panel(fig, gs[0, 0], df, date, tz_offset, tz_abbr,
                          forecast_high=forecast_high, is_forecast=is_forecast,
                          location_desc=location_desc)
render_lapse_rate_panel(fig, gs[0, 1], df, skew.ax,
                        compare_sounding_df=compare_df if SHOW_PREVIOUS else None)
filename_stub = f'{station}_{date:%Y%m%d_%HZ}'

if SHOW_PREVIOUS:
    compare_p = compare_df['pressure'].values * units.hPa
    compare_T = compare_df['temperature'].values * units.degC
    compare_Td = compare_df['dewpoint'].values * units.degC
    compare_local_dt = compare_date.replace(tzinfo=None) + tz_offset

    # Default (no --compare/--compare-station given) reads as "the
    # previous run"; an explicit comparison names the station whenever
    # it differs from the main one, since "Previous" alone would be
    # misleading for an unrelated station/date.
    if args.compare or args.compare_station:
        compare_label = (f'{compare_station} comparison' if compare_station != station
                         else 'Comparison')
    else:
        compare_label = f'Previous {station} observation'

    skew.plot(compare_p, compare_T, color='red', linestyle='dashed', alpha=0.4,
             label=f'{compare_label} ({compare_local_dt:%y-%m-%d %H:%M})')
    skew.plot(compare_p, median_filter(compare_Td.m, size=5, mode='nearest') * units.degC,
             color='green', linestyle='dashed', alpha=0.4)
    skew.ax.legend(loc='upper left')

# Export a web-friendly version: PNG at 2x pixel density for crisp
# rendering on high-DPI screens, with the whitespace margin trimmed.
fig.savefig(f'{filename_stub}.png', dpi=150, bbox_inches='tight')

# SVG is smaller and stays sharp at any zoom level for line art like this;
# use it instead of the PNG if the site can serve vector images.
fig.savefig(f'{filename_stub}.svg', bbox_inches='tight')

# Show the plot
#plt.show()
