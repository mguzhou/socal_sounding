# Copyright (c) 2015,2016,2017 MetPy Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
"""
===============
NKX Sounding
===============

NKX-only variant of Simple_Sounding.py. Station and source are hardcoded
(not CLI flags) so this file is a place to bolt on NKX-specific hardcoded
tweaks -- e.g. local terrain/sea-breeze quirks, station-specific axis
bounds -- without adding conditionals to the general script or affecting
other stations. Pull generally-useful changes back into Simple_Sounding.py;
leave only the NKX-only stuff here.

Fetch and plot a Skew-T LogP sounding (plus the previous run, overlaid)
for NKX, using MetPy, siphon, and Open-Meteo.
"""

import argparse
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from pathlib import Path

import metpy.calc as mpcalc
from metpy.plots import SkewT
from metpy.units import units

from siphon.simplewebservice.wyoming import WyomingUpperAir
from siphon.simplewebservice.igra2 import IGRAUpperAir

# IGRA2 reports wind speed in m/s; Wyoming (and everything downstream in
# this script) uses knots, so IGRA2 speeds get converted on load to keep
# render_panel source-agnostic.
MPS_TO_KNOTS = 1.9438444924

###########################################
# Fixed for this variant -- not exposed as CLI flags. See module docstring.
station = 'NKX'
source = 'wyoming'

###########################################
# --- NKX-specific tweaks (hardcoded, not generalized to other stations) ---
# Nothing here yet; add station-specific constants/overrides in this block
# as they come up (e.g. a different ALTITUDE cap, a local sea-breeze
# adiabat, a station-specific xlim). Keep anything that would make sense
# for other stations too back in Simple_Sounding.py instead.


def parse_datetime(value):
    for fmt in ('%Y-%m-%dT%H:%M', '%Y-%m-%dT%H', '%Y-%m-%d %H:%M', '%Y-%m-%d %H', '%Y-%m-%d'):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f'Invalid date/time: {value!r} (expected e.g. "2025-01-01T12" or "2025-01-01 12:00")')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fetch and plot a Skew-T sounding for NKX.')
    parser.add_argument('--altitude-unit', choices=['km', 'kft'], default='km',
                        help='Unit for the MSL altitude axis (default: %(default)s)')
    parser.add_argument('--datetime', type=parse_datetime, default=None,
                        help='UTC date/time of the sounding run to plot, e.g. '
                             '"2025-01-01T12" (default: latest available 00Z/12Z run)')
    return parser.parse_args()


args = parse_args()

# Toggle: overlay the previous synoptic sounding (12h earlier) on top of
# the current one, faded and dashed, for comparison.
SHOW_PREVIOUS = True

# Unit for the MSL altitude axis: 'km' (1 km ticks) or 'kft' (thousands
# of feet, 3 kft ticks).
ALTITUDE_UNIT = args.altitude_unit
KM_TO_KFT = 3.280839895

# Cache fetched soundings locally so re-running the script (e.g. while
# tweaking the plot) doesn't have to hit the archive every time.
CACHE_DIR = Path(__file__).parent / 'cache'
CACHE_DIR.mkdir(exist_ok=True)


def fetch_igra2(date, station):
    """Fetch one IGRA2 sounding and reshape it to match the Wyoming
    columns/units the rest of this script expects (knots, per-row
    latitude/longitude, etc.)."""
    df, header = IGRAUpperAir.request_data(date, station)
    df['latitude'] = header['latitude'].iloc[0]
    df['longitude'] = header['longitude'].iloc[0]
    df['speed'] = df['speed'] * MPS_TO_KNOTS
    return df


def load_sounding(date, station, source):
    cache_file = CACHE_DIR / f'{source}_{station}_{date:%Y%m%d_%HZ}.csv'
    if cache_file.exists():
        parse_dates = ['time'] if source == 'wyoming' else None
        return pd.read_csv(cache_file, parse_dates=parse_dates)
    if source == 'wyoming':
        df = WyomingUpperAir.request_data(date.replace(tzinfo=None), station)
    else:
        df = fetch_igra2(date.replace(tzinfo=None), station)
    df.to_csv(cache_file, index=False)
    return df


def fetch_recent_sounding(start_date, station, source, max_tries=4):
    """Load a sounding at start_date, falling back 12h at a time if it
    hasn't posted yet (or doesn't exist for that run)."""
    date = start_date
    for _ in range(max_tries):
        try:
            df = load_sounding(date, station, source)
            df = df.dropna(subset=('temperature', 'dewpoint', 'direction', 'speed'),
                           how='all').reset_index(drop=True)
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
df, date = fetch_recent_sounding(date, station, source)

if SHOW_PREVIOUS:
    prev_df, prev_date = fetch_recent_sounding(date - timedelta(hours=12), station, source)

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


# Pressure <-> MSL altitude conversion for the secondary y-axis, built by
# interpolating the sounding's own reported geopotential heights (which
# are already referenced to mean sea level) rather than assuming a
# standard-atmosphere formula -- this reflects the atmosphere that was
# actually observed, not a theoretical one. `unit` picks the displayed
# altitude unit ('km' or 'kft'); the underlying pressure data is
# unaffected either way.
def make_msl_height_functions(sounding_df, unit='km'):
    scale = KM_TO_KFT if unit == 'kft' else 1.0
    order = np.argsort(sounding_df['pressure'].values)
    p_sorted = sounding_df['pressure'].values[order]
    h_sorted = sounding_df['height'].values[order] / 1000. * scale

    def pressure_to_height(pressure_mb):
        pressure_mb = np.atleast_1d(pressure_mb)
        return np.interp(pressure_mb, p_sorted, h_sorted)

    def height_to_pressure_mb(height_val):
        height_val = np.atleast_1d(height_val)
        h_order = np.argsort(h_sorted)
        return np.interp(height_val, h_sorted[h_order], p_sorted[h_order])

    return pressure_to_height, height_to_pressure_mb


def render_panel(fig, subplot, sounding_df, sounding_date, tz_offset, tz_abbr,
                  forecast_high=None, is_forecast=True):
    """Draw one full Skew-T panel (data, barbs, adiabats, altitude axis,
    title) into the given subplot position of fig."""
    p = sounding_df['pressure'].values * units.hPa
    T = sounding_df['temperature'].values * units.degC
    Td = sounding_df['dewpoint'].values * units.degC
    wind_speed = sounding_df['speed'].values * units.knots
    wind_dir = sounding_df['direction'].values * units.degrees
    u, v = mpcalc.wind_components(wind_speed, wind_dir)

    skew = SkewT(fig, aspect='auto', subplot=subplot, rotation=37.5)

    # Plot the data using normal plotting functions, in this case using
    # log scaling in Y, as dictated by the typical meteorological plot
    skew.plot(p, T, 'r')
    skew.plot(p, Td, 'g')

    skew.ax.set_xlabel('Temperature (\N{DEGREE CELSIUS})')

    # Set spacing interval--Every 50 mb from 1000 to 100 mb
    my_interval = np.arange(100, 1000, 50) * units('mbar')

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
    skew.ax.set_ylim(1000, top_pressure)
    skew.ax.set_xlim(-5, 55)

    # The high-temp dry adiabat (the "convective temperature" technique)
    # is only drawn when a value is passed in (i.e. the current sounding's
    # panel, not a previous run's). For a recent date this is Open-Meteo's
    # forecast high; for a historical date it's that day's actual observed
    # high, pulled from Open-Meteo's archive instead.
    if forecast_high is not None:
        high_label = 'Forecast high' if is_forecast else 'Observed high'

        # Full parcel ascent (dry below the LCL, moist above) starting from
        # the high temp at the surface, using the sounding's own surface
        # dewpoint -- shows any CAPE a parcel heated to that day's high
        # would actually have.
        forecast_profile = mpcalc.parcel_profile(
            p, units.Quantity(forecast_high, 'degC'), Td[0]).to('degC')
        skew.plot(p, forecast_profile, color='black', linewidth=2, linestyle='dashed',
                 label=f'{high_label} parcel ({forecast_high:.0f}\N{DEGREE CELSIUS}) - OpenMeteo')

        skew.ax.legend(loc='upper left')

    # MSL altitude is the primary axis: it sits at the main (0-offset)
    # position, while the underlying pressure values -- still what actually
    # drives the plot's log-p y-coordinate -- are pushed out to a secondary
    # axis instead.
    skew.ax.tick_params(labelleft=False)
    skew.ax.set_ylabel('')

    alt_ax = skew.ax.secondary_yaxis(0, functions=(pressure_to_alt, alt_to_pressure_mb))
    alt_ax.set_ylabel(f'MSL Altitude ({ALTITUDE_UNIT})', labelpad=44)
    alt_tick_step = 3 if ALTITUDE_UNIT == 'kft' else 1
    alt_ax.set_yticks(np.arange(0, top_alt + 0.1, alt_tick_step))
    alt_ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda val, pos: f'{val:,.0f}'))
    alt_ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    # alt_ax.grid() doesn't actually render here -- matplotlib's secondary
    # axis gridline transform gets confused by the inverted range (low
    # pressure/high altitude at the top), so the dashed altitude grid is
    # drawn directly on skew.ax instead. Recomputing the pressure with
    # alt_to_pressure_mb() doesn't reliably land exactly on the tick
    # (the secondary axis's internal transform and a fresh call to our
    # interpolation function don't perfectly round-trip), so instead we
    # read back the tick's actual rendered position and convert that.
    fig.canvas.draw()
    for alt_val in alt_ax.get_yticks():
        disp_y = alt_ax.transData.transform((0, alt_val))[1]
        pressure_at_tick = skew.ax.transData.inverted().transform((0, disp_y))[1]
        skew.ax.axhline(
            y=pressure_at_tick,
            linestyle='dashed', alpha=0.5, color='gray', linewidth=0.8)

    pres_ax = skew.ax.secondary_yaxis(-0.06)
    pres_ax.set_ylabel('Pressure (hPa)', labelpad=10)
    pres_ax.set_yticks(np.arange(500, 1001, 100))
    pres_ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda val, pos: f'{val:,.0f}'))
    pres_ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    # Add a bold 0C isotherm for readability
    skew.ax.axvline(0, color='grey', linestyle='', linewidth=2)

    # Title with the station's local time (converted from the sounding's
    # UTC launch time using the station's timezone offset/abbreviation).
    local_dt = sounding_date.replace(tzinfo=None) + tz_offset
    skew.ax.set_title(
        f'{station} Radiosonde',
        fontsize=14,
        loc='left'
    )
    skew.ax.set_title(
        f'{local_dt:%Y-%m-%d %H:%M} {tz_abbr}',
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
is_forecast = (datetime.now(timezone.utc) - date) < timedelta(days=90)
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

fig = plt.figure(figsize=(7.5, 10))
skew = render_panel(fig, (1, 1, 1), df, date, tz_offset, tz_abbr,
                    forecast_high=forecast_high, is_forecast=is_forecast)
filename_stub = f'{station}_{date:%Y%m%d_%HZ}'

if SHOW_PREVIOUS:
    prev_p = prev_df['pressure'].values * units.hPa
    prev_T = prev_df['temperature'].values * units.degC
    prev_Td = prev_df['dewpoint'].values * units.degC
    prev_local_dt = prev_date.replace(tzinfo=None) + tz_offset

    skew.plot(prev_p, prev_T, color='red', linestyle='dashed', alpha=0.2,
             label=f'Previous {station} Sounding ({prev_local_dt:%m-%d %H:%M})')
    skew.plot(prev_p, prev_Td, color='green', linestyle='dashed', alpha=0.2)
    skew.ax.legend(loc='upper left')

# Export a web-friendly version: PNG at 2x pixel density for crisp
# rendering on high-DPI screens, with the whitespace margin trimmed.
fig.savefig(f'{filename_stub}.png', dpi=150, bbox_inches='tight')

# SVG is smaller and stays sharp at any zoom level for line art like this;
# use it instead of the PNG if the site can serve vector images.
fig.savefig(f'{filename_stub}.svg', bbox_inches='tight')

# Show the plot
#plt.show()
