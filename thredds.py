"""Modeled vertical profiles served by Unidata's THREDDS, subset with
NCSS -- currently GFS, the global, coarse fallback for points HRRR and
RRFS do not cover.

Named for the access mechanism rather than the model, the same way
grib.py is: the two are separate modules because they share no
machinery at all. Nothing here does a GRIB2 decode, a byte-range fetch,
or any caching -- THREDDS does the subsetting server-side and hands
back exactly the profile asked for."""

from datetime import datetime, timezone

import metpy.calc as mpcalc
import numpy as np
import pandas as pd
from metpy.units import units
from siphon.catalog import TDSCatalog
from siphon.ncss import NCSS


def fetch_gfs_profile(lat, lon, date, forecast_hour=None, run_datetime=None):
    """Modeled vertical profile at the nearest GFS (0.25 deg global) grid
    point, via Unidata's THREDDS 'Best' time series -- a virtual
    aggregation across GFS cycles that NCSS can query at any valid time
    within its ~8 day rolling window (forecast or analysis) without
    having to resolve which specific run/cycle covers it. Global
    coverage, but coarse (~28 km) compared to HRRR/RAP -- the fallback
    for points HRRR doesn't cover or when HRRR's feed is unreachable.

    GFS's grid is a plain lat/lon grid, so unlike HRRR its u/v wind
    components are already earth-relative -- no rotation needed.

    date=None means "whatever's freshest" -- the current time, which the
    'Best' series resolves to its most recent available data. Unlike
    HRRR/RRFS, the specific run/cycle behind a given valid time isn't
    exposed by this NCSS query, so the returned "run time" is just the
    valid time itself (no separate init-time label is shown for GFS);
    pinning a specific run/lead (forecast_hour/run_datetime) isn't
    supported here for the same reason -- pass --model hrrr or --model
    rrfs to use those instead."""
    if forecast_hour is not None or run_datetime is not None:
        raise RuntimeError('--run-datetime/--forecast-hour are not supported for GFS '
                           '(its NCSS "Best" series does not expose individual cycles) '
                           '-- use --model hrrr or --model rrfs instead')
    if date is None:
        date = datetime.now(timezone.utc)
    cat = TDSCatalog('https://thredds.ucar.edu/thredds/catalog/grib/NCEP/'
                     'GFS/Global_0p25deg/catalog.xml')
    ds = cat.datasets['Best GFS Quarter Degree Forecast Time Series']
    ncss = NCSS(ds.access_urls['NetcdfSubset'])

    query = ncss.query()
    query.lonlat_point(lon, lat)
    query.time(date)
    query.accept('csv')
    query.variables('Temperature_isobaric', 'Relative_humidity_isobaric',
                    'u-component_of_wind_isobaric', 'v-component_of_wind_isobaric',
                    'Geopotential_height_isobaric')
    data = ncss.get_data(query)

    T = data['Temperature_isobaric'] - 273.15
    # RH is exactly 0% at some upper-level points (genuinely dry air, not
    # missing data) -- dewpoint_from_relative_humidity takes log(RH) internally,
    # so an exact 0 produces -inf/NaN rather than a very cold-but-finite Td.
    # A tiny floor keeps the value physically meaningless-but-finite instead
    # of NaN, without perturbing any level that isn't already near-zero RH.
    rh = np.clip(data['Relative_humidity_isobaric'], 0.01, None)
    Td = mpcalc.dewpoint_from_relative_humidity(
        units.Quantity(T, 'degC'), units.Quantity(rh, 'percent')).m
    u, v = data['ucomponent_of_wind_isobaric'], data['vcomponent_of_wind_isobaric']

    df = pd.DataFrame({
        'pressure': data['alt'] / 100.,  # Pa -> hPa
        'height': data['Geopotential_height_isobaric'],
        'temperature': T,
        'dewpoint': Td,
        'speed': np.hypot(u, v) * 1.9438445,  # m/s -> kt
        'direction': np.degrees(np.arctan2(-u, -v)) % 360,
        'latitude': data['latitude'][0],
        'longitude': data['longitude'][0],
    }).sort_values('pressure', ascending=False).reset_index(drop=True)
    valid_time = pd.Timestamp(str(data['time'][0])).to_pydatetime().replace(tzinfo=timezone.utc)
    return df, valid_time, valid_time
