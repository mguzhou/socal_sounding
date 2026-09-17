"""Observed radiosonde soundings from the University of Wyoming
upper-air archive: real balloon launches, fixed station locations,
twice daily. See grib.py/thredds.py for the modeled alternative at an
arbitrary point."""

from datetime import datetime, timedelta, timezone

import pandas as pd
from siphon.simplewebservice.wyoming import WyomingUpperAir

from config import CACHE_DIR


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
