"""Shared configuration: values and small helpers more than one module
needs.

Deliberately free of project imports, so every other module can
import this without any risk of an import cycle."""

from datetime import timedelta
from pathlib import Path


def gmt_offset_label(dt):
    """'GMT-7' for an aware datetime's own UTC offset, to sit in front of
    the zone's abbreviation in a title ("14:00 GMT-7 PDT").

    Taken from the datetime rather than the zone, so it reports the
    offset in force on *that* date rather than today's -- the whole point
    of converting through a real timezone. Half-hour and quarter-hour
    zones come out as 'GMT+5:30' / 'GMT+5:45'."""
    total_minutes = int((dt.utcoffset() or timedelta(0)).total_seconds()) // 60
    sign = '-' if total_minutes < 0 else '+'
    hours, minutes = divmod(abs(total_minutes), 60)
    return f'GMT{sign}{hours}' + (f':{minutes:02d}' if minutes else '')


# Priority order for modeled (arbitrary lat/lon) profiles -- RRFS first
# (NOAA's newer HRRR/RAP successor), then HRRR, then GFS as the global
# fallback. Defined here (rather than down by the fetch functions that
# use it) since parse_args() needs it for --model/--compare-model's
# choices.
MODEL_PRIORITY = ['rrfs', 'hrrr', 'gfs']


# Toggle: overlay the previous synoptic sounding (12h earlier) on top of
# the current one, faded and dashed, for comparison.
SHOW_PREVIOUS = True

# Default unit for the MSL altitude axis: 'km' (1 km ticks) or 'kft'
# (thousands of feet, 3 kft ticks). --altitude-unit overrides it per run,
# threaded through to render_skewt_panel rather than read from here, so
# nothing in this module depends on command-line state having been parsed.
DEFAULT_ALTITUDE_UNIT = 'km'
KM_TO_KFT = 3.280839895

# Cache fetched soundings locally so re-running the script (e.g. while
# tweaking the plot) doesn't have to hit the Wyoming archive every time.
CACHE_DIR = Path(__file__).parent / 'cache'
CACHE_DIR.mkdir(exist_ok=True)
