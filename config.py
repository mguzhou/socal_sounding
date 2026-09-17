"""Shared configuration: values more than one module needs.

Deliberately free of project imports, so every other module can
import this without any risk of an import cycle."""

from pathlib import Path


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
