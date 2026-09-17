"""Command-line interface: argument parsing and its date handling."""

import argparse
from datetime import timezone

from dateutil import parser as dateutil_parser

from config import MODEL_PRIORITY


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


def parse_args(argv=None):
    """Parse the command line (or an explicit argv list, so an in-process
    caller can build arguments without touching the real sys.argv)."""
    parser = argparse.ArgumentParser(description='Fetch and plot a Skew-T sounding.')
    parser.add_argument('--station', default='NKX',
                        help='Station identifier, e.g. NKX (default: %(default)s)')
    parser.add_argument('--altitude-unit', choices=['km', 'kft'], default='km',
                        help='Unit for the MSL altitude axis (default: %(default)s)')
    parser.add_argument('--datetime', type=parse_datetime, default=None,
                        help='UTC date/time of the sounding run to plot, e.g. '
                             '"2025-01-01T12" (default: latest available 00Z/12Z run)')
    parser.add_argument('--name', default=None,
                        help='Arbitrary label shown in the title above the geocoded '
                             'location, e.g. your own name for the site (default: none)')
    parser.add_argument('--geocoder', choices=['offline', 'online'], default='online',
                        help='How to turn the station into a place name for the title: '
                             '"online" (airportsapi.com, looked up by station code -- '
                             'names the airport itself, falls back to "offline" on any '
                             'error) or "offline" (nearest city to the coordinates by '
                             'point distance, no network call) (default: %(default)s)')
    parser.add_argument('--compare', type=parse_datetime, default=None,
                        help='UTC date/time of a specific sounding run to overlay for '
                             'comparison, e.g. "2025-01-01T00" (default: the previous '
                             'synoptic run, 12h earlier, same station -- for a modeled '
                             '--lat/--lon primary run, no comparison is shown by default '
                             'at all, since it would double an already-expensive fetch; '
                             'pass --compare/--compare-station/--compare-lat to get one)')
    parser.add_argument('--compare-station', default=None,
                        help='Station for the --compare sounding, if different from '
                             '--station (default: same station)')
    parser.add_argument('--lat', type=float, default=None,
                        help='Latitude for a modeled profile at an arbitrary point '
                             '(used instead of --station; requires --lon too)')
    parser.add_argument('--lon', type=float, default=None,
                        help='Longitude for a modeled profile at an arbitrary point '
                             '(used instead of --station; requires --lat too)')
    parser.add_argument('--model', choices=MODEL_PRIORITY, default=None,
                        help='Force a specific model for --lat/--lon instead of the '
                             'default priority chain (%(default)s: ' + ' -> '.join(MODEL_PRIORITY) + ')')
    parser.add_argument('--compare-lat', type=float, default=None,
                        help='Latitude for a modeled profile to overlay for comparison '
                             '(used instead of --compare-station; requires --compare-lon)')
    parser.add_argument('--compare-lon', type=float, default=None,
                        help='Longitude for a modeled profile to overlay for comparison '
                             '(used instead of --compare-station; requires --compare-lat)')
    parser.add_argument('--compare-model', choices=MODEL_PRIORITY, default=None,
                        help='Force a specific model for --compare-lat/--compare-lon '
                             'instead of the default priority chain')
    parser.add_argument('--run-datetime', type=parse_datetime, default=None,
                        help='UTC init time of a specific model run to use, e.g. '
                             '"2026-09-16T15" -- pins the run directly instead of '
                             'resolving one from --datetime/the latest available run '
                             '(fails if that exact hour was not posted, no fallback). '
                             'Only meaningful with --lat/--lon; not supported for GFS. '
                             'Cannot be combined with --datetime.')
    parser.add_argument('--forecast-hour', type=int, default=None,
                        help='Forecast lead in hours from the run to use, e.g. 6 -- '
                             'without --run-datetime, uses the latest available run at '
                             'that lead. Only meaningful with --lat/--lon; not '
                             'supported for GFS. Cannot be combined with --datetime.')
    parser.add_argument('--compare-run-datetime', type=parse_datetime, default=None,
                        help='Same as --run-datetime, for the --compare-lat/'
                             '--compare-lon comparison. Cannot be combined with '
                             '--compare.')
    parser.add_argument('--compare-forecast-hour', type=int, default=None,
                        help='Same as --forecast-hour, for the --compare-lat/'
                             '--compare-lon comparison. Cannot be combined with '
                             '--compare.')
    args = parser.parse_args(argv)
    if (args.lat is None) != (args.lon is None):
        parser.error('--lat and --lon must be given together')
    if (args.compare_lat is None) != (args.compare_lon is None):
        parser.error('--compare-lat and --compare-lon must be given together')
    if args.datetime is not None and (args.run_datetime is not None or args.forecast_hour is not None):
        parser.error('--datetime cannot be combined with --run-datetime/--forecast-hour')
    if args.compare is not None and (args.compare_run_datetime is not None
                                     or args.compare_forecast_hour is not None):
        parser.error('--compare cannot be combined with --compare-run-datetime/--compare-forecast-hour')
    return args
