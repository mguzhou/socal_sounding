# Simple_Sounding.py — change log

Started as MetPy's stock "Simple Sounding" example. Summary of everything
changed since, roughly in the order it happened.

## Configuration (CLI args)

Originally these were environment variables (`SOUNDING_STATION`,
`ALTITUDE_UNIT`); converted to proper `argparse` flags for discoverability
(`--help` documents them) and so a single shell command captures the full
invocation instead of needing exported env vars alongside it.

- `--station STATION` — station identifier, e.g. `NKX` (default: `NKX`).
- `--altitude-unit {km,kft}` — unit for the MSL altitude axis (default:
  `km`). `km` ticks every 1 km; `kft` (thousands of feet, `KM_TO_KFT =
  3.280839895`) ticks every 3 kft.
- `--datetime` — UTC date/time of the sounding run to plot, e.g.
  `"2025-01-01T12"` or `"2025-01-01 12:00"` (default: the latest
  available 00Z/12Z run). Accepts a handful of formats via
  `parse_datetime()`.

`SHOW_PREVIOUS` (overlay the prior run) is still a plain `True`/`False`
constant in the script, not yet exposed as a flag.

## Rendering / display

- **Backend switched to QtAgg** (`matplotlib.use('QtAgg')`, `PyQt6`
  installed into the venv). The original `TkAgg` backend rendered the
  window at the wrong pixel size under GNOME's fractional display scaling
  on Wayland (Tk has no native Wayland support and runs via XWayland,
  which doesn't handle fractional scaling correctly). Qt handles it fine.
- Fixed an invalid `figsize=(9, 9, "in")` call (matplotlib silently
  ignored the stray `"in"`; now just `(9, 9)`/`(7.5, 10)`).
- `SkewT(..., aspect='auto')` plus tighter `xlim`/`ylim` so the skewed
  plot area fills the figure instead of leaving large blank margins
  (MetPy's default fixed `aspect=80.5` centers a fixed-shape box with
  padding regardless of the figure's actual dimensions).
- Title added via `skew.ax.set_title()` (not `fig.suptitle()` — the
  latter anchors to the whole figure and left a large gap above the
  axes; `set_title()` sits directly above the plot).

## Data fetching

- **Latest-synoptic-time lookup**: `latest_synoptic_time()` rounds down
  to the most recent 00Z/12Z launch; `fetch_recent_sounding()` retries
  12 hours further back (up to 4 tries) since a sounding takes ~1-2
  hours to post after launch.
- **Local caching**: fetched soundings are saved as CSV under
  `cache/<station>_<run>.csv`; re-running the script re-reads the cache
  instead of re-hitting the Wyoming archive. Cache is per-run, so it
  never goes stale for a past sounding.
- **Open-Meteo integration** (`requests`, no API key): pulls that day's
  high temperature for the sounding station's own coordinates (read
  straight from the fetched sounding's `latitude`/`longitude` columns),
  plus the station's local UTC offset/timezone abbreviation used for the
  title and legend times.
- **Forecast vs. historical Open-Meteo endpoint**: dates within ~90 days
  use the live `/v1/forecast` endpoint (real forecast high, legend reads
  "Forecast high"); older dates automatically switch to the
  `/v1/archive` historical endpoint instead (actual observed high,
  legend reads "Observed high") — the live forecast endpoint doesn't
  serve data that far back, which matters for the 2025 batch runs below.

## Plot content

- Fixed a copy/paste bug: `np.arange(22, 24, 26, 28)` (4 positional args
  misinterpreted the 4th as a `dtype`) → an explicit `np.array([...])`.
- **`plot_clipped_dry_adiabats()`**: a hand-rolled replacement for
  MetPy's `plot_dry_adiabats()`, which has no awareness of the actual
  temperature profile and draws full-height reference lines. This
  version computes each dry adiabat, finds where it first crosses the
  environmental temperature curve going up from the surface, and draws
  only the segment below that crossing — so lines stop where they meet
  real data instead of running the full plot height.
- **Forecast-high adiabat**: the Open-Meteo forecast high is drawn as
  a special dry adiabat (the "convective temperature" technique) —
  where it crosses the morning sounding shows the forecast afternoon
  mixed-layer depth.
- **Previous-sounding overlay** (`SHOW_PREVIOUS` toggle): the prior
  synoptic run's temperature/dewpoint traces are overlaid on the same
  axes as dashed, faded (`alpha=0.4`) lines for comparison, labeled with
  their local time in the legend. (An earlier side-by-side two-panel
  version was tried and reverted in favor of this simpler overlay.)

## Axes

- **MSL altitude axis**: originally added as a secondary axis using
  MetPy's `pressure_to_height_std`/`height_to_pressure_std` (a fixed
  ICAO standard-atmosphere formula). Replaced with
  `make_msl_height_functions()`, which interpolates the sounding's own
  reported `height` column (real geopotential height, MSL-referenced)
  instead — this reflects the atmosphere actually observed that run,
  not a theoretical one.
- **Axis roles swapped**: MSL altitude (km) is now the primary
  (innermost) y-axis, capped at a 6000 m `ylim` (converted to the
  equivalent pressure bound via the sounding's own data, since the
  underlying SkewT projection is always pressure/log-p internally).
  Pressure (hPa) was moved to a secondary, outer axis.
- Along the way, fixed axis-label collisions (widened the figure,
  adjusted secondary-axis offsets/`labelpad`), a mismatched unit label
  (axis said "(m)" while actually plotting km), scientific-notation tick
  labels on log-scale secondary axes (explicit `FixedLocator`/
  `FuncFormatter` instead of the default log formatter), and a stray
  auto-generated "hectopascal" label (Pint/MetPy's matplotlib units
  integration auto-labels axes from Quantity units; needed an explicit
  `skew.ax.set_ylabel('')` to clear it).
- **`--altitude-unit kft`**: `make_msl_height_functions()` takes a `unit`
  argument and scales the interpolated heights by `KM_TO_KFT`; tick
  spacing switches to every 3 kft (vs. every 1 km) accordingly.
- **Dashed MSL altitude gridlines**: `alt_ax.grid()` silently produced
  gridline artists that never actually rendered — matplotlib's secondary
  axis gridline transform gets confused when the secondary axis's range
  is inverted (low pressure/high altitude at the top of the plot).
  Worked around by drawing the dashed lines directly on `skew.ax` via
  `axhline()`, at the pressure value each altitude tick converts to.

## Export

- `fig.savefig(..., dpi=150, bbox_inches='tight')` for a web-friendly
  PNG (2x pixel density for high-DPI screens, no dead margin).
- Matching `.svg` export for vector use.

## Housekeeping

- Dropped dead weight left over from the original MetPy example: unused
  `col_names`, the unused `get_test_data`/`add_metpy_logo` imports, and
  the `plt.rcParams['figure.figsize']` line (dead since `figsize` is
  always passed explicitly to `plt.figure()` now).
- Fixed a handful of formatting artifacts from iterative edits (stray
  trailing whitespace, a title line split across two f-strings for no
  reason, misaligned wrapped function signature).

## Historical / batch runs

With `--datetime`, the script can plot any past 00Z/12Z run, not just
the latest one. Example — the 1st of every month at 12Z for 2025:

```bash
for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
    python Simple_Sounding.py --datetime "2025-${m}-01T12"
done
```

Produces `NKX_2025<MM>01_12Z.png`/`.svg` for each month. (One run, for
September, needed a longer-than-60s timeout on the Wyoming archive
fetch on retry — a transient slow response, not missing data.)

## Not done

- Looked into Meteoblue support in `siphon` — it isn't there (`siphon`
  only wraps ACIS, IA State, IGRA2, NDBC, and Wyoming). Would need a
  direct API call, same pattern as the Open-Meteo integration.
