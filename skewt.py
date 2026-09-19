"""The main Skew-T panel: the sounding traces, wind barbs, clipped
dry adiabats, lifted-parcel analysis, and the MSL altitude axis."""

import matplotlib
import metpy.calc as mpcalc
import numpy as np
from metpy.plots import SkewT
from metpy.units import units
from scipy.ndimage import median_filter

from config import DEFAULT_ALTITUDE_UNIT, KM_TO_KFT, utc_offset_label
# The skew-T panel marks the surface inversion the lapse-rate panel
# detects, so the detection itself lives there and is shared from there.
from lapse_rate import find_surface_inversion_top

# Depth over which the condensation levels take their moisture, as the
# layer's mean mixing ratio rather than the single surface value.
#
# Real cumulus comes from a population of thermals with differing surface
# moisture: the moistest condense, the rest don't, which is what partial
# cover actually is. A lone 2 m dewpoint is one sample of that population
# masquerading as the whole of it, and it is also the noisiest point in
# the profile. Averaging over the mixed layer is the conventional fix
# (the "mixed-layer parcel"), and it makes cloud base representative of
# the thermals rather than of one grid cell's surface.
#
# 50 hPa, not the textbook 100 hPa: these are shallow, often
# high-elevation boundary layers, and 100 hPa would average in free
# atmosphere well above the layer thermals actually mix.
MIXED_LAYER_DEPTH = units.Quantity(50, 'hPa')


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


def render_skewt_panel(fig, subplot, sounding_df, sounding_date, tz,
                       forecast_high=None, is_forecast=True, location_desc=None,
                       title_prefix=None, run_label=None, is_modeled=False, site_name=None,
                       station=None, altitude_unit=DEFAULT_ALTITUDE_UNIT):
    """Draw one full Skew-T panel (data, barbs, adiabats, altitude axis,
    title) into the given subplot position of fig.

    tz is a tzinfo (the site's own zone) that sounding_date is converted
    through for the title, rather than a fixed offset -- so a sounding
    from the other side of a DST transition is labelled with the offset
    that was actually in force on its date.

    station is only used to build the default title when title_prefix
    isn't given; altitude_unit picks the MSL axis' unit ('km' or 'kft')."""
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
    skew.ax.set_xlabel('Temperature (\N{DEGREE SIGN}C)')

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
    pressure_to_alt, alt_to_pressure_mb = make_msl_height_functions(sounding_df, altitude_unit)
    top_alt = 6.0 * KM_TO_KFT if altitude_unit == 'kft' else 6.0
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
    bottom_margin = 0.02 * KM_TO_KFT if altitude_unit == 'kft' else 0.02
    bottom_pressure = alt_to_pressure_mb(-bottom_margin)[0]
    skew.ax.set_ylim(bottom_pressure, top_pressure)
    skew.ax.set_xlim(-5, 50)

    # The lifted-parcel annotations (forecast-high dry adiabat for
    # observed soundings, or the most-unstable/surface parcel for
    # modeled ones) are only drawn for the current sounding's panel, not
    # a previous run's overlay -- forecast_high is None there. For an
    # observed sounding with a value, that's Open-Meteo's forecast high
    # (recent date) or that day's actual observed high (historical
    # date, from Open-Meteo's archive instead); is_modeled bypasses
    # forecast_high entirely regardless of whether one was given.
    # parcel_p_path/parcel_profile/parcel_T_path (pressure, parcel temp,
    # and matching environmental temp over that same pressure range) are
    # handed back to the caller so the lapse-rate panel can plot the
    # parcel-minus-environment delta alongside its lapse rate curve.
    parcel_p_path = parcel_profile = parcel_T_path = None
    if is_modeled or forecast_high is not None:
        # base_temp is the parcel's starting (surface) temperature to
        # lift: the model's own reported surface value for a modeled
        # sounding -- reflecting actual surface-based convection, not
        # an externally-forecast high (the model profile already *is* a
        # forecast; independently forecasting the surface high on top
        # of it would stack two forecasts' worth of uncertainty) -- or
        # Open-Meteo's forecast/observed high for an observed sounding.
        if is_modeled:
            base_temp = T[0].m
            parcel_label = 'Surface'
            parcel_suffix = ''
        else:
            base_temp = forecast_high
            parcel_label = 'Forecast high' if is_forecast else 'Observed high'
            parcel_suffix = ' - OpenMeteo'

        forecast_uncertainty = 1.0
        # Moisture for every condensation calculation below: the mixed
        # layer's mean mixing ratio, expressed as the dewpoint a surface
        # parcel would carry (see MIXED_LAYER_DEPTH). Only the moisture is
        # taken from the mix -- the parcel is still heated to base_temp,
        # the model's own surface temperature, since that is what drives
        # the thermal.
        _, _, mixed_dewpoint = mpcalc.mixed_parcel(p, T, Td, depth=MIXED_LAYER_DEPTH)

        # Full parcel ascent (dry below the LCL, moist above) starting from
        # the surface -- shows any CAPE a parcel heated to base_temp would
        # actually have.
        forecast_profile = mpcalc.parcel_profile(
            p, units.Quantity(base_temp, 'degC'), mixed_dewpoint).to('degC')
        forecast_profile_minus = mpcalc.parcel_profile(
            p, units.Quantity(base_temp - forecast_uncertainty, 'degC'), mixed_dewpoint).to('degC')
        forecast_profile_plus = mpcalc.parcel_profile(
            p, units.Quantity(base_temp + forecast_uncertainty, 'degC'), mixed_dewpoint).to('degC')
        skew.plot(p, forecast_profile, color='black', linewidth=1.2, linestyle='solid',
                 label=f'{parcel_label} parcel ({base_temp:.1f} \N{PLUS-MINUS SIGN} '
                       f'{forecast_uncertainty:.1f}) \N{DEGREE SIGN}C{parcel_suffix}')

        # Lifted condensation level for that same parcel -- marks where
        # it would saturate, i.e. the base of any clouds that heating to
        # base_temp would produce.
        lcl_pressure, lcl_temperature = mpcalc.lcl(
            p[0], units.Quantity(base_temp, 'degC'), mixed_dewpoint)

        parcel_p_path = p
        parcel_T_path = T
        parcel_profile = forecast_profile

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
        # A dry enough profile simply has no CCL: the surface mixing-ratio
        # line never meets the temperature curve within the data. MetPy
        # indexes its (empty) intersection array unconditionally and so
        # raises IndexError rather than returning nan, which would take
        # the whole plot down over an annotation that legitimately doesn't
        # exist here -- so treat it as "no CCL" and carry on without it.
        try:
            ccl_pressure, ccl_temperature, convective_temp = mpcalc.ccl(
                p, T_for_ccl, Td_for_ccl,
                mixed_layer_depth=MIXED_LAYER_DEPTH, which='bottom')
        except IndexError:
            ccl_pressure = ccl_temperature = convective_temp = None

        if ccl_pressure is not None:
            skew.plot(ccl_pressure, ccl_temperature, marker='^', color='black',
                     markerfacecolor='none', markersize=9, linestyle='none',
                     label='Convective condensation level')
            conv_path_pressure = units.Quantity(np.linspace(p[0].m, ccl_pressure.m, 50), 'hPa')
            conv_path_temp = mpcalc.dry_lapse(conv_path_pressure, convective_temp).to('degC')
            skew.plot(conv_path_pressure, conv_path_temp, color='darkorange', linewidth=1.2,
                     linestyle='solid', alpha=0.8,
                     label=f'Convective temperature ({convective_temp.m:.1f}\N{DEGREE SIGN}C)')

        # The mixing line (constant mixing ratio) from the surface dewpoint
        # up to the higher of the LCL/CCL -- the classic graphical
        # technique for finding both is where this line (constant
        # moisture) crosses the relevant temperature curve: the lifted
        # parcel's dry adiabat for the LCL, the environmental profile
        # itself for the CCL. Both land on this same line since both are
        # built from the same starting (surface) dewpoint.
        surface_mixing_ratio = mpcalc.saturation_mixing_ratio(p[0], mixed_dewpoint)
        # Up to the LCL alone when there's no CCL to be the higher of the two.
        mixing_line_top = (lcl_pressure.m if ccl_pressure is None
                           else min(lcl_pressure.m, ccl_pressure.m))
        skew.plot_mixing_lines(
            mixing_ratio=np.atleast_1d(surface_mixing_ratio.m),
            pressure=units.Quantity(np.linspace(p[0].m, mixing_line_top, 50), 'hPa'),
            colors='black', linestyles='dashed', linewidths=1, alpha=0.6)
        
        # Meters specifically (not altitude_unit), via a fresh unscaled
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
        def capped_mixing_top(env_p, env_T, profile, lcl_top_pressure):
            cross_pressure, _ = mpcalc.find_intersections(
                env_p, profile, env_T, direction='decreasing', log_x=True)
            top = cross_pressure[0].m if len(cross_pressure) > 0 else lcl_top_pressure
            top = max(top, lcl_top_pressure)
            #if surface_inversion_top is not None:
            #    top = max(top, surface_inversion_top)

            # When find_intersections finds no crossing at all (common
            # for the -1 degC uncertainty bound in particular), the
            # fallback above reports the LCL itself -- but reaching the
            # LCL doesn't mean the parcel was ever actually buoyant
            # enough to get there under its own power; it can stay
            # colder than the environment the entire way and still
            # "reach" its LCL on paper. Checking the parcel against the
            # environment at the reported top catches that: if the
            # parcel isn't at or above the environment there, this
            # isn't a real thermal top/cloud base, so report none at
            # all rather than a point free convection never reaches.
            # A genuine crossing from find_intersections has parcel_T ==
            # env_T there by definition, but re-interpolating both with a
            # plain np.interp here (a different, less precise method than
            # find_intersections' own root-finding) can land a hair to
            # either side of equal just from that method mismatch -- an
            # exact "<" would then reject some genuine crossings on
            # nothing but interpolation noise. A small absolute tolerance
            # absorbs that noise while still catching the real case (a
            # fallback-to-LCL point where the parcel stayed meaningfully
            # colder than the environment the whole way, e.g. by several
            # degrees, not thousandths of one).
            parcel_T_at_top = np.interp(top, env_p.m[::-1], profile.m[::-1])
            env_T_at_top = np.interp(top, env_p.m[::-1], env_T.m[::-1])
            if parcel_T_at_top < env_T_at_top - 0.05:
                return None, None

            # LCL was the binding constraint (parcel saturates before it
            # runs out of buoyancy) -- this is a cloud base, not a dry
            # thermal top.
            is_cloud_base = np.isclose(top, lcl_top_pressure)
            return top, is_cloud_base

        # Where the *central* parcel (no uncertainty offset, for the
        # observed-sounding forecast-high case) first drops back below
        # the environmental temperature -- a single-line best estimate
        # of the mixing top.
        mid_top, mid_is_cloud_base = capped_mixing_top(
            parcel_p_path, parcel_T_path, parcel_profile, lcl_pressure.m)
        if mid_top is not None:
            mixing_height_m = pressure_to_height_km(mid_top)[0] * 1000
            mid_label = 'Cloud base' if mid_is_cloud_base else 'Thermal tops'
            skew.ax.axhline(mid_top, color='indigo', linestyle='dotted',
                            linewidth=1, alpha=0.6,
                            label=f'{mid_label} ({parcel_label.lower()}, {mixing_height_m:,.0f} m)')

        # Where each bound of the ±uncertainty range first drops back
        # below the environmental temperature -- the top of the layer
        # that parcel would actively mix through. This is the *first*
        # (lowest) parcel/environment crossing, not the *last* one
        # mpcalc.el() finds (the true equilibrium level, generally much
        # higher up on a convective sounding) -- see the earlier
        # discussion on why el() doesn't fit this use.
        #
        # Starts with mid_top (the central estimate) already in the list:
        # near a sharp threshold (e.g. a capping inversion just above the
        # surface), the response to the +/-1 degC offset can be sharply
        # nonlinear -- the central case might find its crossing right at
        # the very first level (~no mixing) while +1 degC alone punches
        # through the cap to a much higher crossing. Without mid_top
        # included, a shaded band built from just the two bounds can end
        # up sitting entirely above (or below) the central dotted line
        # instead of bracketing it, which reads as the band being offset
        # from its own point estimate rather than what it actually is: a
        # real, sharp sensitivity to a nearby threshold.
        #
        # A bound can also come back as None (see capped_mixing_top) --
        # most often -1 degC, which frequently never gets warmer than the
        # environment at all and would otherwise fall back to reporting
        # its own LCL as if it were a reachable cloud base. Skipped
        # entirely (no line, not counted in the shaded span) rather than
        # plotted as a point free convection never actually reaches.
        mixing_top_pressures = [mid_top] if mid_top is not None else []
        for profile, bound_temp in ((forecast_profile_plus, base_temp + forecast_uncertainty),
                                    (forecast_profile_minus, base_temp - forecast_uncertainty)):
            bound_lcl_pressure, _ = mpcalc.lcl(p[0], units.Quantity(bound_temp, 'degC'),
                                               mixed_dewpoint)
            top, _ = capped_mixing_top(p, T, profile, bound_lcl_pressure.m)
            if top is None:
                continue
            mixing_top_pressures.append(top)
            skew.ax.axhline(top, color='steelblue', linestyle='dotted',
                            linewidth=1, alpha=0.7)

        # Shade the pressure band spanning all three (bounds + central) --
        # the range of possible mixing heights given the ±uncertainty,
        # guaranteed to include the central estimate. Labeled with the
        # actual height span, since the same temperature spread can mean
        # very different height spreads depending on how steep the local
        # lapse rate is there.
        if len(mixing_top_pressures) >= 2:
            mixing_heights_m = sorted(pressure_to_height_km(pr)[0] * 1000
                                      for pr in mixing_top_pressures)
            mixing_height_range_m = mixing_heights_m[-1] - mixing_heights_m[0]
            skew.ax.axhspan(min(mixing_top_pressures), max(mixing_top_pressures),
                            color='steelblue', alpha=0.12,
                            label=f'Thermal top uncertainty ({mixing_height_range_m:,.0f} m)')

        # The cloud layer -- labelled by depth rather than by cloud type,
        # since the same calculation covers everything from a shallow fair
        # weather cumulus to a 5.8 km cumulonimbus (SLC, verified), and
        # naming one of those would be wrong for the other.
        #
        # Cloud base is the LCL, but only where the
        # parcel actually got there under its own buoyancy -- which is
        # exactly what capped_mixing_top reports via mid_is_cloud_base
        # (it caps the thermal top at the LCL and says so when it did).
        # Where thermals top out below the LCL instead, the day is blue
        # and nothing is drawn, which keeps a blue day visibly blue.
        #
        # Cloud top is the equilibrium level -- where the saturated
        # parcel, now following the moist adiabat, finally loses its
        # buoyancy. Note this is the one place el() is the right tool:
        # it is deliberately avoided for the *thermal* top above (see
        # capped_mixing_top, which wants the first crossing, not the
        # last), but the last crossing is precisely what caps a cloud.
        if mid_is_cloud_base and mid_top is not None:
            el_pressure, _ = mpcalc.el(p, T, Td, forecast_profile)
            # No EL doesn't mean no cloud top -- it usually means the top
            # is above the data. These profiles stop at 400 mb (see
            # MODEL_PRESSURE_LEVELS) while a convective EL commonly sits
            # nearer 200-300 mb, so a parcel still buoyant at the ceiling
            # never crosses back and el() returns nan. Measured on a test
            # profile with 826 J/kg of CAPE. Fall back to the top of the
            # data and mark the depth as a lower bound, rather than
            # drawing no cloud at all on the very days that have one.
            open_topped = el_pressure is None or not np.isfinite(el_pressure.m)
            cloud_top = p.m.min() if open_topped else el_pressure.m
            if cloud_top < mid_top:
                cloud_depth_m = (pressure_to_height_km(cloud_top)[0]
                                 - pressure_to_height_km(mid_top)[0]) * 1000
                depth_text = (f'\N{GREATER-THAN OR EQUAL TO}{cloud_depth_m:,.0f} m deep'
                              if open_topped else f'{cloud_depth_m:,.0f} m deep')
                # Drawn as a narrow column in the left margin rather than
                # a full-width band: a deep cloud spans most of the plot
                # (SLC verified at 5,764 m, base 2,652 m against a 6 km
                # axis cap), and shading all of that washes out the
                # traces, barbs and the thermal-top band underneath it.
                # A margin column shows base and top just as precisely
                # without covering the data.
                #
                # Blended transform: x in axes fractions so the column
                # keeps its width and stays vertical despite the skewed
                # x-axis, y in data coordinates so it tracks pressure
                # correctly on the log scale. fill_betweenx rather than a
                # Rectangle for that same reason -- a Rectangle's height
                # is linear and would misplace the top.
                cloud_trans = matplotlib.transforms.blended_transform_factory(
                    skew.ax.transAxes, skew.ax.transData)
                skew.ax.fill_betweenx(np.linspace(mid_top, cloud_top, 50), 0.015, 0.075,
                                      transform=cloud_trans, color='lightblue', alpha=0.85,
                                      edgecolor='steelblue', linewidth=0.6, zorder=2.5,
                                      label=f'Cloud depth ({depth_text})')

        skew.shade_cape(p, T, forecast_profile_plus, alpha=0.1)
        skew.shade_cin(p, T, forecast_profile_minus, alpha=0.05)
        skew.plot(p, forecast_profile_minus, color='steelblue', linewidth=.5, linestyle='dashed', alpha=.8)
        skew.plot(p, forecast_profile_plus, color='steelblue', linewidth=.5, linestyle='dashed', alpha=.8)
        skew.ax.fill_betweenx(p, forecast_profile_minus, forecast_profile_plus,
                              color='steelblue', alpha=0.15)

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
    scale = KM_TO_KFT if altitude_unit == 'kft' else 1.0
    bottom_level_alt = sounding_df.loc[sounding_df['pressure'].idxmax(), 'height'] / 1000. * scale

    alt_ax = skew.ax.secondary_yaxis(0, functions=(pressure_to_alt, alt_to_pressure_mb))
    alt_ax.set_ylabel(f'MSL Altitude ({altitude_unit})', labelpad=0)
    alt_tick_step = 3 if altitude_unit == 'kft' else 1
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
    # site_name (a caller-supplied arbitrary label, e.g. from the user's
    # own named list of sites) and location_desc (geocoded) each add a
    # line under the station name, site_name first -- the right side is
    # padded with a matching number of blank lines so both stay stacked
    # row for row instead of the left title's extra lines (which can run
    # long, e.g. a full airport name) colliding with the right title's
    # single line sharing that row.
    local_dt = sounding_date.astimezone(tz)
    title_left = title_prefix if title_prefix is not None else f'{station} Observed Sounding'
    title_right = f'{local_dt:%Y-%m-%d %H:%M} {local_dt:%Z} ({utc_offset_label(local_dt)})'
    extra_left_lines = [line for line in (site_name, location_desc) if line]
    if extra_left_lines:
        title_left += '\n' + '\n'.join(extra_left_lines)
        title_right += '\n' * len(extra_left_lines)
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

    if run_label:
        # Below both axes (negative axes-fraction y clears the x-axis
        # tick labels and "Temperature (degC)" label beneath it), not
        # inside the plotted data area.
        skew.ax.text(-0.2, -0.08, run_label, transform=skew.ax.transAxes,
                     fontsize=9, color='dimgray', va='top', ha='left')

    return skew, parcel_p_path, parcel_profile, parcel_T_path
