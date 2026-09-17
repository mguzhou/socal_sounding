"""Turning a station code or a lat/lon into a human place name for
the plot title."""

import requests
import reverse_geocoder


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
