#!/usr/bin/env bash
# All sites.tsv sites, pinned to the exact run we've been using: RRFS's
# 2026-09-16 12Z cycle, 31h forecast lead (valid 2026-09-17 19Z = noon
# GMT-7 the next day). --run-datetime/--forecast-hour pin the run/lead
# directly rather than resolving them from a valid time, so this stays
# reproducible even once that run has aged out of RRFS's own rolling
# archive from "latest available" resolution.
sites=(
    "Little Black|32.987952027309035|-117.12235348316403"
    "Torrey Pines|32.890234820022194|-117.25180563384053"
    "Blossom Valley|32.87405141739149|-116.85019889350394"
    "Horse Canyon|32.77464791023273|-116.47664103527119"
    "Fuzz|32.75859746410207|-116.50674592160713"
    "Laguna|32.93886162815183|-116.48420521874074"
    "Backshots|33.31570423772878|-116.73363284343071"
    "Big Black|33.15912127429388|-116.8083724651809"
    "Elsinore|33.628268786158195|-117.37011729863373"
    "Soboba|33.82070400719413|-116.95941627466937"
    "Whales|32.6402909855531|-116.8865630082742"
    "Palomar|33.335235134845746|-116.9437728988871"
    "Marshall|34.210388000507166|-117.30301680380092"
    "Blackhawk|34.33878978482683|-116.81326131187706"
    "Ord|34.403990826175985|-117.17860269023457"
    "Kagel|34.33353413016472|-118.38471551293345"
)

for entry in "${sites[@]}"; do
    IFS='|' read -r name lat lon <<< "$entry"
    python Simple_Sounding.py --lat "$lat" --lon "$lon" --model rrfs \
        --datetime "2026-09-17T19" --name "$name"
done
