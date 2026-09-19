#!/usr/bin/env bash
# All sites.tsv sites at one valid time: 2026-09-19 19Z, noon UTC-7.
#
# --datetime asks for a *valid* time and lets the fetch resolve which run
# and lead reach it -- normally the freshest run posted. Where that run
# doesn't go out far enough (these models only run to a long lead on
# their 00/06/12/18Z cycles), it falls back to older runs at a longer
# lead until one covers this time, so the plot may come from an earlier
# run than the newest available. The "Model run:" label says which.
#
# No --model on purpose: pinning the chain to one source means a valid
# time no run of that model can reach fails outright, instead of falling
# back to HRRR and then GFS.
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
    "Skyport|34.4809515|-119.6846529"
)

for entry in "${sites[@]}"; do
    IFS='|' read -r name lat lon <<< "$entry"
    python Simple_Sounding.py --lat "$lat" --lon "$lon" \
        --datetime "2026-09-19T19" --name "$name"
done
