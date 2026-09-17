#!/usr/bin/env bash
dates=(
    "2020-05-20"
    "2023-09-23"
    "2021-03-16"
    "2021-01-21"
    "2026-04-11"
    "2026-05-27"
    "2026-04-13"
    "2025-05-13"
    "2020-03-08"
    "2025-05-26"
)

for d in "${dates[@]}"; do
    python Simple_Sounding.py --datetime "${d}T12"
done
