"""List NSRDB datasets available near an intended study location."""

from __future__ import annotations

import argparse

from fairflex.data import NSRDBDataClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    args = parser.parse_args()

    response = NSRDBDataClient().discover(args.lat, args.lon)
    for dataset in response["outputs"]:
        years = dataset.get("availableYears", [])
        intervals = dataset.get("availableIntervals", [])
        print(
            f"{dataset.get('name', 'unknown')}: "
            f"years={years[:5]}{'...' if len(years) > 5 else ''}, "
            f"intervals_minutes={intervals}"
        )


if __name__ == "__main__":
    main()
