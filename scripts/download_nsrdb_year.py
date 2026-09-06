"""Download the fixed NSRDB solar file specified by a FairFlex study config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fairflex.data import NSRDBDataClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/caltech_2019_pilot.json")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    solar = config["solar_data"]
    output = Path("data/raw") / f"nsrdb_{config['study_id']}_{solar['year']}.csv"
    NSRDBDataClient().download_csv(
        output,
        dataset=solar["dataset"],
        year=solar["year"],
        interval_minutes=solar["interval_minutes"],
        latitude=solar["latitude"],
        longitude=solar["longitude"],
    )
    print(f"Saved NSRDB CSV to {output}")


if __name__ == "__main__":
    main()
