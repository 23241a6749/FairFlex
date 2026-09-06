"""Validate raw study inputs and create a credential-free provenance manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fairflex.scenarios import (
    build_data_manifest,
    load_pv_proxy,
    load_study_config,
    prepare_acn_split,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/caltech_2019_pilot.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/caltech_2019_pilot/data_manifest.json"),
        help="credential-free JSON provenance output",
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_study_config(config_path)
    prepared = {
        split_name: prepare_acn_split(config, config_path, split_name)
        for split_name in config["splits"]
    }
    proxy_settings = config["pv_proxy_sensitivity"]
    pv_proxy = load_pv_proxy(
        config,
        config_path,
        dc_capacity_kw=float(proxy_settings["dc_capacity_kw"]),
        ac_capacity_kw=float(proxy_settings["ac_capacity_kw"]),
    )
    manifest = build_data_manifest(config, config_path, prepared)
    manifest["pv_proxy_sensitivity"] = {
        **proxy_settings,
        "points": len(pv_proxy),
        "start": pv_proxy.index.min().isoformat(),
        "end": pv_proxy.index.max().isoformat(),
        "max_kw": float(pv_proxy.max()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Prepared study: {config['study_id']}")
    for name, split in prepared.items():
        print(
            f"{name}: {len(split.sessions)} usable / {split.source_records} raw sessions; "
            f"requested={split.requested_energy_kwh:.1f} kWh; "
            f"stations={dict(split.station_session_counts)}"
        )
    print(f"PV proxy: {len(pv_proxy)} points, maximum {pv_proxy.max():.2f} kW")
    print(f"Manifest: {args.output}")


if __name__ == "__main__":
    main()
