"""Download one bounded ACN session window to ignored raw-data storage."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from email.utils import format_datetime, parsedate_to_datetime
from pathlib import Path

from fairflex.data import ACNDataClient


def _parse_utc(value: str) -> datetime:
    """Parse ISO-8601 or RFC-1123 input and normalize it to UTC."""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        parsed = parsedate_to_datetime(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _date_label(value: str) -> str:
    """Return a stable YYYYMMDD filename label for ISO or RFC-1123 input."""
    return _parse_utc(value).strftime("%Y%m%d")


def _acn_timestamp(value: str) -> str:
    """Format a user-supplied time as ACN's RFC-1123/GMT filter syntax."""
    return format_datetime(_parse_utc(value), usegmt=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", default="caltech", choices=("caltech", "jpl", "office001"))
    parser.add_argument(
        "--start",
        required=True,
        help='UTC time; ISO 8601 (recommended), e.g. "2019-05-01T00:00:00Z", or RFC 1123',
    )
    parser.add_argument(
        "--end",
        required=True,
        help='UTC time; ISO 8601 (recommended), e.g. "2019-05-08T00:00:00Z", or RFC 1123',
    )
    parser.add_argument("--max-pages", type=int, default=100)
    parser.add_argument("--output", type=Path, help="optional explicit JSON output path")
    args = parser.parse_args()
    if args.max_pages <= 0:
        parser.error("--max-pages must be positive")

    # Match the replay protocol exactly: connection time in [start, end).
    # An inclusive endpoint can silently duplicate a midnight session when two
    # neighbouring acquisition windows are combined.
    where = f'connectionTime>="{_acn_timestamp(args.start)}" and connectionTime<"{_acn_timestamp(args.end)}"'
    records = ACNDataClient().fetch_sessions(args.site, where=where, max_pages=args.max_pages)
    if args.output:
        output = args.output
    else:
        start_label = _date_label(args.start)
        end_label = _date_label(args.end)
        output = Path("data/raw") / f"acn_{args.site}_{start_label}_{end_label}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Saved {len(records)} ACN sessions to {output}")


if __name__ == "__main__":
    main()
