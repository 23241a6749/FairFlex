"""Verify local ACN and NSRDB credentials without printing either secret."""

from fairflex.data import ACNDataClient, NSRDBDataClient


def main() -> None:
    acn_records_in_page = ACNDataClient().ping("caltech")
    # Hyderabad is only a connectivity check. Select the actual study location
    # and year before downloading any NSRDB dataset for an experiment.
    nsrdb_response = NSRDBDataClient().discover(17.3850, 78.4867)
    print(f"ACN authentication: OK (first page has {acn_records_in_page} records)")
    print(f"NSRDB authentication: OK ({len(nsrdb_response['outputs'])} nearby datasets found)")


if __name__ == "__main__":
    main()
