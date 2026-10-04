"""Download raw data into data/raw (not committed).

Usage:
    python scripts/download_data.py                  # prices only
    python scripts/download_data.py --refresh-weights  # also re-snapshot iShares weights
"""

import argparse

from quiet_index import config as cfg
from quiet_index import data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh-weights", action="store_true",
                        help="download the current iShares holdings file and save a "
                             "new dated snapshot (update config.WEIGHTS_FILE to use it)")
    args = parser.parse_args()

    if args.refresh_weights:
        equity, as_of = data.fetch_holdings()
        snapshot = data.save_weights_snapshot(equity, as_of)
        print(f"weights snapshot: {len(snapshot)} tickers as of {as_of}")

    manifest = data.download_all()
    for name, info in manifest["series"].items():
        print(f"{name:11s} {info['tickers']:4d} tickers  "
              f"{info['first_date']} -> {info['last_date_pulled']}")
    print(f"study end pinned at {cfg.STUDY_END}; manifest written to {data.MANIFEST_FILE}")


if __name__ == "__main__":
    main()
