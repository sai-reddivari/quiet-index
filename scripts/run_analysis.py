"""Run Sections 1 to 4 from the cached data and write every table and figure.

Usage:
    python scripts/run_analysis.py              # tables to results/, figures to figures/
    python scripts/run_analysis.py --no-figures # tables only
"""

import argparse

from quiet_index import config as cfg
from quiet_index import pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-figures", action="store_true", help="skip the figures and only write the tables")
    args = parser.parse_args()

    results = pipeline.run_all()
    pipeline.save_tables(results)
    print(f"tables written to {cfg.RESULTS}")

    if not args.no_figures:
        from quiet_index import plots   #imported here so the tables can be produced without a display
        plots.save_all(results)
        print(f"figures written to {cfg.FIGURES}")

    index_risk = results["section_2"]["tables"]["s2_index_risk"]
    print("\nindex volatility by estimator")
    print(index_risk["index_volatility"].round(4).to_string())


if __name__ == "__main__":
    main()
