"""Reproduce study tables in memory, run ML evaluation, and export demo artifacts."""

import argparse
import json

import pandas as pd

from quiet_index import config as cfg
from quiet_index.demo import BUNDLE_PATH, analyze, build_bundle
from quiet_index.demo_web import render_page


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-inputs", action="store_true", help="export report from existing demo_inputs.json without retraining")
    parser.add_argument("--report-date", help="YYYY-MM-DD; defaults to today's America/Chicago date")
    args = parser.parse_args()
    bundle = json.loads(BUNDLE_PATH.read_text()) if args.reuse_inputs else build_bundle()
    result = analyze(bundle, report_date=args.report_date)
    report = cfg.RESULTS / f"demo_report_{result['report_date']}.html"
    report.write_text(render_page(result, report=True))
    pd.DataFrame(bundle["ml"]["series"]).to_csv(cfg.RESULTS / "demo_forecasts.csv", index=False)
    pd.DataFrame(bundle["ml"]["test_metrics"]).to_csv(cfg.RESULTS / "demo_forecast_metrics.csv", index=False)
    (cfg.RESULTS / "demo_verification.json").write_text(json.dumps({"research_tables": bundle["reproduction"], "ml_conclusion": bundle["ml"]["conclusion"], "model_failures": bundle["ml"]["failures"], "excluded_test_dates": bundle["ml"]["excluded_test_dates"]}, indent=2) + "\n")
    print(f"Report: {report}")
    print(bundle["ml"]["conclusion"])
    print(pd.DataFrame(bundle["ml"]["test_metrics"]).to_string(index=False))
    differences = [row["table"] for row in bundle["reproduction"] if row["status"] != "reproduced"]
    print(f"Research tables compared: {len(bundle['reproduction'])}; differences: {differences}")


if __name__ == "__main__":
    main()
