"""Dependency-free local UI and self-contained HTML reports using one renderer."""

import html
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

from .demo import BUNDLE_PATH, analyze

ASSETS = Path(__file__).with_name("demo_assets")
LABELS = {"M7": "Magnificent 7", "AI_HI_MOM_HI": "High AI · High momentum", "AI_HI_MOM_LO": "High AI · Low momentum",
          "AI_LO_MOM_HI": "Low AI · High momentum", "AI_LO_MOM_LO": "Low AI · Low momentum"}


def esc(value):
    return html.escape(str(value), quote=True)


def pct(value):
    return f"{value * 100:.2f}%"


def table(headers, rows, caption=""):
    return f'<div class="table-scroll"><table><caption>{esc(caption)}</caption><thead><tr>' + "".join(f"<th>{esc(header)}</th>" for header in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{esc(value)}</td>" for value in row) + "</tr>" for row in rows) + "</tbody></table></div>"


def paired_bars(rows):
    bars = []
    scale = max(1, max(max(row["weight"], row["risk_share"]) for row in rows))
    for index, row in enumerate(rows):
        y = 28 + index * 64
        label = LABELS[row["basket"]]
        bars.append(f'<text x="0" y="{y}" class="chart-label">{esc(label)}</text>')
        for offset, key, color in [(8, "weight", "#b4c6cf"), (25, "risk_share", "#007f75")]:
            value = row[key]
            width = abs(value) / scale * 440
            bars.append(f'<rect x="230" y="{y + offset - 12}" width="{width:.2f}" height="12" rx="3" fill="{color}"/><text x="{240 + width:.2f}" y="{y + offset - 2}" class="chart-value">{value:.1%}</text>')
    return '<div class="chart-scroll" tabindex="0" aria-label="Scrollable concentration chart"><svg class="chart" viewBox="0 0 760 350" role="img" aria-label="Capital weights and component volatility shares by basket">' + "".join(bars) + "</svg></div>"


def forecast_chart(experiment):
    frame = pd.DataFrame(experiment["series"])
    frame["date"] = pd.to_datetime(frame.date)
    columns = ["realized_variance", experiment["validation_selected_baseline"], "Decision tree"]
    monthly = frame.set_index("date")[columns].resample("MS").mean().pow(0.5) * np.sqrt(252) * 100
    maximum = max(10, float(monthly.max().max()) * 1.1)
    lines = []
    for name, color in zip(columns, ["#aab9c1", "#143647", "#007f75"]):
        points = " ".join(f"{55 + index / max(1, len(monthly) - 1) * 660:.1f},{230 - value / maximum * 190:.1f}" for index, value in enumerate(monthly[name]))
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
    for fraction in [0, 0.5, 1]:
        y = 230 - fraction * 190
        lines.insert(0, f'<line x1="55" x2="715" y1="{y}" y2="{y}" stroke="#e0e7e9"/><text x="4" y="{y + 4}" class="chart-value">{maximum * fraction:.0f}%</text>')
    lines.append(f'<text x="55" y="256" class="chart-value">{monthly.index[0]:%b %Y}</text><text x="645" y="256" class="chart-value">{monthly.index[-1]:%b %Y}</text>')
    return '<div class="chart-scroll" tabindex="0" aria-label="Scrollable forecast chart"><svg class="chart" viewBox="0 0 760 275" role="img" aria-label="Monthly averages of next-day variance forecasts and realized squared returns, converted to annualized volatility">' + "".join(lines) + "</svg></div>"


def render_analysis(result):
    bundle, settings = result["bundle"], result["settings"]
    selected = next(row for row in result["metrics"] if row["estimator"] == settings["estimator"])
    scenario, experiment = result["scenario"], bundle["ml"]
    max_row = max(result["decomposition"], key=lambda row: row["risk_share"])
    fragments = [f'''<section id="overview"><div class="section-heading"><div><p class="eyebrow">01 / Risk concentration</p><h2>Calm is only part of the picture.</h2></div><span class="badge">{esc(settings['estimator'])}</span></div>
    <p class="question">Where does risk exceed the capital allocated to it?</p>
    <div class="metrics"><article><span>Annualized volatility</span><strong>{pct(selected['volatility'])}</strong><small>Selected covariance estimator</small></article>
    <article><span>10-day 99% VaR</span><strong>{pct(selected['var_loss'])}</strong><small>Normal loss approximation</small></article>
    <article><span>10-day 99% expected shortfall</span><strong>{pct(selected['es_loss'])}</strong><small>Normal tail-mean loss approximation</small></article></div>
    <div class="card"><div class="card-title"><h3>Capital allocation vs. risk contribution</h3><div class="legend"><span><i class="capital"></i>Capital</span><span><i></i>Risk</span></div></div>
    {paired_bars(result['decomposition'])}<p class="insight">{esc(LABELS[max_row['basket']])} holds {pct(max_row['weight'])} of capital and contributes {pct(max_row['risk_share'])} of volatility. Compare the estimator rows before interpreting the current risk level.</p></div>
    <div class="card"><h3>One portfolio, three estimates</h3>''']
    fragments.append(table(["Covariance estimator", "Annual volatility", "10-day 99% VaR loss", "10-day 99% ES loss"],
                           [[row["estimator"], pct(row["volatility"]), pct(row["var_loss"]), pct(row["es_loss"])] for row in result["metrics"]]))
    fragments.append(f'<p class="fine">Basket log-return approximation · zero mean · 252 trading days/year · 10-day horizon · 99% confidence. Risk sample: {esc(bundle["risk_sample"][0])} to {esc(bundle["risk_sample"][1])}, {bundle["risk_sample"][2]} observations. VaR/ES are not scenario P&L.</p></div>')
    fragments.append('<div class="card"><h3>Basket drill-down</h3><p class="muted">Open a basket to inspect its current constituents. AI loading measures residual exposure to SMH; it is not a machine-learning prediction.</p>')
    for row in result["decomposition"]:
        holdings = [holding for holding in bundle["holdings"] if holding["basket"] == row["basket"]]
        original_weight = sum(holding["weight"] for holding in holdings)
        holdings.sort(key=lambda holding: holding["weight"], reverse=True)
        fragments.append(f'<details><summary>{esc(LABELS[row["basket"]])}<span>{len(holdings)} tickers · {pct(row["component_risk"])} vol contribution</span></summary>')
        fragments.append(table(["Ticker", "Company", "Sector", "Within basket", "Portfolio weight", "AI loading"], [[holding["ticker"], holding["name"], holding["sector"], pct(holding["weight"] / original_weight), pct(holding["weight"] / original_weight * row["weight"]), f'{holding["ai_beta"]:.3f}' if holding["ai_beta"] is not None else "N/A"] for holding in holdings]))
        fragments.append('</details>')
    fragments.append('</div></section>')
    fragments.append(f'''<section id="stress"><div class="section-heading"><div><p class="eyebrow">02 / Stress laboratory</p><h2>What would the portfolio lose?</h2></div><span class="badge">{esc(scenario['type'])}</span></div>
    <p class="question">Identify the exposures that dominate the loss before investigating hedges and execution.</p>
    <div class="card"><div class="stress-total"><div><span>Portfolio scenario loss</span><strong>{pct(scenario['loss_fraction'])}</strong></div><div><span>Illustrative NAV: ${settings['portfolio_value']:,.0f}</span><strong>${scenario['loss_amount']:,.0f}</strong></div></div>
    <p>{esc(scenario['description'])}</p>''')
    fragments.append(table(["Basket", "Capital weight", "Simple-return shock", "P&L / NAV", "Dollar P&L"], [[LABELS[row["basket"]], pct(row["weight"]), pct(row["shock"]), pct(row["pnl_fraction"]), f'${row["pnl_amount"]:,.0f}'] for row in scenario["contributions"]]))
    fragments.append('<p class="fine">P&L / NAV = Σ(weight × simple-return shock). A loss is shown as a positive total; contribution P&L is negative for losses. Each shock is a total basket return, not an additive market-factor coefficient.</p></div><div class="card"><h3>Historical replay availability</h3>')
    for name, replay in bundle["historical"].items():
        description = f'{replay["start"]} to {replay["end"]}. Complete endpoint prices.' if replay["available"] else replay["reason"] + ": " + ", ".join(replay.get("missing_tickers", []))
        fragments.append(f'<p><span class="status {"ok" if replay["available"] else "warn"}">{"AVAILABLE" if replay["available"] else "BLOCKED"}</span> <b>{esc(name)}</b> — {esc(description)}</p>')
    fragments.append('</div></section>')
    fragments.append(f'<section id="controls"><p class="eyebrow">03 / Limits & controls</p><h2>{len(result["exceptions"])} items require review.</h2><p class="question">Which findings deserve investigation, and what should be checked next?</p><div class="card"><p>Maximum basket risk share: <b>{pct(settings["concentration_limit"])}</b> · Maximum scenario loss: <b>{pct(settings["scenario_loss_limit"])}</b> · Stale after <b>{settings["stale_calendar_days"]:g} calendar days</b>.</p><p class="fine">Illustrative thresholds, not institutional risk appetite or regulatory compliance. Exceptions use strict greater-than comparisons.</p>')
    for exception in result["exceptions"]:
        fragments.append(f'<article class="exception"><span class="status warn">{esc(exception["status"])}</span><h3>{esc(exception["control"])}</h3><p>{esc(exception["detail"])}</p><p class="muted">Investigate: {esc(exception["action"])}</p></article>')
    if not result["exceptions"]:
        fragments.append('<p class="status ok">No configured threshold exceptions. This does not establish safety.</p>')
    fragments.append('<details><summary>Input checks and coverage</summary><ul>' + "".join(f'<li>{esc(check)}</li>' for check in result["checks_passed"]) + '</ul>')
    fragments.append(f'<p>{bundle["data_checks"]["constituents"]} constituents. Late-listing tickers: {esc(", ".join(bundle["data_checks"]["late_listing_tickers"]))}. The original basket history reweights available constituents; historical replay blocks missing endpoints.</p></details></div></section>')
    fragments.append(f'''<section id="forecast"><div class="section-heading"><div><p class="eyebrow">04 / Machine-learning experiment</p><h2>Does a simple tree add forecast value?</h2></div><span class="badge">SPY · Next-day variance</span></div>
    <p class="question">A fair comparison can support a decision to keep the baseline.</p><div class="callout">{esc(experiment['conclusion'])}</div>
    <div class="card"><h3>Common-date test performance</h3><p class="muted">{esc(experiment['test'][0])} to {esc(experiment['test'][1])} · {experiment['test'][2]} forecasts · Last realized label: {esc(experiment['last_test_label'])}</p>''')
    fragments.append(table(["Model", "QLIKE ↓", "Variance RMSE ↓", "Volatility MAE (pp) ↓", "N"], [[row["model"], f'{row["qlike"]:.4f}', f'{row["variance_rmse"]:.6f}', f'{row["volatility_mae_pp"]:.2f}', row["observations"]] for row in experiment["test_metrics"]]))
    interval = experiment["paired_block_95_interval"]
    fragments.append(f'<p class="insight">Tree minus validation-selected {esc(experiment["validation_selected_baseline"])} QLIKE: <b>{experiment["tree_minus_baseline_qlike"]:+.4f}</b>. Paired 20-day block-bootstrap 95% interval: [{interval[0]:+.4f}, {interval[1]:+.4f}]. Negative favors the tree.</p><p class="fine">QLIKE = mean(log forecast variance + realized squared return / forecast variance); it can be negative. All forecasts use decimal-return squared units. MAE uses annualized absolute daily return as a noisy volatility proxy. These are forecast errors, not investment returns.</p></div>')
    fragments.append(f'<div class="card"><h3>How forecasts respond across the test period</h3><div class="legend"><span><i class="capital"></i>Realized proxy</span><span><i class="baseline"></i>{esc(experiment["validation_selected_baseline"])}</span><span><i></i>Decision tree</span></div>{forecast_chart(experiment)}<p class="fine">Chart only: monthly mean daily variances converted to annualized volatility. Evaluation above uses every daily forecast. Grey is the realized squared-return proxy.</p></div>')
    fragments.append('<div class="card"><h3>Chronological design</h3>')
    fragments.append(table(["Stage", "Forecast dates", "Purpose"], [["Train", " → ".join(experiment["train"][:2]), "Fit six tree candidates"], ["Validation", " → ".join(experiment["validation"][:2]), "Select depth/leaf size and comparison baseline"], ["Test", " → ".join(experiment["test"][:2]), "Frozen tree; no tuning on test outcomes"]]))
    fragments.append(f'<p>Target: {esc(experiment["target"])}. Features at close t: trailing 5/21/63-day mean squared returns, 21-day downside squared returns, today’s squared return and EWMA variance (λ = 0.94).</p><p>Selected depth: {experiment["selected_settings"]["max_depth"]}; minimum leaf: {experiment["selected_settings"]["min_samples_leaf"]}. Final tree uses only labels observed through {esc(experiment["final_training_label_end"])}. Each target is the next trading day; boundary labels are purged. GARCH uses the original zero-mean normal (1,1) specification with checked convergence and refits every 21 trading days.</p>')
    fragments.append('<details><summary>Validation selection scores</summary>')
    fragments.append(table(["Depth", "Minimum leaf", "Validation QLIKE"], [[row["max_depth"], row["min_samples_leaf"], f'{row["validation_qlike"]:.4f}'] for row in experiment["candidates"]]))
    fragments.append(table(["Baseline", "Validation QLIKE"], [[row["model"], f'{row["qlike"]:.4f}'] for row in experiment["validation_metrics"]]))
    fragments.append('</details><details><summary>Frozen tree rules</summary><p class="fine">Leaf values are predicted variance × 10,000; divide by 10,000 for daily decimal variance. Features remain in decimal variance units.</p><pre>' + esc(experiment["tree_rules"]) + '</pre></details></div>')
    fragments.append('<div class="card"><h3>Performance by observable volatility regime</h3><p class="fine">High = trailing 21-day mean squared return exceeds the training sample’s 75th percentile. Regime assignment uses information available at the forecast close.</p>')
    fragments.append(table(["Regime", "Model", "QLIKE", "Volatility MAE (pp)", "N"], [[row["regime"], row["model"], f'{row["qlike"]:.4f}', f'{row["volatility_mae_pp"]:.2f}', row["observations"]] for row in experiment["regime_metrics"]]))
    fragments.append('</div><div class="card"><h3>Validation permutation importance</h3><p class="fine">Increase in QLIKE after shuffling one feature, 20 repeats. Negative values mean shuffling improved the validation score. Correlated inputs and disrupted temporal structure limit interpretation.</p>')
    fragments.append(table(["Feature", "QLIKE increase", "Shuffle standard deviation"], [[row["feature"], f'{row["qlike_increase"]:+.4f}', f'{row["shuffle_std"]:.4f}'] for row in sorted(experiment["importance"], key=lambda row: row["qlike_increase"], reverse=True)]))
    fragments.append(f'<p>Model warning/failure records: {len(experiment["failures"])}. Excluded common test dates: {experiment["excluded_test_dates"]}; excluded validation dates: {experiment["excluded_validation_dates"]}.</p>')
    if experiment["failures"]:
        fragments.append('<pre>' + esc(json.dumps(experiment["failures"], indent=2)) + '</pre>')
    fragments.append('<details><summary>Forecast limitations</summary><ul>' + "".join(f'<li>{esc(item)}</li>' for item in experiment["limitations"]) + '</ul></details></div></section>')
    fragments.append(f'<section id="provenance"><p class="eyebrow">05 / Evidence & assumptions</p><h2>A dated research demonstration.</h2><div class="card"><p>Report date: <b>{esc(result["report_date"])}</b> · Price cutoff: <b>{esc(bundle["study_end"])}</b> · Weights: <b>{esc(bundle["weights_as_of"])}</b> · Basket formation: <b>{esc(bundle["formation_date"])}</b>.</p><p>Source: {esc(bundle["provenance"]["manifest"]["price_source"])}. Pulled at {esc(bundle["provenance"]["manifest"]["pulled_at_utc"])} UTC. Inputs are frozen, not live prices.</p><ul>')
    fragments.append("".join(f'<li>{esc(item)}</li>' for item in bundle["assumptions"]))
    fragments.append('</ul><details><summary>Reproduction of existing research tables</summary>')
    fragments.append(table(["Table", "Comparison with saved CSV", "Detail"], [[row["table"], row["status"], row.get("detail", "")] for row in bundle["reproduction"]]))
    fragments.append('</details><details><summary>Settings, versions and SHA-256 fingerprints</summary><pre>' + esc(json.dumps({"demo_version": bundle["version"], "model_version": experiment["version"], "generated_at": bundle["generated_at"], "report_settings": settings, **bundle["provenance"]}, indent=2)) + '</pre></details></div></section>')
    return "".join(fragments)


def render_page(result, report=False):
    settings, bundle = result["settings"], result["bundle"]
    form = ''
    if not report:
        inputs = ''.join(f'<label class="basket-input">{esc(LABELS[name])}<span><input aria-label="{esc(name)} weight percent" name="weight" type="number" min="0" max="100" step="any" value="{weight * 100:.8f}"/>%</span></label>' for name, weight in zip(bundle["basket_names"], settings["weights"]))
        shock_inputs = ''.join(f'<label class="basket-input">{esc(LABELS[name])}<span><input aria-label="{esc(name)} shock percent" name="shock" type="number" min="-100" step="0.1" value="{shock * 100:g}"/>%</span></label>' for name, shock in zip(bundle["basket_names"], settings["shocks"]))
        options = ''.join(f'<option value="{esc(name)}" {"selected" if name == settings["estimator"] else ""}>{esc(name)}</option>' for name in bundle["covariances"])
        scenarios = ''.join(f'<option value="{esc(name)}" {"disabled" if not replay["available"] else ""}>{esc(name)}{" · blocked" if not replay["available"] else ""}</option>' for name, replay in bundle["historical"].items())
        form = f'''<aside><form id="settings"><div class="aside-title"><span class="eyebrow">Analysis settings</span><button type="button" id="reset" class="text-button">Reset</button></div>
        <label>Covariance estimator<select name="estimator">{options}</select></label>
        <label>Illustrative portfolio NAV ($)<input name="portfolio_value" type="number" min="1" step="any" value="{settings['portfolio_value']}"/></label>
        <details open><summary>Basket allocation <span>Sum must be 100%</span></summary>{inputs}</details>
        <label>Scenario type<select name="scenario"><option value="hypothetical">Hypothetical basket shocks</option>{scenarios}</select></label>
        <div id="shock-controls"><label>Hypothetical preset<select id="preset"><option value="tech">Concentrated technology selloff</option><option value="market">Broad market −10%</option><option value="custom">Custom shocks</option></select></label><p class="fine" id="scenario-rationale">Tests whether technology-linked baskets dominate losses when the rest of the market falls less.</p>{shock_inputs}</div>
        <details><summary>Illustrative control thresholds</summary><label>Max basket risk share (%)<input name="concentration_limit" type="number" min="0.01" max="100" step="any" value="{settings['concentration_limit'] * 100:g}"/></label><label>Max scenario loss (%)<input name="scenario_loss_limit" type="number" min="0.01" max="100" step="any" value="{settings['scenario_loss_limit'] * 100:g}"/></label><label>Stale after calendar days<input name="stale_calendar_days" type="number" min="1" value="{settings['stale_calendar_days']:g}"/></label></details>
        <button type="submit" class="primary">Apply & recalculate <span>↗</span></button><p id="form-status" role="status" aria-live="polite">Showing applied settings.</p><button type="button" id="download">Export applied HTML report ↓</button><p class="fine">Runs locally. Export uses the last successfully applied settings.</p></form></aside>'''
    css = (ASSETS / "style.css").read_text()
    script = '' if report else '<script>' + (ASSETS / "app.js").read_text() + '</script>'
    bootstrap = json.dumps(settings).replace('<', '\\u003c')
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>quiet-index · Risk intelligence</title><style>{css}</style></head>
    <body class="{'report' if report else 'dashboard'}"><header><a class="brand" href="#">quiet<span>—</span>index<span class="brand-dot">●</span></a><span class="header-label">RISK INTELLIGENCE / RESEARCH DEMO</span><span class="local-badge">{'EXPORTED REPORT' if report else 'LOCAL & OFFLINE'}</span></header>
    <div class="hero"><p class="eyebrow">Concentration · Stress testing · Forecast evaluation</p><h1>The portfolio looks calm.<br><em>Look beneath the surface.</em></h1><p>An interactive study of where risk sits, how losses build under stress,<br class="desktop-break"> and whether machine learning earns its place in the forecast.</p><div class="hero-meta"><span>Prices through {esc(bundle['study_end'])}</span><span>Report {esc(result['report_date'])}</span><span>Frozen research data</span></div></div>
    <nav aria-label="Analysis sections"><a href="#overview">01 Concentration</a><a href="#stress">02 Stress</a><a href="#controls">03 Controls</a><a href="#forecast">04 ML experiment</a><a href="#provenance">05 Evidence</a></nav>
    <div class="workspace">{form}<main id="analysis">{render_analysis(result)}</main></div><footer>quiet-index · Sai Reddivari · Research demonstration · No live trading or investment recommendation.</footer>
    <script type="application/json" id="applied-settings">{bootstrap}</script>{script}</body></html>'''


def serve(port=8790):
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError("Run .venv/bin/python scripts/build_demo.py first")
    bundle = json.loads(BUNDLE_PATH.read_text())

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, content, content_type="text/html; charset=utf-8", attachment=None):
            encoded = content.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if attachment:
                self.send_header("Content-Disposition", f'attachment; filename="{attachment}"')
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            if self.path == "/":
                self.respond(200, render_page(analyze(bundle)))
            elif self.path == "/health":
                self.respond(200, '{"status":"ok"}', "application/json")
            elif self.path == "/sample-report":
                reports = sorted(BUNDLE_PATH.parent.glob("demo_report_*.html"))
                if reports:
                    self.respond(200, reports[-1].read_text())
                else:
                    self.respond(404, "Build a sample report first", "text/plain")
            else:
                self.respond(404, "Not found", "text/plain")

        def do_POST(self):
            if self.path not in ["/api/analyze", "/api/report"]:
                self.respond(404, "Not found", "text/plain")
                return
            if self.headers.get("Origin") not in [None, f"http://127.0.0.1:{port}", f"http://localhost:{port}"]:
                self.respond(403, "Local requests only", "text/plain")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 16000:
                    raise ValueError("Invalid request size")
                supplied = json.loads(self.rfile.read(length))
                if not isinstance(supplied, dict):
                    raise ValueError("Expected a settings object")
                result = analyze(bundle, supplied)
                if self.path == "/api/report":
                    self.respond(200, render_page(result, report=True), attachment=f'quiet-index-risk-report-{result["report_date"]}.html')
                else:
                    self.respond(200, json.dumps({"html": render_analysis(result), "settings": result["settings"], "exceptions": len(result["exceptions"])}), "application/json")
            except (ValueError, KeyError, TypeError, OverflowError) as error:
                self.respond(400, json.dumps({"error": str(error)}), "application/json")

    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"quiet-index dashboard: http://127.0.0.1:{port} (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
