# CLAUDE.md

Quantitative research project: risk decomposition and estimator sensitivity of
the S&P 100. **TASK_PAPER.md is the single reference** for scope, definitions,
and tasks. README.md is the public summary.

## Repo structure — deliberately minimal
All work currently lives in `notebooks/01_data_and_sleeves.ipynb`. Earlier
scaffolding (notebooks 02–05, the `src/quiet_index` package, PROJECT_SPEC.md,
test stubs) was deliberately removed by Sai. **Never recreate deleted files or
add new files, folders, or structure unless Sai explicitly asks.** He will
promote notebook code into .py modules himself, with guidance, when he chooses.
Verification anchor values live in TASK_PAPER.md section 5.

## How to behave here
- Sai drives. Answer what he asks; don't set the pace, propose next steps
  unprompted, run tutorials, or split work into confirm-each-step sequences.
- Analysis code is his — factor estimation, covariance, decomposition,
  optimization, backtests, interpretation. Don't write it unless he explicitly
  hands over a specific piece. Reviewing, debugging, and explaining are welcome.
- Plumbing (downloads, dataframe wrangling, plotting boilerplate, git, venv) may
  be written outright on request.
- Never delete files; if something should go, list it for Sai to remove.
- Any code block given to him must be labeled RUNS AS-IS or NEEDS: <what>.

## Hard rules
1. `data/raw/` is never committed.
2. Every reported number's data source and pull date is noted.
3. Results are recorded in README.md only after independent reproduction and
   review in Sai's calibration chat.
