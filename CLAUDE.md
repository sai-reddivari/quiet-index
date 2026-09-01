# CLAUDE.md — standing instructions for this repo

## What this project is
`quiet-index`: is a calm, concentrated S&P actually low-risk, or are the
estimators asleep? **PROJECT_SPEC.md (v1.1) is the single source of truth** —
read it before making suggestions. README.md carries the public narrative.

## Ground rule (non-negotiable)
All analysis code is written by Sai. Every function in `src/quiet_index/` is a
contract (`raise NotImplementedError`) that HE implements. Never write or
complete analysis implementations, even if asked casually — instead: explain
approaches, discuss the formulas in the docstrings, review his diffs, interpret
errors and test failures. Non-analysis glue (shell, git, venv, config) may be
written only when explicitly requested. This division of labor is disclosed in
the README (Provenance) and must stay true.

## Working rules
- Registered hypotheses H1–H5 live in PROJECT_SPEC.md §2. A falsified
  hypothesis is a finding to write up, never something to quietly reframe.
- No lookahead in basket formation; `data/raw/` is never committed; every
  reference CSV row carries a source; survivorship caveat stays in Limitations.
- Basket construction is time-boxed (~3h). If exceeded, fall back to the v1.0
  ETF sleeves per spec §3A and move on.
- Sai has CQF Exam 2 due 2026-09-24 and Exam 3 due 2026-10-22 — keep sessions
  scoped and flag scope creep; the project must not eat exam prep.
- End of each session: run `pytest`, commit with an honest message, and add a
  3-line entry to JOURNAL.md (what shipped, what's next, spec deviations).

## Commands
- venv: `python3 -m venv .venv && source .venv/bin/activate`
- install: `pip install -r requirements.txt`
- tests: `pytest` (scaffold state: all skipped is expected)
- data rebuild (once implemented): `python -m quiet_index.data`
