# scripts/

One-off tools that are not part of the running app. Run them from the repo root as modules,
so project imports (`config`, `graph`, …) and data paths (`data/…`) resolve:

```bash
.venv/Scripts/python.exe -m scripts.checks.check_neo4j
.venv/Scripts/python.exe -m scripts.efl.efl_post_processing
```

| Folder | Contents |
|---|---|
| `checks/` | Database and API checks, plus old ad-hoc `test_*.py` scripts (the real suite is `tests/`) |
| `data/` | Fixture and season data preparation (fetch, scrape, fix team names, process); `fetch_manager_changes.py` rebuilds `data/manager_changes.csv` from Wikipedia |
| `efl/` | Championship (EFL) data ingestion and validation for promoted teams |
| `embeddings/` | `build_embeddings_from_csv.py` rebuilds `embeddings/prebuilt/` (run by the `build-embeddings` workflow) |
| `analysis/` | LLM evaluation (writes `docs/reports/evaluation_results/`), prediction plots, and `ml_backtest.py`, the points-model backtest run by the **ML backtest** workflow |
| `site/` | `node scripts/site/render.mjs` regenerates the OG image and favicon set in `public/` (headless Edge/Chrome + Pillow); `node scripts/site/screenshots.mjs [url]` recaptures the README screenshots in `docs/screenshots/` from the live site |
| top level | Model training helpers and promoted-team deployment |

Many of these were written for earlier stages of the project and may need updates before
they run against the current graph.
