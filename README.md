# Humane Intelligence Bias Bounty: Mapping Equity Challenge

**Author**: Byunggwon So (`@kevin9327ai`)  
**Competition**: [Zindi — Humane Intelligence Bias Bounty: Mapping Equity Challenge](https://zindi.world/competitions/bias-bounty-mapping-equity-challenge)  
**License**: CC-BY-SA 4.0 / MIT (100% Open-Source Python Stack)

---

## Overview & Strict Rule-Compliance Guarantee

This repository implements **100% pure challenge-package-only, deterministic (`SEED = 42`, CPU-only) coverage-gap estimators** across all **9,794 scored US Census Tracts** in the five study regions (`eastern-ok`, `eastern-wa`, `maricopa-az`, `northern-ca`, `south-central-tx`), together with our reproducible audit suite for the **Best Bias Discovery ($1,000)** and **Best Documentation ($500)** special prizes:

- **Zero Withdrawn Reference Layers & Zero External Datasets**: Scored submissions use **only** the official `2026-09-25` Source Cooperative challenge package (`Overture Maps 2026-08-19.0`, `census-acs-housing`, `census-tracts`, `strata-tract-table`, `usgs-combined-tract-table`, `SampleSubmission.csv`). No TIGER/Line roads, standalone Microsoft Building Footprints, HIFLD/USGS structure points, Census CBP, external datasets, reconstructed ground-truth targets, leaderboard weight fitting, or third-party participant files are used anywhere in the pipeline.
- **Official Selected Zindi Private-LB & Special-Prize Submissions (100% Pure `2026-09-25` Challenge Package)**:

| Zindi Submission ID | Public LB RMSE | Output CSV | Reproduction Command | Full SHA-256 | Feature Scope |
|---|---:|---|---|---|---|
| **`91U7cbTP`** *(Selected #1)* | **`0.084048897`** | `zindi_v8_pure_package_hierarchical.csv` | `python pure_package_coverage_gap_estimator.py` | `8f747f4813355218f43e2497418a7c5223068452a9cb94aa878650e147865e8f` | Full `2026-09-25` Challenge Package (`Overture 2026-08-19.0` + `census-acs-housing` + `census-tracts` + `strata-tract-table` + `usgs-combined-tract-table`) |
| **`aPoZjE6b`** *(Selected #2)* | **`0.085139920`** | `zindi_v7_pure_package_structural.csv` | `python pure_package_coverage_gap_estimator.py` | `03eeee1df3e88a4686b346c4e4b85e560fa350afc1fe3be6f4fd52e3549debf8` | Minimal Firewall (`Overture 2026-08-19.0` + `census-acs-housing` + `census-tracts` & basic Census demographics **only** — zero USFS/CarbonPlan structure counts) |
| *(Research Extension `v9`)* | *(Post-quota)* | `zindi_v9_pure_package_champion.csv` | `python pure_package_v9_estimator.py` | `1a11aff4799f0d31677c2cfe0e21eac017b9ddc27dbe9e784bead80dc2e1c574` | Full `2026-09-25` Challenge Package + EPSG:5070 Federal/State Highway Boundary-Buffer Mixture & 300m Campus Clustering |

- **Model Details**:
  1. **`zindi_v8_pure_package_hierarchical.csv` (Model `v8` — Full-Package Hierarchical PU-Anchor & Spatial Empirical-Bayes, Zindi Submission `91U7cbTP`, Public RMSE `0.084048897`)**, overall mean `0.05657`: Combines closed-form Black-Scholes truncated log-normal expectation, Positive-Unlabeled (PU) Ridge-Poisson anchor regression on high-completeness Overture tracts, dual-system capture-recapture facility Bayes across all `2^4 = 16` `(fire, ems, schools, cbp)` states, dynamic `2^3 - 1 = 7` undefined-divisor integration, County Fay-Herriot Empirical-Bayes shrinkage, and Queen-contiguity Graph-Laplacian smoothing. Matches `Best_Bias_Discovery_kevin9327ai.pdf` and `Best_Documentation_kevin9327ai.pdf` byte-identically.
  2. **`zindi_v7_pure_package_structural.csv` (Model `v7` — Minimal-Firewall Structural Bayes, Zindi Submission `aPoZjE6b`, Public RMSE `0.085139920`)**, overall mean `0.05396`: Enforces a strict minimal feature firewall excluding `usfs_BuildingCount_sum` and `carbonplan_buildings`.
  3. **`zindi_v9_pure_package_champion.csv` (Model `v9` — EPSG:5070 Highway Network Boundary-Buffer Mixture + 300m Spatial Campus Clustering + 5-Fold County-Grouped PU GBDT Ensemble)**, overall mean `0.03316`, std `0.04673`: Reconstructs federal/state route networks (excluding non-TIGER Texas `FM/RM` and county sub-networks) over `22m/38m` exterior EPSG:5070 tract buffers, aligns school numerators strictly to the 6 official primary categories, and reconciles `ovt_pois` against 5-digit-ZIP commercial address ratios.

---

## Repository Structure

| File | Purpose |
|---|---|
| `extract_v9_pure_package_features.py` | Streams live official `2026-09-25` Source Cooperative GeoParquets (`EPSG:5070`) via DuckDB Spatial to extract federal/state highway route buffers, 6-category school & 5-digit ZIP POI counts, and 300m physical facility clusters (`challenge_package_v9_features_9794.parquet`). |
| `pure_package_v9_estimator.py` | Deterministically generates **`zindi_v9_pure_package_champion.csv`** and `pure_package_v9_diagnostics.json` (`SEED = 42`, CPU-only). |
| `extract_challenge_package_features.py` | Extracts Overture provenance, typology, sibling emergency POIs & geodesic water fractions from official `2026-09-25` Source Cooperative GeoParquets. |
| `pure_package_coverage_gap_estimator.py` | Deterministically generates `zindi_v7_pure_package_structural.csv`, `zindi_v8_pure_package_hierarchical.csv` (`91U7cbTP`), and `pure_package_v7_v8_diagnostics.json`. |
| `reproduce_bias_discovery.py` | Recomputes all 2,000-iteration bootstrap 95% CIs, Mann-Whitney *U* tests, and named-tract tables across `19,169,786` Overture buildings (`reproducible_bias_discovery_results.json`). |
| `render_special_prize_pdfs.py` | Renders vector-illustrated `Best_Bias_Discovery_kevin9327ai.pdf`, `Best_Documentation_kevin9327ai.pdf`, and `BIAS_BOUNTY_WRITEUP.md`. |
| `Best_Bias_Discovery_kevin9327ai.pdf` | Dedicated PDF submission for **Best Bias Discovery ($1,000)** (Impact 30%, Novelty 25%, Named-Tract Evidence 25%, Reproducibility 20%). |
| `Best_Documentation_kevin9327ai.pdf` | Dedicated PDF submission for **Best Documentation ($500)** (Data sources, closed-form mathematics, 9,794-tract edge-case audit, 5-scheme weighting sensitivity). |
| `BIAS_BOUNTY_WRITEUP.md` | Full Markdown methodology & bias-discovery report. |
| `data/` | Official regional challenge CSVs (`census-acs-housing`, `strata-tract-table`, `usgs-combined-tract-table`, `sample-submission`) for all 5 regions. |

---

## Key Bias Discovery Highlights (`reproduce_bias_discovery.py`)

1. **The Machine-Footprint Semantic Mirage**: Across `19,169,786` Overture buildings, `8,118,258` (`42.3%`) are unverified single-source `Microsoft ML Buildings` roof polygons with `NULL` `subtype`/`class` and `NULL` `height`. Non-Tribal tracts have **3.76x** higher building semantic typing than Tribal tracts (`tribal_pct >= 25%`) even though raw footprint counts appear `~99.6%` complete everywhere.
2. **Invisible Volunteer Fire Stations & Route-Shield Highway Inversion**: Identifies **354 tracts (1,286,312 residents)** where Overture reports `fire_department = 0` despite the station existing inside Overture under `fire_protection_service`, VFD names, or OSM `fire_station` buildings, alongside **5.20x** higher exclusion of numbered rural evacuation highways down-tagged as `tertiary`.

---

## One-Command Deterministic Reproduction (`SEED = 42`, CPU-Only)

```bash
pip install -r requirements.txt

# (Optional) Re-extract Overture semantic & EPSG:5070 spatial features directly from Source Cooperative S3:
python extract_challenge_package_features.py
python extract_v9_pure_package_features.py

# 1. Reproduce BOTH Official Selected Zindi Submissions (`91U7cbTP` v8 & `aPoZjE6b` v7) + diagnostics:
python pure_package_coverage_gap_estimator.py

# 2. Reproduce Model v9 pure-package research extension (`zindi_v9_pure_package_champion.csv`):
python pure_package_v9_estimator.py

# 3. Reproduce all Best Bias Discovery bootstrap 95% CIs, p-values & named tracts:
python reproduce_bias_discovery.py

# 4. Render Special-Prize PDFs & Markdown reports:
python render_special_prize_pdfs.py
```
