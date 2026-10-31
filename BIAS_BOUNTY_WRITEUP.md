# Humane Intelligence Bias Bounty: Mapping Equity Challenge — Methodology, Best Bias Discovery & Best Documentation

- **Author / Zindi Username**: Byunggwon So (`kevin9327ai`)
- **Selected 100% Pure Challenge-Package Submissions**:
  - `zindi_v7_pure_package_structural.csv` (Zindi ID: `aPoZjE6b (Public RMSE 0.08513992)`, SHA-256: `03eeee1df3e88a4686b346c4e4b85e560fa350afc1fe3be6f4fd52e3549debf8`)
  - `zindi_v8_pure_package_hierarchical.csv` (Zindi ID: `91U7cbTP (Public RMSE 0.084048897)`, SHA-256: `8f747f4813355218f43e2497418a7c5223068452a9cb94aa878650e147865e8f`)
- **Special Prize Consideration**: **Both** — **Best Bias Discovery ($1,000 USD)** and **Best Documentation ($500 USD)**
- **Public Repository**: https://github.com/kevin9327/ZindiBiasBounty

---

## Part I — Best Bias Discovery ($1,000 Prize Section)

### 1. Executive Summary: Why the Automated 5-Metric Bias Scorecard Declares False Equity
Across all **9,794 scored census tracts** (19,169,786 Overture `2026-08-19.0` building footprints across 5 study regions), count-based building completeness metrics report near-zero deficits (`building_gap ≈ 0.004`, ~99.6% parity) across Tribal, wildfire-exposed, and unincorporated border communities. By auditing Overture's nested `sources[].dataset` provenance, `subtype`/`class` typology, primary-vs-sibling emergency POI categories, and highway route shields (`routes`), we expose **three compounding geospatial biases outside the automated API**:

1. **The Machine-Footprint Semantic Mirage (Upstream Source-Conflation Bias)**: **8,118,258 buildings (42.3%)** are single-source `Microsoft ML Buildings` computer-vision roof polygons with **0% human verification, 100% `NULL` `subtype`/`class`, and 100% `NULL` `height`**. Non-Tribal tracts have **3.76x higher semantic building typing** (95% CI `[3.31x, 4.28x]`, Mann-Whitney `p = < 1e-16`) than Tribal tracts (`tribal_pct >= 25%`), while Tribal tracts rely **2.37x more heavily** on unverified single-source ML blobs (`69.4%` in Eastern OK vs `13.0%` in Maricopa AZ).
2. **Invisible Volunteer Fire/EMS Stations & Route-Shield Highway Inversion (Taxonomy Bias)**:
   - In **354 tracts (1,286,312 residents)**, Overture reports `fire_department = 0` (100% deficit) even though the physical station exists inside Overture under commercial `fire_protection_service`, VFD text names, or OSM `class='fire_station'` buildings!
   - In rural evacuation corridors, **20.9%** of numbered highway segments (`routes IS NOT NULL`) are down-tagged as `tertiary` and severed by `(motorway, trunk, primary, secondary)` filters vs **4.0%** in urban tracts (**5.20x disparity**, `p = < 1e-16`).
3. **Within-Rural Wildfire Burn Scars & Intra-County *Colonia* Pocket Masking**: High-SVI peri-urban/*colonia*-profile pockets (`SVI >= 0.85`, `SVI_housing_transport >= 0.80`) nested inside benign low-gap counties (`county mean < 0.072`) suffer **2.29x higher coverage gaps** (95% CI `[2.04x, 2.54x]`, `p = < 1e-16`).

### 2. Regional Upstream Provenance & Semantic Completeness Table (All 9,794 Tracts)

| Study Region | Tracts | Population | Overture Buildings | MS-ML-Only Share | Semantically Typed Share | Height Present | Mean `v8` Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| `eastern-ok` | 1,192 | 3,926,466 | 2,551,694 | **69.4%** | **5.8%** | 70.35% | `0.09672` |
| `eastern-wa` | 415 | 1,676,917 | 1,081,345 | **38.1%** | **12.2%** | 71.86% | `0.07165` |
| `northern-ca` | 591 | 2,458,891 | 1,164,723 | **50.8%** | **11.7%** | 83.16% | `0.06233` |
| `south-central-tx` | 6,001 | 25,758,569 | 11,463,800 | **43.3%** | **19.7%** | 74.28% | `0.05174` |
| `maricopa-az` | 1,593 | 6,606,494 | 2,908,224 | **13.0%** | **37.9%** | 80.80% | `0.03871` |

### 3. Named Tract Evidence (Selected High-Impact Tracts)

| GEOID | Region | Jurisdiction / Tribal Area | Pop | Buildings | Raw `building_gap` | MS-ML-Only % | Untyped % | SVI |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `48201433505` | `south-central-tx` | Harris County, TX | 2,898 | 189 | `0.0063` | **100.0%** | **100.0%** | 0.954 |
| `40049681600` | `eastern-ok` | Chickasaw OTSA | 1,839 | 1,969 | `0.0188` | **100.0%** | **100.0%** | 0.425 |
| `06007002601` | `northern-ca` | Butte County, CA | 2,474 | 2,021 | `0.0065` | **100.0%** | **100.0%** | 0.480 |
| `53065950200` | `eastern-wa` | Stevens County, WA | 4,076 | 4,916 | `0.0090` | **98.0%** | **99.4%** | 0.487 |
| `04009961202` | `maricopa-az` | Graham County, AZ | 4,102 | 1,923 | `0.0089` | **95.8%** | **98.0%** | 0.512 |
| `40099790600` | `eastern-ok` | Chickasaw OTSA | 2,041 | 2,419 | `0.0137` | **95.8%** | **100.0%** | 0.360 |
| `53005010701` | `eastern-wa` | Benton County, WA | 2,184 | 1,707 | `0.0066` | **95.8%** | **99.5%** | 0.373 |
| `48215020505` | `south-central-tx` | Hidalgo County, TX | 2,350 | 873 | `0.0258` | **95.8%** | **99.4%** | 0.974 |

---

## Part II — Best Documentation ($500 Prize Section)

### 1. Strict 2026-09-25 Challenge-Package Compliance & Dual-Model Architecture
- **Zero Prohibited Reference Data**: Neither `v7` nor `v8` touches withdrawn TIGER/Line roads, standalone Microsoft Building Footprints, HIFLD/USGS structure points, Census CBP, reconstructed ground-truth files, or external participant repositories.
- **Model `v7` (`zindi_v7_pure_package_structural.csv`)**: Enforces a minimal feature firewall (`Overture 2026-08-19.0` + `census-acs-housing` + `census-tracts` polygons & basic census demographics only — zero USFS/CarbonPlan structure counts).
- **Model `v8` (`zindi_v8_pure_package_hierarchical.csv`)**: Incorporates the full official `2026-09-25` challenge package (`strata-tract-table` 232 columns + `usgs-combined-tract-table`), adding Positive-Unlabeled (PU) Ridge-Poisson Anchor Regression on high-completeness Overture tracts, USFS/CarbonPlan physical structure reconciliation, County Empirical-Bayes shrinkage, and Queen-contiguity Graph-Laplacian smoothing.

### 2. Closed-Form Component Mathematics
1. **Black-Scholes Truncated Log-Normal Expectation**: Evaluates `E[max(0, 1 - O/R)] = Phi(z) - (O/R_med) * exp(0.5*sigma^2) * Phi(z - sigma)` (`z = ln(R_med/O)/sigma`) in closed form to eliminate Jensen's-inequality downward plug-in bias.
2. **Dual-System Capture-Recapture & Finite-Slot Facility Bayes**: Estimates Overture Places recall `q_k` endogenously against OSM typed facility buildings (`fire_station`, `school`) and integrates over all `2^4 = 16` `(fire, ems, schools, cbp)` defined states (excluding inflated `hospital` POIs).
3. **Dynamic Undefined-Component Divisor Integration**: Integrates `(d_r*g_r + d_b*g_b + d_p*g_p)/(d_r + d_b + d_p)` over all `2^3 - 1 = 7` active states calibrated to the Evaluation tab's regional undefined highway priors (`55%` Maricopa, `37%` Northern CA, `28%` South-Central TX, `21%` Eastern OK, `22%` Eastern WA).

### 3. Edge-Case Audit Summary (All 9,794 Tracts)
- **100% Offshore Water / Zero-Overture Tracts (`n = 2`)**: `48039990000` & `48057990000` (`ALAND = 0`, `pop = 0`, `Overture = 0`) -> all `*_defined = False`, `coverage_gap_score = 0.0`.
- **Water-Dominated Tracts >=50% Water (`n = 43`)**: All spatial densities normalized strictly by geodesic land area `max(ALAND/1e6, 0.02 km^2)` against `EPSG:5070` Albers Equal-Area total polygon area (mean `v8` = `0.06989`).
- **Zero-Population Tracts (`n = 27`)**: Separates commercial/airport hubs from uninhabited parklands; gates K-12 school demand to ~0 unless observed while preserving commercial building/POI expectations (mean `v8` = `0.04430`).
- **Cross-State & Leading-Zero 11-Char FIPS Preservation**: Preserves leading zeros across Arizona (`04*`: 1,592 tracts) and California (`06*`: 591 tracts) and retains Maricopa's cross-border Hidalgo County, New Mexico tract `35023970000`.

### 4. Alternative Weighting Sensitivity (All 9,794 Tracts, Model `v8`)
- **Official Equal-Tract Dynamic-Divisor Mean**: `0.05657` (`v7`: `0.05396`)
- **Emergency-First Domain Weighting (0.45 Road / 0.15 Bldg / 0.40 POI)**: `0.07270`
- **Population-Weighted Mean**: `0.05236`
- **SVI + CVI Climate-Justice Equity-Weighted Mean**: `0.05661`
- **Land-Area-Weighted Mean (`ALAND`)**: `0.13128`

### 5. Deterministic 1-Command Reproduction (`SEED = 42`)
```bash
python extract_challenge_package_features.py
python pure_package_coverage_gap_estimator.py
python reproduce_bias_discovery.py
python render_special_prize_pdfs.py --v7-id <V7_ID> --v8-id <V8_ID>
```
