# Bias Bounties @ Scale: Map Data Extracts

<p align="right"><b>Tutorial Notebook:</b> <a href="https://github.com/kentstephen/bias-bounty-map-tutorial">on Github</a> | <a href="https://colab.research.google.com/github/kentstephen/bias-bounty-map-tutorial/blob/main/bias-bounty-explore-tutorial.ipynb">open in Colab</a></p>

Reference map data for the **Bias Bounties @ Scale** challenge (Humane Intelligence /
Reliabl). The challenge measures whether open map data (**Overture Maps**) systematically
under-covers climate-vulnerable, rural, and tribal communities in the US, scored as a
**coverage-gap score per Census tract**.

This product holds the **Overture extracts**, **ACS housing units** per tract, and national
tract-level **strata** tables, clipped to five study regions and packaged as cloud-native
**GeoParquet**.

> **2026-09-25:** TIGER/Line roads, Microsoft building footprints, HIFLD facilities and CBP
> establishments were removed from every region at the challenge organizers' request.
> `eastern-wa` was added (Overture, ACS, strata, boundaries).


> ### 📥 Getting the data
>
> **No download needed.** Every snippet in this README reads straight from the bucket, no
> account, no credentials; that is the suggested way to work with it.
>
> &nbsp;&nbsp;&nbsp;&nbsp;**[Jump to: Access (read remotely)](#access)**
>
> **Need the whole package on disk?** Use `aws s3 sync`, not the browser. It resumes if the
> connection drops and re-running it only fetches what changed. Zips per region exist as a
> fallback for anyone who cannot install the CLI, but a multi-GB browser download that fails
> partway does not resume.
>
> &nbsp;&nbsp;&nbsp;&nbsp;**[Jump to: Bulk download, step by step (aws s3 sync)](#bulk-download-aws-cli)**
>
> ```bash
> aws s3 sync s3://us-west-2.opendata.source.coop/humane-intelligence/bias-bounty-mapping-equity-challenge/reference/ ./reference/ --no-sign-request
> ```

## Regions

| Folder | Region | Primary hazard |
|---|---|---|
| `maricopa-az` | Maricopa County + surrounding, AZ | Heat + drought |
| `northern-ca` | Butte / Shasta / Tehama fire corridor, CA | Wildfire |
| `eastern-ok` | Eastern Oklahoma (Cherokee/Muscogee/Choctaw area) | Drought + heat |
| `south-central-tx` | Bexar → Hidalgo along I-35/I-37, TX | Heat + drought |
| `eastern-wa` | Eastern Washington (Columbia Basin, Yakima to Spokane; added 2026-09-08) | Wildfire + drought |

Each region covers well over the challenge's 5-tract minimum (415 to 6,010 tracts).

**Every region boundary is the dissolve of its own Census tracts** (`eastern-ok` since
2026-08-12, the next three since 2026-08-26, `eastern-wa` from its creation on 2026-09-08), so every member tract lies fully inside its
region and every reference layer covers the whole of every scored tract. Membership is frozen
in `boundaries/<region>-tract-geoids.csv` (1,593 / 591 / 1,192 / 6,010 / 415). The authoritative
scored tract list for every region is its `<region>-sample-submission.csv` (one row per scored
tract), which matches `strata/<region>/<region>-census-tracts.parquet` exactly. Two rules keep
those lists clean: **a tract belongs to exactly one region, decided by state FIPS** (two
Oklahoma tracts on the Texas border, `40013795900` and `40085094200`, are `eastern-ok` only),
and **tracts with no scorable reference data are dropped from the list** (7 in
`south-central-tx`, all 99xx water tracts; none elsewhere). Scored rows: 1,192 / 1,593 / 591 / 6,003 / 415.
One membership note: `maricopa-az` is 1,592 Arizona tracts plus one New Mexico tract,
`35023970000` (Hidalgo County, Tract 9700, the bootheel), which overlaps the original AOI ring
where it meets the state line. It is a member on the same terms as every other tract; the
region is "Maricopa County and surrounding", not Arizona.

## Layout

Region-major. Every file name is self-describing (`<region>-<source>-<layer>.parquet`), so a
file pulled out of its folder still identifies itself.

```
reference/                                   # map data scored for coverage
  <region>/
    <region>-overture-buildings.parquet
    <region>-overture-roads.parquet
    <region>-overture-roads-unfiltered.parquet
    <region>-overture-rail.parquet
    <region>-overture-infrastructure.parquet
    <region>-overture-pois.parquet
    <region>-census-acs-housing.parquet       (+ .csv)
    <region>-sample-submission.csv            # the scored tract set, submission-shaped
strata/                                      # tract-keyed demographic / vulnerability / hazard strata
  national/                                  # source of truth, 85,396 tracts (full list below)
  <region>/                                  # the same tables, subset to one region
    <region>-strata-tract-table.parquet       (+ .csv)   # every layer joined
    <region>-svi-tract-table.parquet          (+ .csv)
    <region>-cvi-tract-table.parquet          (+ .csv)
    <region>-ruca-tract-table.parquet         (+ .csv)
    <region>-rucc-tract-table.parquet         (+ .csv)
    <region>-nchs-tract-table.parquet         (+ .csv)
    <region>-tribal-tract-table.parquet       (+ .csv)
    <region>-usdm-drought-tract-table.parquet (+ .csv)
    ...                                       # and the rest of the hazard layers
    <region>-census-tracts.parquet
    <region>-census-aiannh.parquet
    <region>-census-tribal-tracts.parquet
    <region>-census-tribal-subdivisions.parquet   # where present
boundaries/
  all-aois.geojson
```

### Strata: national tables

The strata are **national**, one row per 2020 census tract, **85,396 rows**, keyed on `GEOID`.
`strata/national/` is the source of truth; `strata/<region>/` holds the same tables subset to a
region, so the two cannot drift. Every table below ships as **`.parquet` and `.csv`**.

```
strata/
  national/
    national-strata-tract-table.parquet       # ALL layers joined, 232 columns: start here
    national-census-tract-table.parquet       # the spine: population, urban/rural, land area
    national-census-tracts.parquet            # tract polygons (parquet only): join geometry here

    national-svi-tract-table.parquet          # CDC Social Vulnerability Index 2022
    national-cvi-tract-table.parquet          # Climate Vulnerability Index (composite + pillars)
    national-ruca-tract-table.parquet         # USDA RUCA 2020
    national-rucc-tract-table.parquet         # USDA Rural-Urban Continuum 2023
    national-nchs-tract-table.parquet         # NCHS Urban-Rural Scheme 2013
    national-tribal-tract-table.parquet       # tribal overlap: legal / statistical / union

    national-usdm-drought-tract-table.parquet     # US Drought Monitor, seasonal DSCI
    national-drought-gov-tract-table.parquet      # SPI (5 timescales) + Palmer PMDI
    national-epht-heat-tract-table.parquet        # CDC EPHT extreme heat
    national-nasa-heat-tract-table.parquet        # NASA GEHE + UHE-Daily
    national-cdc-wonder-tract-table.parquet       # CDC WONDER heat-wave days
    national-cdc-wonder-heat-mortality.parquet    # heat mortality: NATIONAL totals, not tract-keyed
    national-noaa-ghcn-tract-table.parquet        # weather-station coverage per tract
    national-noaa-ghcn-stations.parquet           # the stations themselves, with their tract

    national-usfs-wildfire-tract-table.parquet    # USFS Wildfire Risk to Communities, all layers
    national-carbonplan-tract-table.parquet       # CarbonPlan fire risk v1.1.0
    national-mtbs-tract-table.parquet             # MTBS burned area 1984-2023
    national-nifc-tract-table.parquet             # NIFC perimeters 1984-2025
    national-fpa-fod-tract-table.parquet          # FPA FOD ignitions 1992-2020
    national-usgs-combined-tract-table.parquet    # USGS combined: reburn + fire return interval

    national-census-aiannh.parquet            # tribal area geometry (parquet only)
    national-census-tribal-tracts.parquet      #   "
    national-census-tribal-subdivisions.parquet #  "
  <region>/                                   # every table above, subset to the region
```

**`national-strata-tract-table`** joins every layer into one 232-column table so you can stratify
without a dozen joins. It carries each source's headline columns; the per-source tables keep the
full detail (per-year components, per-raster min/max/stdev). Column names are prefixed by source
(`svi_`, `cvi_`, `usdm_`, `usfs_`, `mtbs_`, …) so provenance reads off the name.

**Coverage is not uniform and is never filled with zero.** Each source carries a `*_covered` flag
and explicit nulls where it does not reach. Several layers are CONUS-only (CarbonPlan, USFS, CDC
EPHT, CDC WONDER, SPI/PMDI); NCHS and CVI have never covered Puerto Rico; USDM and MTBS are the
only fully national hazard layers.

## Layers

| Source | Layer | Geometry | What it is |
|---|---|---|---|
| Overture | `buildings` | polygons | All building footprints (`theme=buildings`). |
| Overture | `roads` | lines | Drivable road segments (`transportation/segment`, `subtype=road`): motorway, trunk, primary, secondary, tertiary, residential, unclassified. Escape-route framing; service/pedestrian classes dropped. |
| Overture | `roads-unfiltered` | lines | The same `subtype=road` extract with **no class filter** (adds service, track, footway, cycleway, path, living_street, ...), so what the class filter drops can be explored directly (mostly `service` roads: driveways, parking aisles, alleys). Not the scored layer. |
| Overture | `rail` | lines | Rail segments (`subtype=rail`, all classes). |
| Overture | `infrastructure` | points (+ some lines) | Transit stations/stops and airports (`base/infrastructure`, class-filtered to stations, halts, platforms, bus stops/stations, and airport facilities). |
| Overture | `pois` | points | Places of interest (`theme=places`), full `categories` taxonomy retained. |
| Census | `acs-housing` | tract polygons | ACS 5-Year table **B25001** total housing units per tract, joined to TIGER tract geometry. |

## Format & conventions

- **GeoParquet 1.1**, ZSTD-compressed, Hilbert-sorted, with a spec covering `bbox` column for
  cloud-native range reads. Written with [`geoparquet-io`](https://github.com/cholmes/geoparquet-io).
- **CRS: `OGC:CRS84`** (lon/lat, equivalent to EPSG:4326) on every layer.
- **Spatial layers are cut at the region boundary, and the boundary follows whole tracts.**
  A footprint or road crossing the boundary ships as its portion inside the region; nothing
  extends past it. Because every region boundary is the outline of its own tracts (2026-08-26;
  earlier versions used a rougher polygon, which left 256 edge tracts comparing clipped map data
  against whole-tract ACS / CBP counts and inflated their gaps 5 to 8x), the cut never lands
  inside a scored tract. `<region>-census-tracts.parquet` still carries **`frac_inside_aoi`**
  (each tract's share of area inside the region, EPSG:5070); it is 1.0 for every tract and is
  kept so the invariant is visible in the data.
- **Tribal areas/tracts (strata) can fall outside the AOI and overlap a neighboring region.**
  Those geometry layers keep the whole-feature rule: a large tribal area clipping the edge is
  kept in full, so it can extend well past the region boundary.
- **Parquet is primary; CSV is a companion, not a second product.** Where both exist they are
  the same rows and the same values, so pick whichever your tooling prefers. CSV is offered
  wherever it can carry the *same information* as the parquet: the tract-keyed tables (join on
  `GEOID`) and the point layers (geometry becomes `lon` / `lat` columns, which is lossless for a
  point and needs no geospatial library). Polygon and line layers, and the Overture layers with
  nested columns (`categories`, `names`, `sources`, `addresses`), are **parquet-only**, a CSV of
  those would either be enormous WKT or would quietly drop structure.
- **Reference coverage-gap CSVs (`<region>-coverage-gap.csv`, Reliabl's scoring) publish every
  component with a `_defined` flag.** `transport_gap`, `building_gap` and `poi_gap` each have a
  `transport_defined` / `building_defined` / `poi_defined` true/false column beside them. A
  component is undefined when the tract has no reference of that kind to compare against (no
  named highway, no Microsoft footprint, no critical facility and no establishment); it is then
  written as `0` with `_defined = false`. **`coverage_gap_score` is the mean of the defined
  components only**, so a `0` with `_defined = false` means "nothing to compare", not "fully
  mapped", and the composite reproduces from the CSV in every row. Tracts with all three
  undefined are dropped from the CSV and the sample submission.
- **The road gap compares named-highway classes only**: TIGER/Line `S1100` + `S1200` (primary
  and secondary roads) against Overture `motorway` / `trunk` / `primary` / `secondary`. Those
  class definitions differ between the two sources, so the road component measures Overture's
  coverage of the highway network, and its values are **not comparable across regions** (a rural
  region with few highways scores differently from a metro one for reasons that have nothing to
  do with mapping effort). Tracts with no TIGER highway at all have no road component
  (`transport_defined = false`).
- **`poi_gap` is per facility type.** Its HIFLD half matches Overture places to each HIFLD layer
  on `categories.primary`: fire stations = `fire_department`; EMS = `ambulance_and_ems_services`;
  schools = `elementary_school`, `middle_school`, `high_school`, `school`, `private_school`,
  `public_school`. For each type, `1 - min(1, overture / hifld)` per tract, undefined where the
  tract has no HIFLD facility of that type; `poi_gap_hifld` is the mean over the defined types.
  Hospitals are excluded: Overture's `hospital` category is about 12x HIFLD's hospital count
  everywhere, so that term never binds. EMS is close to 1:1 and rarely binds either; fire
  stations carry the component. The CBP half (all Overture places against CBP establishments)
  is unchanged, and `poi_gap` is the mean of the two halves. The per-type values ship as
  `poi_gap_fire`, `poi_gap_ems`, `poi_gap_schools`, `poi_gap_cbp`, each with its own
  `poi_defined_*` flag.
- **Strata tables carry no geometry.** The tract polygons ship once, in
  `strata/national/national-census-tracts.parquet`; join any strata table to it on `GEOID` to map
  it. At region scale the same columns ride on `<region>-census-tracts.parquet` directly.

## Provenance / vintage

- **Overture Maps** release **`2026-08-19.0`** (pinned; all Overture layers use it. The first
  issue of this product was cut from `2026-06-17.0`; Overture removes each release 60 days after
  publication, so the 2026-08-26 re-issue moved every region to the current release).
  The Overture attributes are shipped as released: every column and value is Overture's own,
  and nothing was added, renamed, or trimmed. The one thing that changed is the `bbox` column,
  which was regenerated by the repartition (the same values written in GeoParquet-spec covering
  order, `xmin, ymin, xmax, ymax`, instead of Overture's native `xmin, xmax, ymin, ymax`) so
  the extracts carry the same bbox convention as every other layer in this product.
- **Census ACS** 2020-2024 5-Year Estimates (2024 vintage, the latest 5-year release at pull time, July 2026).
- **TIGER/Line** 2025 (tract geometry, tribal areas).

Strata vintages differ by source, deliberately. Each table carries its own vintage column, so
this list is never the only record of it.

| Source | Vintage | | Source | Vintage |
|---|---|---|---|---|
| CDC SVI | 2022 | | USFS Wildfire Risk | 2024 (2nd ed.) |
| Climate Vulnerability Index | 2010 tracts, crosswalked to 2020 | | CarbonPlan | v1.1.0 (2026) |
| USDA RUCA | 2020 | | MTBS | 1984-2023 |
| USDA RUCC | 2023 | | NIFC | 1984-2025 |
| NCHS Urban-Rural | 2013 | | FPA FOD | 6th ed., 1992-2020 |
| Census tracts / population | 2020 (DHC), TIGER 2025 | | USGS Combined Wildfire | FY21, to 2020 |
| USDM, SPI, PMDI, EPHT heat | 2015-2023 seasonal indices | | NASA GEHE / UHE-Daily | 1983-2016 |
| CDC WONDER heat-wave days | 1981-2010 | | NOAA GHCN | 2015-2023 |

**Tract vintage is 2020 throughout** (as published in TIGER 2025), so every table joins to every
other on `GEOID`. Sources published on older tract geography were crosswalked, population-weighted.

## Access

No credentials, no signup. One line:

```python
import geopandas as gpd
from lonboard import viz

gdf = gpd.read_parquet("s3://us-west-2.opendata.source.coop/humane-intelligence/bias-bounty-mapping-equity-challenge/reference/northern-ca/northern-ca-overture-infrastructure.parquet")

viz(gdf) # visualize
```

> ### Two routes to the same bytes
>
> Everything here is reachable directly from the bucket
> (`s3://us-west-2.opendata.source.coop/…`) or through the Source Cooperative data proxy
> (`https://data.source.coop/…`). Both are public, no credentials either way. The proxy is
> S3-compatible, so it works with `s3://` clients too when you pass it as the endpoint.
> The snippets below use the bucket URI because it's the shortest form every library shown
> accepts. Support matrix at the end of this section.

The files are cloud-native, so **you don't have to download them to query them.** Pull a window out
of the 3 GB buildings layer in about two seconds. `bbox=` pushes the filter down to the row groups:

```python
gdf_buildings = gpd.read_parquet(
    "s3://us-west-2.opendata.source.coop/humane-intelligence/bias-bounty-mapping-equity-challenge/reference/south-central-tx/south-central-tx-overture-buildings.parquet",
    bbox=(-98.52, 29.40, -98.46, 29.45),   # downtown San Antonio
)
```

That returns 20,066 footprints out of 4.4M, without downloading the file.

Tabular layers (no geometry) are the same call with `pandas`:

```python
import pandas as pd

df_acs = pd.read_parquet("s3://us-west-2.opendata.source.coop/humane-intelligence/bias-bounty-mapping-equity-challenge/reference/eastern-ok/eastern-ok-census-acs-housing.parquet")
```

### Strata

Same call. The strata are national tract tables, so read the whole thing and filter on `GEOID`:

```python
BASE = "s3://us-west-2.opendata.source.coop/humane-intelligence/bias-bounty-mapping-equity-challenge"

# every stratum in one table, 85,396 tracts x 232 columns
df = pd.read_parquet(f"{BASE}/strata/national/national-strata-tract-table.parquet")

# or one source at a time, with its full detail
svi = pd.read_parquet(f"{BASE}/strata/national/national-svi-tract-table.parquet")
```

To map any of it, join to the tract polygons, which ship once:

```python
tracts = gpd.read_parquet(f"{BASE}/strata/national/national-census-tracts.parquet")
gdf = tracts.merge(df, on="GEOID", how="left")
```

Pull just the columns you need with `columns=`; at 232 columns that is the difference between
a few MB and the whole file:

```python
df = pd.read_parquet(
    f"{BASE}/strata/national/national-strata-tract-table.parquet",
    columns=["GEOID", "pop_total", "ur_class", "svi_overall", "tribal_any", "usdm_summer_dsci"],
)
```

### DuckDB

Open every connection the same way. `httpfs` does the reading, `spatial` gives you the `ST_*`
functions, and the two `SET`s are what make `s3://` work against this bucket:

```python
import duckdb

con = duckdb.connect()
con.sql("INSTALL httpfs; LOAD httpfs;")
con.sql("INSTALL spatial; LOAD spatial;")
con.sql("SET s3_region='us-west-2';")
con.sql("SET s3_url_style='path';")

con.sql("""
    SELECT count(*)
    FROM 's3://us-west-2.opendata.source.coop/humane-intelligence/bias-bounty-mapping-equity-challenge/reference/northern-ca/northern-ca-overture-pois.parquet'
""")
```

`s3_url_style='path'` is required because the bucket name contains dots. Without it DuckDB puts the
bucket in the hostname, where Amazon's wildcard certificate can't cover it, and TLS fails.

The geometry column is already a `GEOMETRY` (DuckDB reads GeoParquet natively), so `ST_*` functions
take it directly. Don't wrap it in `ST_GeomFromWKB()`.

Same bbox trick as above. **Filter on `bbox` to prune row groups, then `ST_Intersects` to be exact**
(the bbox prune is what makes this fast; `ST_Intersects` alone is roughly 6x slower):

```python
con.sql("""
    SELECT *
    FROM 's3://us-west-2.opendata.source.coop/humane-intelligence/bias-bounty-mapping-equity-challenge/reference/south-central-tx/south-central-tx-overture-buildings.parquet'
    WHERE bbox.xmin <= -98.46 AND bbox.xmax >= -98.52
      AND bbox.ymin <=  29.45 AND bbox.ymax >=  29.40
      AND ST_Intersects(geometry, ST_MakeEnvelope(-98.52, 29.40, -98.46, 29.45))
""")
```

> Use all four comparisons. Filtering on `bbox.xmin`/`bbox.ymin` alone is a corner-in-box test, not
> an overlap test, and silently drops features that straddle the edge of your window.

### Two DuckDB gotchas (silent, so read this before computing lengths or areas)

**Axis order.** Every layer here is lon/lat (`OGC:CRS84`), as GeoParquet requires. DuckDB's
`spatial` extension follows the EPSG authority axis order when you name `EPSG:4326`, which is
**lat/lon**, so the two obvious ways to measure a road in metres both fail without an error:

- `ST_Transform(geometry, 'EPSG:4326', 'EPSG:5070')` returns `POINT (inf inf)` for every feature.
  Pass **`always_xy := true`**:

  ```sql
  SELECT sum(ST_Length(ST_Transform(geometry, 'EPSG:4326', 'EPSG:5070', always_xy := true))) / 1000 AS km
  FROM 's3://.../reference/eastern-ok/eastern-ok-overture-roads.parquet'
  ```

- `ST_Length_Spheroid(geometry)` (and `ST_Distance_Spheroid`) returns `NaN`, because those
  functions take lat/lon input. Flip first: **`ST_Length_Spheroid(ST_FlipCoordinates(geometry))`**.

Neither failure raises. `sum()` over `NaN` is `NaN` and `inf` propagates, so a whole run can
finish "successfully" with nothing usable in it. Test one segment whose length you know before
running anything over a layer. Verified on DuckDB 1.5.4 with the core `spatial` extension.

**GEOID is text, not a number.** Maricopa is state FIPS `04` (one tract is `35`); an integer read drops the leading
zero and every join silently fails for that region. Read it as a string everywhere:

```python
# DuckDB
con.sql("SELECT * FROM read_csv('maricopa-az-sample-submission.csv', types={'GEOID': 'VARCHAR'})")
# pandas
pd.read_csv("maricopa-az-sample-submission.csv", dtype={"GEOID": str})
```

The parquet files already store `GEOID` as a string; the CSVs are where this bites.

### Boundaries (GeoJSON)

```python
import geopandas as gpd

aois = gpd.read_file("/vsicurl/https://data.source.coop/humane-intelligence/bias-bounty-mapping-equity-challenge/boundaries/all-aois.geojson")
```

DuckDB reads it straight from the URL too:

```python
con.sql("""
    SELECT *
    FROM ST_Read('https://data.source.coop/humane-intelligence/bias-bounty-mapping-equity-challenge/boundaries/all-aois.geojson')
""")
```

> The `/vsicurl/` prefix matters. On a bare `https://` URL, GeoPandas probes the URL with `urllib`
> first to check for range support, the proxy rejects that User-Agent, and you get a **403** before
> GDAL ever sees it. `/vsicurl/` hands the URL straight to GDAL, so plain `gpd.read_file()` works
> with no credentials and no environment variables. `pyogrio.read_dataframe()` on the bare URL
> works too, for the same reason.

### All regions at once

Globbing is another reason to use `s3://`: it needs a LIST, which plain HTTPS cannot do.

```python
con.sql("""
    SELECT count(*)
    FROM 's3://us-west-2.opendata.source.coop/humane-intelligence/bias-bounty-mapping-equity-challenge/reference/*/*-overture-infrastructure.parquet'
""")
```

### Bulk download (aws CLI)

The suggested way to work with this data is remotely, with the tooling above: every snippet in
this README reads straight from the bucket, and none of them needs a local copy. If you do need
the entire package on disk, use `aws s3 sync` rather than downloading files from the site one at
a time in the browser. It needs no account, resumes if the connection drops, and re-running it
only fetches what has changed, so a re-issued file is a one-line update instead of a fresh
download.

Sizes per region, `reference/` only: Maricopa about 0.7 GB, Northern CA 0.35 GB, Eastern OK
0.6 GB, South Texas 2.55 GB, Eastern WA 0.3 GB. The strata tables (parquet + csv) add about 1.7 GB, and the boundaries file
is small.

**Step by step**

1. Install the aws CLI if you do not have it (`aws --version` tells you). Any one of these:

   ```bash
   brew install awscli          # macOS
   pip install awscli           # any platform with Python
   # Windows: https://awscli.amazonaws.com/AWSCLIV2.msi
   ```

   No `aws configure` step. The bucket is public and `--no-sign-request` below tells the CLI not
   to look for credentials.

2. Open a terminal in the project folder you are using for the challenge. The files land in the
   directory you run the command from (`./`), so `cd` there first.

3. Run the line for the region you want. Point the CLI at the bucket directly, no `--endpoint-url`.

   ```bash
   # one whole region
   aws s3 sync s3://us-west-2.opendata.source.coop/humane-intelligence/bias-bounty-mapping-equity-challenge/reference/south-central-tx/ ./south-central-tx/ \
     --no-sign-request
   ```

   Or everything at once, into a folder tree that mirrors the bucket:

   ```bash
   BUCKET=s3://us-west-2.opendata.source.coop/humane-intelligence/bias-bounty-mapping-equity-challenge

   aws s3 sync $BUCKET/reference/  ./reference/  --no-sign-request   # all five regions, ~4.5 GB
   aws s3 sync $BUCKET/strata/     ./strata/     --no-sign-request   # national + region strata
   aws s3 sync $BUCKET/boundaries/ ./boundaries/ --no-sign-request   # the AOI GeoJSON
   ```

   One layer across all regions also works, because the self-describing names make `--include`
   match at any depth:

   ```bash
   aws s3 sync $BUCKET/reference/ ./reference/ \
     --exclude '*' --include '*-overture-buildings.parquet' \
     --no-sign-request
   ```

4. **Check the byte counts, not just the file names.** A parallel download can exit cleanly and
   still leave truncated files behind (one tester ended up with five short files, including a
   5.6 MB copy of a 299.5 MB buildings layer, and nothing reported an error). Compare local
   sizes to the bucket listing:

   ```bash
   aws s3 ls $BUCKET/reference/south-central-tx/ --no-sign-request   # sizes on the bucket
   ls -l reference/south-central-tx/                                 # sizes on disk (dir on Windows)
   ```

   Any file that is smaller locally is incomplete. Re-running the same `aws s3 sync` line fixes
   it: sync compares size and timestamp and re-fetches only the files that differ.

**Zip archives.** Two archives are also on the bucket for anyone who cannot install the CLI,
under `bulk-downloads/` (posted 2026-08-27, rebuilt 2026-09-25 with `eastern-wa` and without the removed layers): `reference.zip` (all five regions of `reference/`, 4.52 GB) and
`strata.zip` (all of `strata/`, 1.69 GB). Each has this README at the top level.

```
https://data.source.coop/humane-intelligence/bias-bounty-mapping-equity-challenge/bulk-downloads/reference.zip
https://data.source.coop/humane-intelligence/bias-bounty-mapping-equity-challenge/bulk-downloads/strata.zip
```

Browser downloads of multi-GB files do not resume, so if one fails partway the archive is
corrupt and has to be fetched again: check the byte count against the size shown in the file
browser before unzipping. Prefer `aws s3 sync` where you can.

### Which address to use

Three ways to address the same bytes. The snippets in this README use the first, because it's the
shortest form every client accepts.

| | GeoPandas / pandas | DuckDB | aws CLI |
|---|---|---|---|
| **bucket**, `s3://us-west-2.opendata.source.coop/…` | ✅ | ✅ | ✅ |
| **proxy as S3 endpoint**, `s3://humane-intelligence/…` + `endpoint_url` | ✅ | ✅ | ✅ with `--endpoint-url` |
| **proxy as a URL string**, `https://data.source.coop/…` | ✗ (see below) | ✅ | n/a |

The proxy speaks the S3 API, so it is not HTTPS-only: point any S3 client at
`https://data.source.coop` as its **endpoint** and the org name (`humane-intelligence`) becomes the
bucket. It's also the only way to compare the two routes apples-to-apples, since it holds the
client stack fixed and changes only the server.

```python
gdf = gpd.read_parquet(
    "s3://humane-intelligence/bias-bounty-mapping-equity-challenge/reference/northern-ca/northern-ca-overture-infrastructure.parquet",
    storage_options={"endpoint_url": "https://data.source.coop", "anon": True},
)
```

`anon=True` is required. Without it fsspec looks for credentials it doesn't need and raises
`NoCredentialsError`. The pyarrow spelling is the same idea:

```python
from pyarrow.fs import S3FileSystem

fs = S3FileSystem(anonymous=True, region="us-west-2", endpoint_override="https://data.source.coop")
gdf = gpd.read_parquet("humane-intelligence/bias-bounty-mapping-equity-challenge/…", filesystem=fs)
```

DuckDB gets there with `SET s3_endpoint='data.source.coop';` (keep `s3_url_style='path'`), and the
aws CLI with `--endpoint-url https://data.source.coop --no-sign-request`.

Note it's the **endpoint**, not the path. Writing `s3://data.source.coop/…` as a plain URI won't
work: the client resolves `data.source.coop` as a bucket name on real AWS, so the request never
reaches the proxy at all.

**The `https://` URI as a path doesn't work in Python either.** `gpd.read_parquet("https://...")`
raises `ArrowInvalid: Unrecognized filesystem type in URI` (pyarrow has no `https` driver);
`pd.read_parquet("https://...")` and `gpd.read_file("https://...")` return **403**, because those
paths probe the URL with Python's default `urllib` User-Agent and the proxy rejects it. Neither is
anything you did wrong: use one of the two `s3://` forms above for parquet, and the `/vsicurl/`
prefix (or `pyogrio.read_dataframe()`) for the GeoJSON.

Every cell in that table is checked by [`scripts/readme_addresses.py`](scripts/readme_addresses.py)
if you want to re-run it on your own machine (`uv run scripts/readme_addresses.py`, no credentials
needed).

### Overture's nested columns

The Overture layers do **not** have flat attributes, and this is the first thing that trips people
up. `names` and `categories` are **structs**; `sources` is a **list of structs**.

In DuckDB, reach into structs with dot access. `names.common` is a `MAP`, so it takes a key, not an
index:

```python
con.sql("""
    SELECT names.primary        AS name,
           names.common['en']   AS name_en,
           categories.primary   AS category,
           categories.alternate AS also,          -- a list
           confidence
    FROM 's3://us-west-2.opendata.source.coop/humane-intelligence/bias-bounty-mapping-equity-challenge/reference/northern-ca/northern-ca-overture-pois.parquet'
    WHERE names.primary IS NOT NULL
    LIMIT 5
""")
```

`sources` is a list, so index it (`sources[1]`, 1-based) or `unnest` it. Unnesting tells you **which
upstream dataset actually supplied each feature**, which is close to the heart of this challenge: a
region built mostly from ML-derived footprints is a different coverage story from one built on
OpenStreetMap.

```python
con.sql("""
    SELECT s.dataset,
           count(*)                                         AS buildings,
           round(100.0 * count(s.confidence) / count(*), 1) AS pct_with_confidence,
           round(avg(s.confidence), 3)                      AS avg_confidence
    FROM 's3://us-west-2.opendata.source.coop/humane-intelligence/bias-bounty-mapping-equity-challenge/reference/northern-ca/northern-ca-overture-buildings.parquet',
         unnest(sources) AS t(s)
    GROUP BY 1
    ORDER BY buildings DESC
""")
```

`confidence` is **sparse**: only some upstream datasets populate it (Microsoft ML on a minority of
its rows, Google Open Buildings on all of theirs, OpenStreetMap never). Treat NULL as "not
reported", not as "low confidence".

In **GeoPandas** the same structs arrive as plain dicts, and lists of structs as arrays of dicts.
The `.str` accessor indexes both, despite the name, and it chains:

```python
pois["name"]      = pois["names"].str["primary"]
pois["category"]  = pois["categories"].str["primary"]
pois["n_sources"] = pois["sources"].str.len()
pois["first_ds"]  = pois["sources"].str[0].str["dataset"]

# explode is the pandas equivalent of DuckDB's unnest, and it keeps the geometry
pois.explode("sources")["sources"].str["dataset"].value_counts()
```

With **pandas** you can ask for Arrow-backed columns instead and get real typed accessors
(`dtype_backend` is a pandas argument; `gpd.read_parquet` does not accept it):

```python
p = pd.read_parquet(URL, columns=["names", "sources"], dtype_backend="pyarrow")

p["names"].struct.field("primary")
p["sources"].list.len()
p["sources"].list[0].struct.field("dataset")
```

### Web preview

Clicking any `.parquet` file in the file browser opens a table preview of that file. It works on
**one file at a time** (it range-reads row groups, so even multi-GB files preview fine) and does
not overlay layers or draw a map.

## Licensing / attribution

- **Overture Maps**: per Overture's per-theme licenses (buildings and transportation, so the
  `buildings` / `roads` / `rail` layers: **ODbL**; places and base, so `pois` / `infrastructure`:
  **CDLA Permissive 2.0**). Attribute © Overture Maps Foundation and its sources (incl.
  OpenStreetMap contributors).
- **Microsoft Building Footprints**: Open Data Commons **ODbL**.
- **US Census (ACS, TIGER, CBP)**: public domain (US Government work).
- **HIFLD (USGS National Map structures)**: public domain (US Government work).

## Links

- Challenge: Bias Bounties @ Scale (Zindi): https://zindi.world/competitions/bias-bounty-mapping-equity-challenge
- Tutorial notebook (download → open → explore, per layer): https://github.com/kentstephen/bias-bounty-map-tutorial
  ([open in Colab](https://colab.research.google.com/github/kentstephen/bias-bounty-map-tutorial/blob/main/bias-bounty-explore-tutorial.ipynb))
- Overture data docs: https://docs.overturemaps.org/
