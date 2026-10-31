#!/usr/bin/env python3
"""
100% Official Challenge-Package Feature Extractor (DuckDB + Source Cooperative)
================================================================================
Extracts tract-level Overture Maps (2026-08-19.0) semantic provenance, building
typology, POI taxonomy/sibling signals, and census-tract water/geometry features
strictly from the live official challenge package:
  s3://us-west-2.opendata.source.coop/humane-intelligence/bias-bounty-mapping-equity-challenge

Zero withdrawn reference datasets (NO TIGER roads, NO Microsoft Buildings reference,
NO HIFLD, NO Census CBP) and zero external URLs are accessed.
"""

from __future__ import annotations
import os
import sys
import time
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
S3_BASE = (
    "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/"
    "humane-intelligence/bias-bounty-mapping-equity-challenge"
)

REGIONS = [
    "eastern-wa",
    "northern-ca",
    "eastern-ok",
    "maricopa-az",
    "south-central-tx",
]


def extract_region_overture_semantic(con: duckdb.DuckDBPyConnection, region: str) -> pd.DataFrame:
    t0 = time.time()
    tracts_url = f"{S3_BASE}/strata/{region}/{region}-census-tracts.parquet"
    bldg_url = f"{S3_BASE}/reference/{region}/{region}-overture-buildings.parquet"
    pois_url = f"{S3_BASE}/reference/{region}/{region}-overture-pois.parquet"

    # Load official census tract polygons + 0.05-deg (~5 km) spatial hash grid for O(N) hash-join
    con.execute("DROP TABLE IF EXISTS reg_tracts; DROP TABLE IF EXISTS reg_tract_grid;")
    con.execute(
        f"""
        CREATE TEMP TABLE reg_tracts AS
        SELECT
            GEOID,
            NAMELSAD,
            CAST(ALAND AS DOUBLE) AS tract_aland_m2,
            ST_Area_Spheroid(geometry) / 1e6 AS tract_total_area_km2,
            bbox.xmin AS t_xmin,
            bbox.xmax AS t_xmax,
            bbox.ymin AS t_ymin,
            bbox.ymax AS t_ymax,
            geometry
        FROM read_parquet('{tracts_url}');

        CREATE TEMP TABLE reg_tract_grid AS
        SELECT
            t.GEOID,
            t.t_xmin, t.t_xmax, t.t_ymin, t.t_ymax,
            t.geometry,
            gx.grid_x,
            gy.grid_y
        FROM reg_tracts t,
             generate_series(CAST(floor(t.t_xmin * 20.0) AS BIGINT), CAST(floor(t.t_xmax * 20.0) AS BIGINT)) AS gx(grid_x),
             generate_series(CAST(floor(t.t_ymin * 20.0) AS BIGINT), CAST(floor(t.t_ymax * 20.0) AS BIGINT)) AS gy(grid_y);
        """
    )
    n_tracts = con.execute("SELECT count(*) FROM reg_tracts").fetchone()[0]

    # 1. Extract Overture Buildings provenance & semantic typing (projected columns only, no WKB download!)
    bldg_sql = f"""
    WITH raw_b AS (
        SELECT
            (bbox.xmin + bbox.xmax) * 0.5 AS cx,
            (bbox.ymin + bbox.ymax) * 0.5 AS cy,
            CAST(floor(((bbox.xmin + bbox.xmax) * 0.5) * 20.0) AS BIGINT) AS grid_x,
            CAST(floor(((bbox.ymin + bbox.ymax) * 0.5) * 20.0) AS BIGINT) AS grid_y,
            ((bbox.xmax - bbox.xmin) * 111320.0 * cos(radians((bbox.ymin + bbox.ymax) * 0.5)))
              * ((bbox.ymax - bbox.ymin) * 110574.0) AS bbox_area_m2,
            length(sources) AS n_src,
            sources[1].dataset AS ds1,
            list_transform(sources, x -> x.dataset) AS ds_list,
            subtype,
            "class" AS bclass,
            height,
            num_floors,
            names.primary AS bname
        FROM read_parquet('{bldg_url}')
    ),
    joined_b AS (
        SELECT
            t.GEOID,
            b.*
        FROM raw_b b
        JOIN reg_tract_grid t
          ON b.grid_x = t.grid_x
         AND b.grid_y = t.grid_y
         AND b.cx BETWEEN t.t_xmin AND t.t_xmax
         AND b.cy BETWEEN t.t_ymin AND t.t_ymax
         AND ST_Within(ST_Point(b.cx, b.cy), t.geometry)
    )
    SELECT
        GEOID,
        count(*) AS ovt_bldg_bbox_cnt,
        sum(CASE WHEN n_src = 1 AND ds1 = 'Microsoft ML Buildings' THEN 1 ELSE 0 END) AS ovt_bldg_ms_only,
        sum(CASE WHEN list_contains(ds_list, 'Microsoft ML Buildings') THEN 1 ELSE 0 END) AS ovt_bldg_ms_any,
        sum(CASE WHEN list_contains(ds_list, 'OpenStreetMap') THEN 1 ELSE 0 END) AS ovt_bldg_osm_any,
        sum(CASE WHEN list_contains(ds_list, 'OpenStreetMap') AND NOT list_contains(ds_list, 'Microsoft ML Buildings') THEN 1 ELSE 0 END) AS ovt_bldg_osm_only,
        sum(CASE WHEN list_contains(ds_list, 'Esri Community Maps') THEN 1 ELSE 0 END) AS ovt_bldg_esri_any,
        sum(CASE WHEN n_src >= 2 THEN 1 ELSE 0 END) AS ovt_bldg_multisrc,
        sum(CASE WHEN subtype IS NOT NULL OR bclass IS NOT NULL THEN 1 ELSE 0 END) AS ovt_bldg_typed,
        sum(CASE WHEN height IS NOT NULL OR num_floors IS NOT NULL THEN 1 ELSE 0 END) AS ovt_bldg_with_height,
        sum(CASE WHEN bname IS NOT NULL THEN 1 ELSE 0 END) AS ovt_bldg_with_name,
        sum(CASE WHEN subtype = 'residential' THEN 1 ELSE 0 END) AS ovt_bldg_residential,
        sum(CASE WHEN bclass IN ('apartments', 'dormitory') OR coalesce(num_floors, 0) >= 3 THEN 1 ELSE 0 END) AS ovt_bldg_apartments,
        sum(CASE WHEN subtype IN ('commercial', 'industrial') THEN 1 ELSE 0 END) AS ovt_bldg_comm_ind,
        sum(CASE WHEN subtype IN ('civic', 'education', 'medical', 'religious')
                   OR bclass IN ('school', 'fire_station', 'hospital', 'civic', 'public', 'government', 'university', 'college', 'kindergarten')
                 THEN 1 ELSE 0 END) AS ovt_bldg_civic_all,
        sum(CASE WHEN bclass IN ('school', 'kindergarten') OR subtype = 'education' THEN 1 ELSE 0 END) AS ovt_bldg_school_typed,
        sum(CASE WHEN bclass = 'fire_station' THEN 1 ELSE 0 END) AS ovt_bldg_fire_typed,
        sum(CASE WHEN subtype = 'outbuilding' OR bclass IN ('shed', 'garage', 'garages', 'barn', 'carport', 'roof') THEN 1 ELSE 0 END) AS ovt_bldg_outbuilding,
        sum(CASE WHEN bbox_area_m2 >= 400.0 THEN 1 ELSE 0 END) AS ovt_bldg_large_400m2,
        sum(CASE WHEN bbox_area_m2 >= 1200.0 THEN 1 ELSE 0 END) AS ovt_bldg_xlarge_1200m2,
        coalesce(sum(bbox_area_m2), 0.0) AS ovt_bldg_bbox_area_m2_sum
    FROM joined_b
    GROUP BY GEOID;
    """
    df_b = con.execute(bldg_sql).df()
    t1 = time.time()

    # 2. Extract Overture POIs semantic taxonomy, hospital exclusion, and sibling emergency/school signals
    poi_sql = f"""
    WITH raw_p AS (
        SELECT
            (bbox.xmin + bbox.xmax) * 0.5 AS cx,
            (bbox.ymin + bbox.ymax) * 0.5 AS cy,
            CAST(floor(((bbox.xmin + bbox.xmax) * 0.5) * 20.0) AS BIGINT) AS grid_x,
            CAST(floor(((bbox.ymin + bbox.ymax) * 0.5) * 20.0) AS BIGINT) AS grid_y,
            categories.primary AS pcat,
            basic_category AS bcat,
            coalesce(confidence, 0.0) AS conf,
            coalesce(operating_status, 'open') AS op_status,
            names.primary AS pname,
            length(sources) AS n_src,
            sources[1].dataset AS ds1
        FROM read_parquet('{pois_url}')
    ),
    joined_p AS (
        SELECT
            t.GEOID,
            p.*
        FROM raw_p p
        JOIN reg_tract_grid t
          ON p.grid_x = t.grid_x
         AND p.grid_y = t.grid_y
         AND p.cx BETWEEN t.t_xmin AND t.t_xmax
         AND p.cy BETWEEN t.t_ymin AND t.t_ymax
         AND ST_Within(ST_Point(p.cx, p.cy), t.geometry)
    )
    SELECT
        GEOID,
        count(*) AS ovt_poi_bbox_cnt,
        sum(CASE WHEN pcat IS DISTINCT FROM 'hospital' THEN 1 ELSE 0 END) AS ovt_poi_ex_hospital,
        sum(CASE WHEN pcat = 'hospital' THEN 1 ELSE 0 END) AS ovt_poi_hospital,
        sum(CASE WHEN pcat = 'fire_department' THEN 1 ELSE 0 END) AS ovt_poi_fire_exact,
        sum(CASE WHEN pcat = 'fire_protection_service'
                   OR (pcat IS DISTINCT FROM 'fire_department' AND (
                       pname ILIKE '%fire dept%' OR pname ILIKE '%fire department%'
                       OR pname ILIKE '%fire station%' OR pname ILIKE '%volunteer fire%'
                       OR pname ILIKE '%fire district%' OR pname ILIKE '%vfd%'
                   )) THEN 1 ELSE 0 END) AS ovt_poi_fire_sibling,
        sum(CASE WHEN pcat = 'ambulance_and_ems_services' THEN 1 ELSE 0 END) AS ovt_poi_ems_exact,
        sum(CASE WHEN pcat IN ('emergency_room', 'urgent_care_clinic')
                   OR (pcat IS DISTINCT FROM 'ambulance_and_ems_services' AND (
                       pname ILIKE '%ambulance%' OR pname ILIKE '%emergency medical%'
                       OR pname ILIKE '%paramedic%' OR pname ILIKE '%rescue squad%'
                       OR pname ILIKE '%ems%'
                   )) THEN 1 ELSE 0 END) AS ovt_poi_ems_sibling,
        sum(CASE WHEN pcat IN ('school', 'elementary_school', 'middle_school', 'high_school',
                               'public_school', 'private_school', 'preschool', 'kindergarten')
                 THEN 1 ELSE 0 END) AS ovt_poi_school_exact,
        sum(CASE WHEN pcat IN ('day_care_preschool', 'religious_school', 'vocational_and_technical_school',
                               'school_district_offices', 'college_university', 'community_college', 'specialty_school')
                   OR (coalesce(pcat, '') NOT IN ('school', 'elementary_school', 'middle_school', 'high_school',
                                                  'public_school', 'private_school', 'preschool', 'kindergarten')
                       AND (pname ILIKE '%elementary%' OR pname ILIKE '%high school%'
                            OR pname ILIKE '%middle school%' OR pname ILIKE '%junior high%'
                            OR pname ILIKE '%independent school%' OR pname ILIKE '%school district%'))
                 THEN 1 ELSE 0 END) AS ovt_poi_school_sibling,
        sum(CASE WHEN pcat IN ('police_department', 'law_enforcement', 'government_services', 'post_office', 'library', 'town_hall', 'community_center')
                 THEN 1 ELSE 0 END) AS ovt_poi_civic_anchor,
        sum(CASE WHEN op_status = 'open' AND conf >= 0.75 THEN 1 ELSE 0 END) AS ovt_poi_verified_open,
        sum(CASE WHEN n_src = 1 AND ds1 ILIKE '%meta%' THEN 1 ELSE 0 END) AS ovt_poi_meta_only
    FROM joined_p
    GROUP BY GEOID;
    """
    df_p = con.execute(poi_sql).df()
    t2 = time.time()

    df_t = con.execute(
        "SELECT GEOID, NAMELSAD, tract_aland_m2, tract_total_area_km2 FROM reg_tracts"
    ).df()

    out = df_t.merge(df_b, on="GEOID", how="left").merge(df_p, on="GEOID", how="left")
    num_cols = [c for c in out.columns if c not in ("GEOID", "NAMELSAD")]
    out[num_cols] = out[num_cols].fillna(0)
    out["region"] = region
    out["tract_water_area_km2"] = np.maximum(0.0, out["tract_total_area_km2"] - out["tract_aland_m2"] / 1e6)
    out["tract_water_frac"] = np.clip(
        out["tract_water_area_km2"] / np.maximum(out["tract_total_area_km2"], 1e-9),
        0.0,
        1.0,
    )

    print(
        f"[{region:18s}] tracts={n_tracts:4d} | bldg={t1-t0:5.1f}s | poi={t2-t1:5.1f}s | "
        f"ms_only_share={out['ovt_bldg_ms_only'].sum()/max(out['ovt_bldg_bbox_cnt'].sum(),1):.3f} | "
        f"typed_share={out['ovt_bldg_typed'].sum()/max(out['ovt_bldg_bbox_cnt'].sum(),1):.3f} | "
        f"fire_sib={int(out['ovt_poi_fire_sibling'].sum())} | sch_sib={int(out['ovt_poi_school_sibling'].sum())}",
        flush=True,
    )
    return out


def main() -> None:
    t_all = time.time()
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial;")

    frames = []
    for reg in REGIONS:
        frames.append(extract_region_overture_semantic(con, reg))

    full_df = pd.concat(frames, ignore_index=True)

    # Align to exact 9,794 SampleSubmission.csv rows by (GEOID, region) / GEOID
    ss = pd.read_csv(ROOT / "SampleSubmission.csv", dtype={"GEOID": str})
    ss["GEOID"] = ss["GEOID"].astype(str).str.strip().str.zfill(11)

    aligned = ss[["GEOID", "region"]].merge(
        full_df, on=["GEOID", "region"], how="left"
    )
    # Fallback for any cross-region GEOID if needed
    miss = aligned["tract_total_area_km2"].isna()
    if miss.any():
        dedup = full_df.drop_duplicates("GEOID").drop(columns=["region"])
        fill_df = aligned.loc[miss, ["GEOID", "region"]].merge(dedup, on="GEOID", how="left")
        aligned.loc[miss, fill_df.columns] = fill_df.values

    assert len(aligned) == 9794, f"Expected 9,794 rows, got {len(aligned)}"
    out_path = ROOT / "challenge_package_rich_features_9794.parquet"
    aligned.to_parquet(out_path, index=False)
    print(f"\n[DONE] Saved {out_path} ({aligned.shape[0]}x{aligned.shape[1]}) in {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
