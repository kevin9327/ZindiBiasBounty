#!/usr/bin/env python3
"""
100% Official 2026-09-25 Challenge-Package Deep Geometry & Spatial-Cluster Extractor (v9)
=========================================================================================
Competition : Zindi — Humane Intelligence Bias Bounty: Mapping Equity Challenge
Author      : Byunggwon So (@kevin9327ai)
Repository  : https://github.com/kevin9327/ZindiBiasBounty

STRICT RULE & PROVENANCE COMPLIANCE
-----------------------------------
Queries ONLY the official 2026-09-25 challenge package on Source Cooperative:
  s3://us-west-2.opendata.source.coop/humane-intelligence/bias-bounty-mapping-equity-challenge
    - `strata/<region>/<region>-census-tracts.parquet`
    - `reference/<region>/<region>-overture-roads.parquet`
    - `reference/<region>/<region>-overture-pois.parquet`
    - `reference/<region>/<region>-overture-buildings.parquet`
    - `reference/<region>/<region>-overture-infrastructure.parquet`
    - `SampleSubmission.csv` (9,794 scored tracts across 5 regions)

ZERO withdrawn reference layers (NO TIGER roads, NO standalone Microsoft US Building
Footprints reference, NO HIFLD/USGS points, NO Census CBP), ZERO external datasets,
and ZERO reconstructed target files are accessed.
"""

from __future__ import annotations
import sys
import time
from pathlib import Path
from typing import List, Tuple

import duckdb
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

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

# Core Interstate, US Highway, and State Highway networks (excluding Texas FM/RM/Park sub-networks)
CORE_TIGER_NET_SQL = """
list_any_value(list_filter(coalesce(routes, []), rt ->
    coalesce(rt.network, '') IN (
        'US:I', 'US:I:Business', 'US:I:Business:Loop', 'US:I:Business:Spur', 'US:I:Downtown:Loop',
        'US:US', 'US:US:Business', 'US:US:Alternate', 'US:US:Bypass', 'US:US:Truck', 'US:US:Spur',
        'US:CA', 'US:CA:Business',
        'US:AZ', 'US:AZ:Loop', 'US:AZ:Business', 'US:AZ:Spur',
        'US:OK', 'US:OK:Turnpike', 'US:OK:Business', 'US:OK:Spur', 'US:OK:Loop',
        'US:TX', 'US:TX:Loop', 'US:TX:Spur', 'US:TX:Business', 'US:TX:Beltway', 'US:TX:Toll', 'Texas Trunk System',
        'US:WA', 'US:WA:Business', 'US:WA:Spur',
        'US:NM', 'US:BIA'
    )
)) IS NOT NULL
"""

# Texas Farm-to-Market / Ranch-to-Market / Park Road networks (tracked separately!)
TX_FM_RM_NET_SQL = """
list_any_value(list_filter(coalesce(routes, []), rt ->
    coalesce(rt.network, '') IN ('US:TX:FM', 'US:TX:RM', 'US:TX:RR', 'US:TX:Park', 'US:TX:PA', 'US:TX:NASA')
)) IS NOT NULL
"""

# County route networks (S1400 in most counties, occasionally S1200 in rural sections)
COUNTY_NET_SQL = """
list_any_value(list_filter(coalesce(routes, []), rt ->
    (rt.network IS NOT NULL AND starts_with(rt.network, 'US:')
     AND NOT starts_with(rt.network, 'US:NHT')
     AND NOT starts_with(rt.network, 'US:NFSR')
     AND NOT starts_with(rt.network, 'US:AB:')
     AND rt.network != 'US:US:Historic'
     AND rt.network != 'US:I:Future')
)) IS NOT NULL
"""

# Core numbered Interstate / US / State Highway name regex (excluding FM/RM)
CORE_HWY_NAME_REGEX = (
    r"(?i)(\bI-\s*\d+|\bIH[-\s]+\d+|\bInterstate\s+\d+"
    r"|\bUS[-\s]+\d+|\bU\.?S\.?\s*(Hwy|Highway|Rte|Route)[-\s]*\d+"
    r"|\bState\s+(Hwy|Highway|Rte|Route|Loop|Spur)[-\s]*\d+"
    r"|\b(SH|SR|SL|SS)[-\s]+\d+\b"
    r"|\b(AZ|CA|OK|TX|WA|NM)[-\s]+\d+\b"
    r"|\bSpur\s+\d+\b|\bHwy\s+\d+|\bHighway\s+\d+\b)"
)

FM_RM_NAME_REGEX = (
    r"(?i)(\b(FM|RM|RR)[-\s]+\d+\b"
    r"|\bFarm[- ]to[- ]Market(\s+Rd|\s+Road)?\s*\d+"
    r"|\bRanch[- ]to[- ]Market(\s+Rd|\s+Road)?\s*\d+"
    r"|\bRanch\s+Road\s+\d+\b)"
)

SCHOOL_STRICT6_SQL = "('elementary_school', 'middle_school', 'high_school', 'school', 'private_school', 'public_school')"

FIRE_TRUE_NAME_REGEX = (
    r"(?i)(\bvolunteer\s+fire|\bv\.?f\.?d\.?\b|\bfire\s*dept|\bfire\s+department|\bfiredept\b"
    r"|\bfire\s+station|\bfire\s+district|\bfire\s+protection\s+district"
    r"|\bfire\s*(&|and|/)\s*(rescue|medical|ems)|\bfire\s+rescue|\bfire\s+company|\bhose\s+company"
    r"|\bhook\s*(&|and)\s*ladder|\bcal\s*fire|\brural[- /]*metro\s+fire|\brural\s+fire|\btribal\s+fire"
    r"|\bstation\s*#?\s*\d+\s*fire|\bfire\s+no\.?\s*\d+)"
)
FIRE_COMMERCIAL_EXCLUDE_REGEX = (
    r"(?i)\b(extinguisher|sprinkler|alarm|equipment|sales|supply|supplies|llc|inc|corp"
    r"|systems|pros|inspection|inspections|clothing|apparel|subs|grill|cafe|restaurant"
    r"|pizza|brewing|brewery|bar|lounge|insurance|restoration|damage|fireworks|fireplace"
    r"|stove|hearth|tire|auto|museum|memorial|park|campground|hydrant|marshal\s+office"
    r"|protection\s+(co|company|inc|llc|corp|service|services|systems)|western\s+states\s+fire"
    r"|training\s+center|motor\s+vehicle|water\s+(treatment|reclamation))\b"
)

EMS_TRUE_NAME_REGEX = (
    r"(?i)\b(ambulance\s+service|ambulance\s+station|ems\s+station|county\s+ems|city\s+ems"
    r"|emergency\s+medical\s+service|emergency\s+medical\s+services|rescue\s+squad|life\s*flight|medstar)\b"
)
EMS_COMMERCIAL_EXCLUDE_REGEX = (
    r"(?i)\b(billing|training|academy|school|supply|equipment|sales|llc|inc|repair|insurance|veterinary|pet)\b"
)

K12_TRUE_NAME_REGEX = (
    r"(?i)\b(elementary\s+school|middle\s+school|junior\s+high(\s+school)?|high\s+school"
    r"|intermediate\s+school|primary\s+school|grade\s+school|charter\s+school"
    r"|catholic\s+school|christian\s+school|lutheran\s+school|adventist\s+school"
    r"|preparatory\s+school|prep\s+school|consolidated\s+school|indian\s+mission\s+school"
    r"|k-8\s+school|k-12\s+school)\b"
)
K12_EXCLUDE_REGEX = (
    r"(?i)\b(preschool|pre-school|pre\s+k|day\s*care|child\s*care|early\s+childhood|head\s+start"
    r"|driving|traffic|beauty|cosmetology|barber|dance|ballet|swim|music|piano|martial|karate|taekwondo|jiu"
    r"|flight|aviation|real\s+estate|dog|obedience|bartend|truck|cdl|welding|nursing|medical\s+school"
    r"|law\s+school|dental|seminary|university|college|stadium|fieldhouse|athletic\s+field|football\s+field"
    r"|bus\s+barn|administration\s+building|district\s+office|board\s+of\s+education"
    r"|school\s+district|district\s*#?\s*\d+|spiritshop|apparel|store|shop|accountability"
    r"|tutoring|school\s+of\s+rock|luthiery|asu\b)\b"
)


def cluster_points_greedy(coords: np.ndarray, radius_m: float) -> np.ndarray:
    """Deterministic greedy Euclidean spatial clustering in EPSG:5070 meters."""
    if len(coords) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    centers: List[np.ndarray] = []
    for pt in coords:
        if not centers:
            centers.append(pt)
            continue
        dists = np.linalg.norm(np.asarray(centers) - pt, axis=1)
        if float(np.min(dists)) > radius_m:
            centers.append(pt)
    return np.asarray(centers, dtype=np.float64)


def compute_spatial_unmapped_clusters(
    df_pts: pd.DataFrame,
    geoids: List[str],
    cluster_radius_m: float = 300.0,
    match_radius_m: float = 380.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Counts distinct physical facility sites per tract (`cluster_radius_m`) that lack any
    scored Overture primary-category pin within `match_radius_m` (even across tract borders).
    """
    idx_map = {g: i for i, g in enumerate(geoids)}
    n_tr = len(geoids)
    n_scored_cl = np.zeros(n_tr, dtype=np.float64)
    n_unmapped_cl = np.zeros(n_tr, dtype=np.float64)
    soft_unmapped = np.zeros(n_tr, dtype=np.float64)

    if df_pts.empty:
        return n_scored_cl, n_unmapped_cl, soft_unmapped

    scored_all = df_pts[df_pts["is_scored"] == 1]
    all_scored_xy = (
        scored_all[["x5070", "y5070"]].values.astype(np.float64)
        if not scored_all.empty
        else np.zeros((0, 2), dtype=np.float64)
    )

    for g, grp in df_pts.groupby("GEOID", sort=False):
        ti = idx_map.get(g)
        if ti is None:
            continue
        sc_g = grp[grp["is_scored"] == 1][["x5070", "y5070"]].values.astype(np.float64)
        if len(sc_g) > 0:
            n_scored_cl[ti] = float(len(cluster_points_greedy(sc_g, cluster_radius_m)))

        un_g = grp[grp["is_scored"] == 0].sort_values("weight", ascending=False)
        if un_g.empty:
            continue
        un_xy = un_g[["x5070", "y5070"]].values.astype(np.float64)
        un_w = un_g["weight"].values.astype(np.float64)

        if len(all_scored_xy) > 0:
            d_to_sc = cdist(un_xy, all_scored_xy)
            far_mask = np.min(d_to_sc, axis=1) > match_radius_m
            un_xy = un_xy[far_mask]
            un_w = un_w[far_mask]

        if len(un_xy) == 0:
            continue

        kept_centers: List[np.ndarray] = []
        kept_weights: List[float] = []
        for pt, w in zip(un_xy, un_w):
            if not kept_centers:
                kept_centers.append(pt)
                kept_weights.append(float(w))
            else:
                dists = np.linalg.norm(np.asarray(kept_centers) - pt, axis=1)
                if float(np.min(dists)) > cluster_radius_m:
                    kept_centers.append(pt)
                    kept_weights.append(float(w))

        wts = np.asarray(kept_weights, dtype=np.float64)
        n_unmapped_cl[ti] = float(np.sum(wts >= 0.75))
        soft_unmapped[ti] = float(np.sum(wts))

    return n_scored_cl, n_unmapped_cl, soft_unmapped


def compute_true_k12_campuses(
    df_pts: pd.DataFrame,
    geoids: List[str],
    cluster_radius_m: float = 300.0,
) -> np.ndarray:
    """Counts distinct verified K-12 physical campuses (300m radius) per tract, excluding generic commercial 'school' pins."""
    idx_map = {g: i for i, g in enumerate(geoids)}
    n_true = np.zeros(len(geoids), dtype=np.float64)
    if df_pts.empty or "is_true_k12" not in df_pts.columns:
        return n_true
    sub = df_pts[df_pts["is_true_k12"] == 1]
    if sub.empty:
        return n_true
    for g, grp in sub.groupby("GEOID", sort=False):
        ti = idx_map.get(g)
        if ti is None:
            continue
        xy = grp[["x5070", "y5070"]].values.astype(np.float64)
        if len(xy) > 0:
            n_true[ti] = float(len(cluster_points_greedy(xy, cluster_radius_m)))
    return n_true


def extract_region_clusters_v9(
    con: duckdb.DuckDBPyConnection,
    region: str,
    geoids: List[str],
) -> pd.DataFrame:
    """Extracts high-precision 300m spatial facility clusters (Fire, EMS, K-12) assuming `tr` and `tr_grid` exist."""
    pois_url = f"{S3_BASE}/reference/{region}/{region}-overture-pois.parquet"
    bldg_url = f"{S3_BASE}/reference/{region}/{region}-overture-buildings.parquet"

    fire_pts_sql = f"""
        WITH poi_fire AS (
            SELECT
                p.geom4326,
                (bbox.xmin + bbox.xmax) * 0.5 AS cx,
                (bbox.ymin + bbox.ymax) * 0.5 AS cy,
                CAST(floor(((bbox.xmin + bbox.xmax) * 0.5) * 20.0) AS BIGINT) AS gx,
                CAST(floor(((bbox.ymin + bbox.ymax) * 0.5) * 20.0) AS BIGINT) AS gy,
                CASE WHEN categories.primary = 'fire_department' THEN 1 ELSE 0 END AS is_scored,
                CASE
                    WHEN categories.primary = 'fire_department' THEN 1.0
                    WHEN regexp_matches(coalesce(names.primary, ''), '{FIRE_TRUE_NAME_REGEX}')
                         AND list_contains(coalesce(categories.alternate, []), 'fire_department') THEN 0.96
                    WHEN regexp_matches(coalesce(names.primary, ''), '{FIRE_TRUE_NAME_REGEX}')
                         AND coalesce(categories.primary, '') IN (
                             'fire_protection_service', 'public_service_and_government',
                             'central_government_office', 'government_services',
                             'community_services_non_profits', 'ambulance_and_ems_services',
                             'emergency_service', 'police_department', 'law_enforcement', ''
                         ) THEN 0.92
                    WHEN regexp_matches(coalesce(names.primary, ''), '{FIRE_TRUE_NAME_REGEX}') THEN 0.84
                    ELSE 0.0
                END AS weight
            FROM (SELECT *, geometry AS geom4326 FROM read_parquet('{pois_url}')) p
            WHERE categories.primary = 'fire_department'
               OR (
                   regexp_matches(coalesce(names.primary, ''), '{FIRE_TRUE_NAME_REGEX}')
                   AND NOT regexp_matches(coalesce(names.primary, ''), '{FIRE_COMMERCIAL_EXCLUDE_REGEX}')
                   AND coalesce(basic_category, '') NOT IN (
                       'restaurant', 'retail', 'automotive', 'accommodation',
                       'entertainment', 'financial_service', 'beauty_and_spa'
                   )
               )
        ),
        bldg_fire AS (
            SELECT
                ST_Point((bbox.xmin + bbox.xmax) * 0.5, (bbox.ymin + bbox.ymax) * 0.5) AS geom4326,
                (bbox.xmin + bbox.xmax) * 0.5 AS cx,
                (bbox.ymin + bbox.ymax) * 0.5 AS cy,
                CAST(floor(((bbox.xmin + bbox.xmax) * 0.5) * 20.0) AS BIGINT) AS gx,
                CAST(floor(((bbox.ymin + bbox.ymax) * 0.5) * 20.0) AS BIGINT) AS gy,
                0 AS is_scored,
                CASE
                    WHEN "class" = 'fire_station' AND regexp_matches(coalesce(names.primary, ''), '{FIRE_TRUE_NAME_REGEX}') THEN 0.96
                    WHEN "class" = 'fire_station' THEN 0.90
                    ELSE 0.88
                END AS weight
            FROM read_parquet('{bldg_url}')
            WHERE (
                (
                    "class" = 'fire_station'
                    AND ((bbox.xmax - bbox.xmin) * 111320.0 * cos(radians((bbox.ymin + bbox.ymax) * 0.5)))
                      * ((bbox.ymax - bbox.ymin) * 110574.0) >= 80.0
                )
                OR (
                    names.primary IS NOT NULL
                    AND regexp_matches(names.primary, '{FIRE_TRUE_NAME_REGEX}')
                )
            )
            AND NOT regexp_matches(coalesce(names.primary, ''), '{FIRE_COMMERCIAL_EXCLUDE_REGEX}')
        ),
        all_f AS (
            SELECT * FROM poi_fire WHERE weight > 0
            UNION ALL
            SELECT * FROM bldg_fire
        )
        SELECT
            t.GEOID,
            ST_X(ST_Transform(ST_Point(f.cx, f.cy), 'EPSG:4326', 'EPSG:5070', always_xy := true)) AS x5070,
            ST_Y(ST_Transform(ST_Point(f.cx, f.cy), 'EPSG:4326', 'EPSG:5070', always_xy := true)) AS y5070,
            f.is_scored,
            f.weight
        FROM all_f f
        JOIN tr_grid tg
          ON f.gx = tg.gx AND f.gy = tg.gy
         AND f.cx BETWEEN tg.xmin_pad AND tg.xmax_pad
         AND f.cy BETWEEN tg.ymin_pad AND tg.ymax_pad
        JOIN tr t ON tg.GEOID = t.GEOID
        WHERE ST_Intersects(f.geom4326, t.geom4326);
    """
    df_fire_pts = con.execute(fire_pts_sql).df()
    fire_sc_cl, fire_unm_cl, fire_unm_soft = compute_spatial_unmapped_clusters(
        df_fire_pts, geoids, cluster_radius_m=300.0, match_radius_m=380.0
    )

    ems_pts_sql = f"""
        WITH poi_ems AS (
            SELECT
                geometry AS geom4326,
                (bbox.xmin + bbox.xmax) * 0.5 AS cx,
                (bbox.ymin + bbox.ymax) * 0.5 AS cy,
                CAST(floor(((bbox.xmin + bbox.xmax) * 0.5) * 20.0) AS BIGINT) AS gx,
                CAST(floor(((bbox.ymin + bbox.ymax) * 0.5) * 20.0) AS BIGINT) AS gy,
                CASE WHEN categories.primary = 'ambulance_and_ems_services' THEN 1 ELSE 0 END AS is_scored,
                CASE
                    WHEN categories.primary = 'ambulance_and_ems_services' THEN 1.0
                    WHEN regexp_matches(coalesce(names.primary, ''), '{EMS_TRUE_NAME_REGEX}')
                         AND list_contains(coalesce(categories.alternate, []), 'ambulance_and_ems_services') THEN 0.92
                    WHEN regexp_matches(coalesce(names.primary, ''), '{EMS_TRUE_NAME_REGEX}') THEN 0.84
                    ELSE 0.0
                END AS weight
            FROM read_parquet('{pois_url}')
            WHERE categories.primary = 'ambulance_and_ems_services'
               OR (
                   regexp_matches(coalesce(names.primary, ''), '{EMS_TRUE_NAME_REGEX}')
                   AND NOT regexp_matches(coalesce(names.primary, ''), '{EMS_COMMERCIAL_EXCLUDE_REGEX}')
                   AND coalesce(categories.primary, '') NOT IN ('fire_department', 'fire_protection_service')
               )
        ),
        bldg_ems AS (
            SELECT
                ST_Point((bbox.xmin + bbox.xmax) * 0.5, (bbox.ymin + bbox.ymax) * 0.5) AS geom4326,
                (bbox.xmin + bbox.xmax) * 0.5 AS cx,
                (bbox.ymin + bbox.ymax) * 0.5 AS cy,
                CAST(floor(((bbox.xmin + bbox.xmax) * 0.5) * 20.0) AS BIGINT) AS gx,
                CAST(floor(((bbox.ymin + bbox.ymax) * 0.5) * 20.0) AS BIGINT) AS gy,
                0 AS is_scored,
                0.85 AS weight
            FROM read_parquet('{bldg_url}')
            WHERE names.primary IS NOT NULL
              AND regexp_matches(names.primary, '{EMS_TRUE_NAME_REGEX}')
              AND NOT regexp_matches(names.primary, '{EMS_COMMERCIAL_EXCLUDE_REGEX}')
        ),
        all_e AS (
            SELECT * FROM poi_ems WHERE weight > 0
            UNION ALL
            SELECT * FROM bldg_ems
        )
        SELECT
            t.GEOID,
            ST_X(ST_Transform(ST_Point(e.cx, e.cy), 'EPSG:4326', 'EPSG:5070', always_xy := true)) AS x5070,
            ST_Y(ST_Transform(ST_Point(e.cx, e.cy), 'EPSG:4326', 'EPSG:5070', always_xy := true)) AS y5070,
            e.is_scored,
            e.weight
        FROM all_e e
        JOIN tr_grid tg
          ON e.gx = tg.gx AND e.gy = tg.gy
         AND e.cx BETWEEN tg.xmin_pad AND tg.xmax_pad
         AND e.cy BETWEEN tg.ymin_pad AND tg.ymax_pad
        JOIN tr t ON tg.GEOID = t.GEOID
        WHERE ST_Intersects(e.geom4326, t.geom4326);
    """
    df_ems_pts = con.execute(ems_pts_sql).df()
    ems_sc_cl, ems_unm_cl, ems_unm_soft = compute_spatial_unmapped_clusters(
        df_ems_pts, geoids, cluster_radius_m=300.0, match_radius_m=380.0
    )

    univ_pts_sql = f"""
        WITH u_raw AS (
            SELECT
                (bbox.xmin + bbox.xmax) * 0.5 AS cx,
                (bbox.ymin + bbox.ymax) * 0.5 AS cy
            FROM read_parquet('{pois_url}')
            WHERE coalesce(categories.primary, '') IN ('college_university', 'community_college')
               OR (
                   regexp_matches(coalesce(names.primary, ''), '(?i)\\b(university|community\\s+college|junior\\s+college)\\b')
                   AND coalesce(categories.primary, '') IN ('education', 'campus_building', 'school')
               )
            UNION ALL
            SELECT
                (bbox.xmin + bbox.xmax) * 0.5 AS cx,
                (bbox.ymin + bbox.ymax) * 0.5 AS cy
            FROM read_parquet('{bldg_url}')
            WHERE "class" IN ('university', 'college')
        )
        SELECT
            ST_X(ST_Transform(ST_Point(cx, cy), 'EPSG:4326', 'EPSG:5070', always_xy := true)) AS x5070,
            ST_Y(ST_Transform(ST_Point(cx, cy), 'EPSG:4326', 'EPSG:5070', always_xy := true)) AS y5070
        FROM u_raw;
    """
    df_univ = con.execute(univ_pts_sql).df()
    univ_xy = (
        df_univ[["x5070", "y5070"]].values.astype(np.float64)
        if not df_univ.empty
        else np.zeros((0, 2), dtype=np.float64)
    )

    k12_pts_sql = f"""
        WITH poi_k12 AS (
            SELECT
                geometry AS geom4326,
                (bbox.xmin + bbox.xmax) * 0.5 AS cx,
                (bbox.ymin + bbox.ymax) * 0.5 AS cy,
                CAST(floor(((bbox.xmin + bbox.xmax) * 0.5) * 20.0) AS BIGINT) AS gx,
                CAST(floor(((bbox.ymin + bbox.ymax) * 0.5) * 20.0) AS BIGINT) AS gy,
                CASE WHEN categories.primary IN {SCHOOL_STRICT6_SQL} THEN 1 ELSE 0 END AS is_scored,
                CASE
                    WHEN NOT regexp_matches(coalesce(names.primary, ''), '{K12_EXCLUDE_REGEX}')
                         AND (
                             coalesce(categories.primary, '') IN (
                                 'elementary_school', 'middle_school', 'high_school',
                                 'private_school', 'public_school', 'charter_school', 'religious_school'
                             )
                             OR (
                                 coalesce(categories.primary, '') IN (
                                     'school', 'education', 'campus_building',
                                     'public_service_and_government', 'kindergarten', ''
                                 )
                                 AND regexp_matches(coalesce(names.primary, ''), '{K12_TRUE_NAME_REGEX}')
                             )
                         ) THEN 1
                    ELSE 0
                END AS is_true_k12,
                0 AS is_bldg,
                CASE WHEN regexp_matches(coalesce(names.primary, ''), '{K12_TRUE_NAME_REGEX}') THEN 1 ELSE 0 END AS has_k12_name,
                CASE
                    WHEN categories.primary IN {SCHOOL_STRICT6_SQL} THEN 1.0
                    WHEN coalesce(categories.primary, '') IN ('charter_school', 'religious_school') THEN 0.93
                    WHEN regexp_matches(coalesce(names.primary, ''), '{K12_TRUE_NAME_REGEX}')
                         AND coalesce(categories.primary, '') IN (
                             'education', 'campus_building', 'public_service_and_government', 'kindergarten', ''
                         ) THEN 0.90
                    WHEN coalesce(categories.primary, '') = 'education'
                         AND list_has_any(
                             coalesce(categories.alternate, []),
                             ['elementary_school', 'middle_school', 'high_school', 'private_school', 'public_school', 'charter_school', 'religious_school']
                         ) THEN 0.85
                    ELSE 0.0
                END AS weight
            FROM read_parquet('{pois_url}')
            WHERE categories.primary IN {SCHOOL_STRICT6_SQL}
               OR (
                   NOT regexp_matches(coalesce(names.primary, ''), '{K12_EXCLUDE_REGEX}')
                   AND coalesce(basic_category, '') NOT IN (
                       'restaurant', 'retail', 'automotive', 'accommodation', 'financial_service', 'beauty_and_spa'
                   )
                   AND (
                       coalesce(categories.primary, '') IN ('charter_school', 'religious_school')
                       OR (
                           regexp_matches(coalesce(names.primary, ''), '{K12_TRUE_NAME_REGEX}')
                           AND coalesce(categories.primary, '') IN (
                               'education', 'campus_building', 'public_service_and_government', 'kindergarten', ''
                           )
                       )
                       OR (
                           coalesce(categories.primary, '') = 'education'
                           AND list_has_any(
                               coalesce(categories.alternate, []),
                               ['elementary_school', 'middle_school', 'high_school', 'private_school', 'public_school', 'charter_school', 'religious_school']
                           )
                       )
                   )
               )
        ),
        bldg_k12 AS (
            SELECT
                ST_Point((bbox.xmin + bbox.xmax) * 0.5, (bbox.ymin + bbox.ymax) * 0.5) AS geom4326,
                (bbox.xmin + bbox.xmax) * 0.5 AS cx,
                (bbox.ymin + bbox.ymax) * 0.5 AS cy,
                CAST(floor(((bbox.xmin + bbox.xmax) * 0.5) * 20.0) AS BIGINT) AS gx,
                CAST(floor(((bbox.ymin + bbox.ymax) * 0.5) * 20.0) AS BIGINT) AS gy,
                0 AS is_scored,
                1 AS is_true_k12,
                1 AS is_bldg,
                CASE WHEN regexp_matches(coalesce(names.primary, ''), '{K12_TRUE_NAME_REGEX}') THEN 1 ELSE 0 END AS has_k12_name,
                CASE
                    WHEN "class" = 'school' AND regexp_matches(coalesce(names.primary, ''), '{K12_TRUE_NAME_REGEX}') THEN 0.94
                    WHEN "class" = 'school' THEN 0.84
                    ELSE 0.88
                END AS weight
            FROM read_parquet('{bldg_url}')
            WHERE (
                (
                    "class" = 'school'
                    AND ((bbox.xmax - bbox.xmin) * 111320.0 * cos(radians((bbox.ymin + bbox.ymax) * 0.5)))
                      * ((bbox.ymax - bbox.ymin) * 110574.0) >= 400.0
                )
                OR (
                    names.primary IS NOT NULL
                    AND regexp_matches(names.primary, '{K12_TRUE_NAME_REGEX}')
                )
            )
            AND NOT regexp_matches(coalesce(names.primary, ''), '{K12_EXCLUDE_REGEX}')
        ),
        all_s AS (
            SELECT * FROM poi_k12 WHERE weight > 0
            UNION ALL
            SELECT * FROM bldg_k12
        )
        SELECT
            t.GEOID,
            ST_X(ST_Transform(ST_Point(s.cx, s.cy), 'EPSG:4326', 'EPSG:5070', always_xy := true)) AS x5070,
            ST_Y(ST_Transform(ST_Point(s.cx, s.cy), 'EPSG:4326', 'EPSG:5070', always_xy := true)) AS y5070,
            s.is_scored,
            s.is_true_k12,
            s.is_bldg,
            s.has_k12_name,
            s.weight
        FROM all_s s
        JOIN tr_grid tg
          ON s.gx = tg.gx AND s.gy = tg.gy
         AND s.cx BETWEEN tg.xmin_pad AND tg.xmax_pad
         AND s.cy BETWEEN tg.ymin_pad AND tg.ymax_pad
        JOIN tr t ON tg.GEOID = t.GEOID
        WHERE ST_Intersects(s.geom4326, t.geom4326);
    """
    df_k12_pts = con.execute(k12_pts_sql).df()
    if not df_k12_pts.empty and len(univ_xy) > 0:
        unnamed_bldg_mask = (df_k12_pts["is_bldg"] == 1) & (df_k12_pts["has_k12_name"] == 0)
        if unnamed_bldg_mask.any():
            b_xy = df_k12_pts.loc[unnamed_bldg_mask, ["x5070", "y5070"]].values.astype(np.float64)
            near_univ = np.min(cdist(b_xy, univ_xy), axis=1) <= 450.0
            drop_idx = df_k12_pts.index[unnamed_bldg_mask][near_univ]
            if len(drop_idx) > 0:
                df_k12_pts = df_k12_pts.drop(index=drop_idx).reset_index(drop=True)

    k12_sc_cl, k12_unm_cl, k12_unm_soft = compute_spatial_unmapped_clusters(
        df_k12_pts, geoids, cluster_radius_m=300.0, match_radius_m=380.0
    )
    k12_true_cl = compute_true_k12_campuses(df_k12_pts, geoids, cluster_radius_m=300.0)

    return pd.DataFrame({
        "GEOID": geoids,
        "fire_scored_clusters_300m": fire_sc_cl,
        "fire_unmapped_clusters_300m": fire_unm_cl,
        "fire_unmapped_soft_300m": fire_unm_soft,
        "ems_scored_clusters_300m": ems_sc_cl,
        "ems_unmapped_clusters_300m": ems_unm_cl,
        "ems_unmapped_soft_300m": ems_unm_soft,
        "k12_scored_campuses_300m": k12_sc_cl,
        "k12_unmapped_campuses_300m": k12_unm_cl,
        "k12_unmapped_soft_300m": k12_unm_soft,
        "k12_true_campuses_300m": k12_true_cl,
    })


def extract_region_v9(con: duckdb.DuckDBPyConnection, region: str) -> pd.DataFrame:
    t0 = time.time()
    tracts_url = f"{S3_BASE}/strata/{region}/{region}-census-tracts.parquet"
    roads_url = f"{S3_BASE}/reference/{region}/{region}-overture-roads.parquet"
    pois_url = f"{S3_BASE}/reference/{region}/{region}-overture-pois.parquet"
    bldg_url = f"{S3_BASE}/reference/{region}/{region}-overture-buildings.parquet"
    infra_url = f"{S3_BASE}/reference/{region}/{region}-overture-infrastructure.parquet"

    con.execute("DROP TABLE IF EXISTS tr; DROP TABLE IF EXISTS tr_grid; DROP TABLE IF EXISTS rds; DROP TABLE IF EXISTS rds_grid;")

    # 1. Load tract polygons + precompute EPSG:5070 exact & 22m/38m buffered polygons + 0.05-deg hash grid
    con.execute(f"""
        CREATE TEMP TABLE tr AS
        WITH base AS (
            SELECT
                GEOID,
                bbox.xmin AS xmin, bbox.xmax AS xmax, bbox.ymin AS ymin, bbox.ymax AS ymax,
                geometry AS geom4326,
                ST_Transform(geometry, 'EPSG:4326', 'EPSG:5070', always_xy := true) AS geom5070
            FROM read_parquet('{tracts_url}')
        )
        SELECT
            GEOID, xmin, xmax, ymin, ymax, geom4326, geom5070,
            ST_Buffer(ST_SimplifyPreserveTopology(geom5070, 2.5), 22.0) AS geom_buf22,
            ST_Buffer(ST_SimplifyPreserveTopology(geom5070, 2.5), 38.0) AS geom_buf38,
            ST_Perimeter(ST_SimplifyPreserveTopology(geom5070, 2.5)) / 1000.0 AS tract_perim_km
        FROM base;

        CREATE TEMP TABLE tr_grid AS
        SELECT
            t.GEOID,
            t.xmin - 0.0006 AS xmin_pad,
            t.xmax + 0.0006 AS xmax_pad,
            t.ymin - 0.0006 AS ymin_pad,
            t.ymax + 0.0006 AS ymax_pad,
            gx.gx,
            gy.gy
        FROM tr t,
             generate_series(CAST(floor((t.xmin - 0.0006) * 20.0) AS BIGINT), CAST(floor((t.xmax + 0.0006) * 20.0) AS BIGINT)) AS gx(gx),
             generate_series(CAST(floor((t.ymin - 0.0006) * 20.0) AS BIGINT), CAST(floor((t.ymax + 0.0006) * 20.0) AS BIGINT)) AS gy(gy);
    """)

    geoids = con.execute("SELECT GEOID FROM tr ORDER BY GEOID").df()["GEOID"].tolist()

    # 2. Extract exact road & TIGER-highway proxy geometry (streaming bbox+ST_Intersects, zero disk spill!)
    con.execute(f"""
        CREATE TEMP TABLE rds AS
        SELECT
            "class" AS rclass,
            coalesce(subclass, '') AS rsub,
            ({CORE_TIGER_NET_SQL}) AS has_core_net,
            ({TX_FM_RM_NET_SQL}) AS has_fm_net,
            ({COUNTY_NET_SQL}) AS has_any_usnet,
            regexp_matches(coalesce(names.primary, ''), '{CORE_HWY_NAME_REGEX}') AS has_core_hwy_name,
            regexp_matches(coalesce(names.primary, ''), '{FM_RM_NAME_REGEX}') AS has_fm_name,
            regexp_matches(coalesce(names.primary, ''), '(?i)\\b(fire\\s*station|firehouse|volunteer\\s+fire|vfd)\\b') AS has_fire_rd,
            regexp_matches(coalesce(names.primary, ''), '(?i)\\b(high\\s+school|elementary\\s+school|middle\\s+school|school\\s+(rd|road|dr|drive|ln|lane|st|way))\\b') AS has_sch_rd,
            list_contains(flatten(coalesce(list_transform(road_flags, x -> x."values"), [])), 'is_link') AS flag_link,
            list_contains(flatten(coalesce(list_transform(road_flags, x -> x."values"), [])), 'is_divided') AS flag_div,
            bbox.xmin AS xmin, bbox.xmax AS xmax, bbox.ymin AS ymin, bbox.ymax AS ymax,
            ST_Transform(geometry, 'EPSG:4326', 'EPSG:5070', always_xy := true) AS geom5070
        FROM read_parquet('{roads_url}')
        WHERE "class" IN ('motorway', 'trunk', 'primary', 'secondary')
           OR ({CORE_TIGER_NET_SQL})
           OR ({TX_FM_RM_NET_SQL})
           OR ({COUNTY_NET_SQL})
           OR regexp_matches(coalesce(names.primary, ''), '{CORE_HWY_NAME_REGEX}')
           OR regexp_matches(coalesce(names.primary, ''), '{FM_RM_NAME_REGEX}')
           OR regexp_matches(coalesce(names.primary, ''), '(?i)\\b(fire\\s*station|firehouse|volunteer\\s+fire|vfd|high\\s+school|elementary\\s+school|middle\\s+school)\\b');
    """)

    df_roads = con.execute("""
        WITH cand AS (
            SELECT
                t.GEOID,
                r.rclass,
                (r.rsub = 'link' OR coalesce(r.flag_link, false)) AS is_lnk,
                coalesce(r.flag_div, false) AS is_div,
                (r.has_core_net OR r.has_core_hwy_name) AS is_core_hwy,
                (r.has_fm_net OR r.has_fm_name) AS is_fm_hwy,
                r.has_core_net,
                r.has_fm_net,
                r.has_any_usnet,
                r.has_fire_rd,
                r.has_sch_rd,
                ST_Length(ST_Intersection(r.geom5070, t.geom5070)) / 1000.0 AS km_in,
                ST_Length(ST_Intersection(r.geom5070, t.geom_buf22)) / 1000.0 AS km_b22,
                ST_Length(ST_Intersection(r.geom5070, t.geom_buf38)) / 1000.0 AS km_b38
            FROM tr t
            JOIN rds r
              ON r.xmax >= t.xmin - 0.0006 AND r.xmin <= t.xmax + 0.0006
             AND r.ymax >= t.ymin - 0.0006 AND r.ymin <= t.ymax + 0.0006
             AND ST_Intersects(r.geom5070, t.geom_buf38)
        )
        SELECT
            t.GEOID,
            t.tract_perim_km,
            coalesce(sum(CASE WHEN c.rclass IN ('motorway','trunk','primary','secondary') THEN c.km_in ELSE 0 END), 0) AS km_ovt4_exact,
            coalesce(sum(CASE WHEN c.rclass IN ('motorway','trunk','primary','secondary') AND c.is_lnk THEN c.km_in ELSE 0 END), 0) AS km_ovt4_link_in,
            coalesce(sum(CASE WHEN NOT c.is_lnk AND c.rclass = 'motorway' THEN c.km_in ELSE 0 END), 0) AS km_mot_nolink_in,
            coalesce(sum(CASE WHEN NOT c.is_lnk AND c.rclass = 'motorway' THEN c.km_b22 ELSE 0 END), 0) AS km_mot_nolink_buf22,
            coalesce(sum(CASE WHEN NOT c.is_lnk AND c.rclass = 'motorway' THEN c.km_b38 ELSE 0 END), 0) AS km_mot_nolink_buf38,
            -- Core Interstate / US / State Highway (NO FM/RM)
            coalesce(sum(CASE WHEN NOT c.is_lnk AND (c.rclass = 'motorway' OR c.has_core_net) THEN c.km_in ELSE 0 END), 0) AS km_core_net_in,
            coalesce(sum(CASE WHEN NOT c.is_lnk AND (c.rclass = 'motorway' OR c.has_core_net) THEN c.km_b22 ELSE 0 END), 0) AS km_core_net_buf22,
            coalesce(sum(CASE WHEN NOT c.is_lnk AND (c.rclass = 'motorway' OR c.has_core_net) THEN c.km_b38 ELSE 0 END), 0) AS km_core_net_buf38,
            coalesce(sum(CASE WHEN NOT c.is_lnk AND (c.rclass = 'motorway' OR c.is_core_hwy) THEN c.km_in ELSE 0 END), 0) AS km_core_nethwy_in,
            coalesce(sum(CASE WHEN NOT c.is_lnk AND (c.rclass = 'motorway' OR c.is_core_hwy) THEN c.km_b22 ELSE 0 END), 0) AS km_core_nethwy_buf22,
            coalesce(sum(CASE WHEN NOT c.is_lnk AND (c.rclass = 'motorway' OR c.is_core_hwy) THEN c.km_b38 ELSE 0 END), 0) AS km_core_nethwy_buf38,
            coalesce(sum(CASE WHEN NOT c.is_lnk AND (c.rclass = 'motorway' OR c.is_core_hwy)
                              THEN c.km_b38 * (CASE WHEN c.is_div AND c.rclass != 'motorway' THEN 0.55 ELSE 1.0 END) ELSE 0 END), 0) AS km_core_nethwy_cl_buf38,
            -- Texas FM / RM separated by major (trunk/primary/secondary) vs minor (tertiary/unclassified)
            coalesce(sum(CASE WHEN NOT c.is_lnk AND NOT (c.rclass = 'motorway' OR c.is_core_hwy) AND c.is_fm_hwy AND c.rclass IN ('trunk','primary','secondary') THEN c.km_in ELSE 0 END), 0) AS km_fm_major_in,
            coalesce(sum(CASE WHEN NOT c.is_lnk AND NOT (c.rclass = 'motorway' OR c.is_core_hwy) AND c.is_fm_hwy AND c.rclass IN ('trunk','primary','secondary') THEN c.km_b38 ELSE 0 END), 0) AS km_fm_major_buf38,
            coalesce(sum(CASE WHEN NOT c.is_lnk AND NOT (c.rclass = 'motorway' OR c.is_core_hwy) AND c.is_fm_hwy AND c.rclass NOT IN ('motorway','trunk','primary','secondary') THEN c.km_in ELSE 0 END), 0) AS km_fm_minor_in,
            coalesce(sum(CASE WHEN NOT c.is_lnk AND NOT (c.rclass = 'motorway' OR c.is_core_hwy) AND c.is_fm_hwy AND c.rclass NOT IN ('motorway','trunk','primary','secondary') THEN c.km_b38 ELSE 0 END), 0) AS km_fm_minor_buf38,
            -- Combined Core + All FM/RM
            coalesce(sum(CASE WHEN NOT c.is_lnk AND (c.rclass = 'motorway' OR c.is_core_hwy OR c.is_fm_hwy) THEN c.km_in ELSE 0 END), 0) AS km_tnethwy_in,
            coalesce(sum(CASE WHEN NOT c.is_lnk AND (c.rclass = 'motorway' OR c.is_core_hwy OR c.is_fm_hwy) THEN c.km_b22 ELSE 0 END), 0) AS km_tnethwy_buf22,
            coalesce(sum(CASE WHEN NOT c.is_lnk AND (c.rclass = 'motorway' OR c.is_core_hwy OR c.is_fm_hwy) THEN c.km_b38 ELSE 0 END), 0) AS km_tnethwy_buf38,
            coalesce(sum(CASE WHEN NOT c.is_lnk AND (c.rclass = 'motorway' OR c.is_core_hwy OR c.is_fm_hwy)
                              THEN c.km_b38 * (CASE WHEN c.is_div AND c.rclass != 'motorway' THEN 0.55 ELSE 1.0 END) ELSE 0 END), 0) AS km_tnethwy_cl_buf38,
            -- Unrouted Trunk / Primary / County-Network auxiliary signals
            coalesce(sum(CASE WHEN NOT c.is_lnk AND c.rclass = 'trunk' AND NOT (c.is_core_hwy OR c.is_fm_hwy) THEN c.km_b22 ELSE 0 END), 0) AS km_trunk_unrouted_buf22,
            coalesce(sum(CASE WHEN NOT c.is_lnk AND c.rclass = 'primary' AND NOT (c.is_core_hwy OR c.is_fm_hwy) THEN c.km_in ELSE 0 END), 0) AS km_pri_unrouted_in,
            coalesce(sum(CASE WHEN NOT c.is_lnk AND c.has_any_usnet AND NOT (c.rclass = 'motorway' OR c.is_core_hwy OR c.is_fm_hwy) THEN c.km_b22 ELSE 0 END), 0) AS km_cnet_only_buf22,
            coalesce(sum(CASE WHEN c.has_fire_rd THEN 1 ELSE 0 END), 0) AS road_fire_name_cnt,
            coalesce(sum(CASE WHEN c.has_sch_rd THEN 1 ELSE 0 END), 0) AS road_school_name_cnt
        FROM tr t
        LEFT JOIN cand c ON t.GEOID = c.GEOID
        GROUP BY t.GEOID, t.tract_perim_km;
    """).df()
    t_rd = time.time()
    con.execute("DROP TABLE IF EXISTS rds;")

    # 3. Extract POI exact counts (strict 6-category schools, town-center civic anchors, 5-digit ZIP postcode allocation)
    con.execute("DROP TABLE IF EXISTS joined_pois_tmp;")
    con.execute(f"""
        CREATE TEMP TABLE joined_pois_tmp AS
        WITH raw_p AS (
            SELECT
                (bbox.xmin + bbox.xmax) * 0.5 AS cx,
                (bbox.ymin + bbox.ymax) * 0.5 AS cy,
                CAST(floor(((bbox.xmin + bbox.xmax) * 0.5) * 20.0) AS BIGINT) AS gx,
                CAST(floor(((bbox.ymin + bbox.ymax) * 0.5) * 20.0) AS BIGINT) AS gy,
                coalesce(categories.primary, '') AS pcat,
                (addresses IS NOT NULL AND length(addresses) > 0) AS has_addr,
                regexp_extract(coalesce(addresses[1].postcode, ''), '([0-9]{{5}})', 1) AS zip5,
                coalesce(operating_status, 'open') AS op_status,
                coalesce(confidence, 0.0) AS conf,
                geometry AS geom4326
            FROM read_parquet('{pois_url}')
        )
        SELECT t.GEOID, p.pcat, p.has_addr, p.zip5, p.op_status, p.conf
        FROM raw_p p
        JOIN tr_grid tg
          ON p.gx = tg.gx AND p.gy = tg.gy
         AND p.cx BETWEEN tg.xmin_pad AND tg.xmax_pad
         AND p.cy BETWEEN tg.ymin_pad AND tg.ymax_pad
        JOIN tr t ON tg.GEOID = t.GEOID
        WHERE ST_Intersects(p.geom4326, t.geom4326);
    """)

    df_pois = con.execute(f"""
        WITH zip_stats AS (
            SELECT
                zip5,
                count(*) AS zip_total_pois,
                count(DISTINCT GEOID) AS zip_n_tracts
            FROM joined_pois_tmp
            WHERE zip5 != ''
            GROUP BY zip5
        ),
        tract_zip AS (
            SELECT
                j.GEOID,
                j.zip5,
                count(*) AS tz_pois,
                z.zip_total_pois,
                z.zip_n_tracts
            FROM joined_pois_tmp j
            JOIN zip_stats z ON j.zip5 = z.zip5
            WHERE j.zip5 != ''
            GROUP BY j.GEOID, j.zip5, z.zip_total_pois, z.zip_n_tracts
        ),
        tract_zip_agg AS (
            SELECT
                GEOID,
                count(DISTINCT zip5) AS ovt_n_zip5,
                sum(tz_pois * (CAST(zip_total_pois AS DOUBLE) / greatest(zip_n_tracts, 1))) / greatest(sum(tz_pois), 1) AS ovt_zip5_mean_tract_pois
            FROM tract_zip
            GROUP BY GEOID
        ),
        base_p AS (
            SELECT
                t.GEOID,
                count(p.pcat) AS ovt_pois_v9,
                coalesce(sum(CASE WHEN p.pcat = 'fire_department' THEN 1 ELSE 0 END), 0) AS ovt_fire_strict,
                coalesce(sum(CASE WHEN p.pcat = 'ambulance_and_ems_services' THEN 1 ELSE 0 END), 0) AS ovt_ems_strict,
                coalesce(sum(CASE WHEN p.pcat IN {SCHOOL_STRICT6_SQL} THEN 1 ELSE 0 END), 0) AS ovt_schools_strict6,
                coalesce(sum(CASE WHEN p.pcat IN ('preschool', 'kindergarten') THEN 1 ELSE 0 END), 0) AS ovt_preschool_kinder,
                coalesce(sum(CASE WHEN p.has_addr THEN 1 ELSE 0 END), 0) AS ovt_pois_with_addr,
                coalesce(sum(CASE WHEN p.op_status = 'open' AND p.has_addr AND p.conf >= 0.70 THEN 1 ELSE 0 END), 0) AS ovt_pois_open_addr_hc,
                coalesce(sum(CASE WHEN p.pcat = 'post_office' THEN 1 ELSE 0 END), 0) AS ovt_post_office_cnt,
                coalesce(sum(CASE WHEN p.pcat IN ('town_hall', 'central_government_office', 'government_services', 'public_service_and_government') THEN 1 ELSE 0 END), 0) AS ovt_gov_hall_cnt,
                coalesce(sum(CASE WHEN p.pcat IN ('police_department', 'law_enforcement') THEN 1 ELSE 0 END), 0) AS ovt_police_cnt,
                coalesce(sum(CASE WHEN p.pcat = 'library' THEN 1 ELSE 0 END), 0) AS ovt_library_cnt
            FROM tr t
            LEFT JOIN joined_pois_tmp p ON t.GEOID = p.GEOID
            GROUP BY t.GEOID
        )
        SELECT
            b.*,
            coalesce(z.ovt_n_zip5, 0) AS ovt_n_zip5,
            coalesce(z.ovt_zip5_mean_tract_pois, 0.0) AS ovt_zip5_mean_tract_pois
        FROM base_p b
        LEFT JOIN tract_zip_agg z ON b.GEOID = z.GEOID;
    """).df()
    con.execute("DROP TABLE IF EXISTS joined_pois_tmp;")

    # 4. Extract candidate Fire, EMS, and K-12 points across `overture-pois` and `overture-buildings` for 300m spatial clustering!
    df_cl = extract_region_clusters_v9(con, region, geoids)

    # 5. Extract Overture Infrastructure (airports / transit stations)
    df_infra = con.execute(f"""
        WITH raw_i AS (
            SELECT
                (bbox.xmin + bbox.xmax) * 0.5 AS cx,
                (bbox.ymin + bbox.ymax) * 0.5 AS cy,
                CAST(floor(((bbox.xmin + bbox.xmax) * 0.5) * 20.0) AS BIGINT) AS gx,
                CAST(floor(((bbox.ymin + bbox.ymax) * 0.5) * 20.0) AS BIGINT) AS gy,
                coalesce("class", '') AS iclass,
                geometry AS geom4326
            FROM read_parquet('{infra_url}')
        ),
        joined_i AS (
            SELECT t.GEOID, i.iclass
            FROM raw_i i
            JOIN tr_grid tg
              ON i.gx = tg.gx AND i.gy = tg.gy
             AND i.cx BETWEEN tg.xmin_pad AND tg.xmax_pad
             AND i.cy BETWEEN tg.ymin_pad AND tg.ymax_pad
            JOIN tr t ON tg.GEOID = t.GEOID
            WHERE ST_Intersects(i.geom4326, t.geom4326)
        )
        SELECT
            t.GEOID,
            count(j.iclass) AS ovt_infra_cnt,
            coalesce(sum(CASE WHEN j.iclass ILIKE '%airport%' OR j.iclass ILIKE '%aerodrome%' OR j.iclass ILIKE '%helipad%' THEN 1 ELSE 0 END), 0) AS ovt_airport_cnt
        FROM tr t
        LEFT JOIN joined_i j ON t.GEOID = j.GEOID
        GROUP BY t.GEOID;
    """).df()

    out = (
        df_roads
        .merge(df_pois, on="GEOID", how="left")
        .merge(df_cl, on="GEOID", how="left")
        .merge(df_infra, on="GEOID", how="left")
    )
    num_cols = [c for c in out.columns if c != "GEOID"]
    out[num_cols] = out[num_cols].fillna(0.0)
    out["region"] = region

    t_end = time.time()
    print(
        f"[{region:16s}] tracts={len(out):4d} | rd={t_rd-t0:4.1f}s | tot={t_end-t0:4.1f}s | "
        f"u_core_b38={int((out['km_core_nethwy_buf38'] <= 0.015).sum()):4d} | "
        f"u_all_b38={int((out['km_tnethwy_buf38'] <= 0.015).sum()):4d} | "
        f"unm_fire={int((out['fire_unmapped_clusters_300m'] >= 1).sum()):3d} (ovt0={int(((out['fire_unmapped_clusters_300m'] >= 1) & (out['ovt_fire_strict'] == 0)).sum()):3d}) | "
        f"unm_k12={int((out['k12_unmapped_campuses_300m'] >= 1).sum()):3d} (sch0={int(((out['k12_unmapped_campuses_300m'] >= 1) & (out['ovt_schools_strict6'] == 0)).sum()):3d})",
        flush=True,
    )
    return out


def refresh_clusters_only(con: duckdb.DuckDBPyConnection, out_path: Path) -> None:
    """Fast (~50s) surgical refresh of the 300m spatial facility cluster columns without re-downloading road WKB."""
    t_start = time.time()
    existing = pd.read_parquet(out_path)
    existing["GEOID"] = existing["GEOID"].astype(str).str.strip().str.zfill(11)

    cl_frames = []
    for region in REGIONS:
        t0 = time.time()
        tracts_url = f"{S3_BASE}/strata/{region}/{region}-census-tracts.parquet"
        con.execute("DROP TABLE IF EXISTS tr; DROP TABLE IF EXISTS tr_grid;")
        con.execute(f"""
            CREATE TEMP TABLE tr AS
            SELECT
                GEOID,
                bbox.xmin AS xmin, bbox.xmax AS xmax, bbox.ymin AS ymin, bbox.ymax AS ymax,
                geometry AS geom4326
            FROM read_parquet('{tracts_url}');

            CREATE TEMP TABLE tr_grid AS
            SELECT
                t.GEOID,
                t.xmin - 0.0006 AS xmin_pad,
                t.xmax + 0.0006 AS xmax_pad,
                t.ymin - 0.0006 AS ymin_pad,
                t.ymax + 0.0006 AS ymax_pad,
                gx.gx,
                gy.gy
            FROM tr t,
                 generate_series(CAST(floor((t.xmin - 0.0006) * 20.0) AS BIGINT), CAST(floor((t.xmax + 0.0006) * 20.0) AS BIGINT)) AS gx(gx),
                 generate_series(CAST(floor((t.ymin - 0.0006) * 20.0) AS BIGINT), CAST(floor((t.ymax + 0.0006) * 20.0) AS BIGINT)) AS gy(gy);
        """)
        geoids = con.execute("SELECT GEOID FROM tr ORDER BY GEOID").df()["GEOID"].tolist()
        df_cl = extract_region_clusters_v9(con, region, geoids)
        df_cl["GEOID"] = df_cl["GEOID"].astype(str).str.strip().str.zfill(11)
        df_cl["region"] = region
        cl_frames.append(df_cl)
        sub_ex = existing[existing["region"] == region].merge(df_cl[["GEOID", "fire_unmapped_clusters_300m", "k12_unmapped_campuses_300m", "k12_true_campuses_300m"]], on="GEOID", suffixes=("_old", ""))
        print(
            f"[REFRESH {region:16s}] {time.time()-t0:4.1f}s | "
            f"unm_fire={int((df_cl['fire_unmapped_clusters_300m'] >= 1).sum()):3d} "
            f"(ovt0={int(((sub_ex['fire_unmapped_clusters_300m'] >= 1) & (sub_ex['ovt_fire_strict'] == 0)).sum()):3d}) | "
            f"unm_k12={int((df_cl['k12_unmapped_campuses_300m'] >= 1).sum()):3d} "
            f"(sch0={int(((sub_ex['k12_unmapped_campuses_300m'] >= 1) & (sub_ex['ovt_schools_strict6'] == 0)).sum()):3d}) | "
            f"true_k12_tracts={int((df_cl['k12_true_campuses_300m'] >= 1).sum()):4d}/{len(df_cl):4d}",
            flush=True,
        )

    full_cl = pd.concat(cl_frames, ignore_index=True)
    cl_cols = [c for c in full_cl.columns if c not in ("GEOID", "region")]
    keep_cols = [c for c in existing.columns if c not in cl_cols]
    updated = existing[keep_cols].merge(full_cl, on=["GEOID", "region"], how="left")
    miss = updated[cl_cols[0]].isna()
    if miss.any():
        dedup = full_cl.drop_duplicates("GEOID").drop(columns=["region"])
        fill_df = updated.loc[miss, ["GEOID"]].merge(dedup, on="GEOID", how="left")
        for c in cl_cols:
            updated.loc[miss, c] = fill_df[c].values
    updated[cl_cols] = updated[cl_cols].fillna(0.0)

    assert len(updated) == 9794, f"Expected 9,794 rows, got {len(updated)}"
    assert not updated.isna().any().any(), "Unexpected NaNs after cluster refresh!"
    updated.to_parquet(out_path, index=False)
    print(f"\n[DONE REFRESH] Updated {out_path} ({updated.shape[0]}x{updated.shape[1]}) in {time.time()-t_start:.1f}s", flush=True)


def main() -> None:
    t_start = time.time()
    out_path = ROOT / "challenge_package_v9_features_9794.parquet"
    con = duckdb.connect()
    con.execute("SET memory_limit = '3GB'; SET threads = 2; SET preserve_insertion_order = false;")
    con.execute("INSTALL httpfs; LOAD httpfs; INSTALL spatial; LOAD spatial;")

    if "--refresh-clusters-only" in sys.argv and out_path.exists():
        refresh_clusters_only(con, out_path)
        return

    frames = []
    for reg in REGIONS:
        frames.append(extract_region_v9(con, reg))

    full_df = pd.concat(frames, ignore_index=True)

    ss = pd.read_csv(ROOT / "SampleSubmission.csv", dtype={"GEOID": str})
    ss["GEOID"] = ss["GEOID"].astype(str).str.strip().str.zfill(11)

    aligned = ss[["GEOID", "region"]].merge(full_df, on=["GEOID", "region"], how="left")
    miss = aligned["km_ovt4_exact"].isna()
    if miss.any():
        dedup = full_df.drop_duplicates("GEOID").drop(columns=["region"])
        fill_df = aligned.loc[miss, ["GEOID", "region"]].merge(dedup, on="GEOID", how="left")
        aligned.loc[miss, fill_df.columns] = fill_df.values

    assert len(aligned) == 9794, f"Expected 9,794 rows, got {len(aligned)}"
    assert not aligned.isna().any().any(), "Unexpected NaNs in v9 feature table!"

    aligned.to_parquet(out_path, index=False)
    print(f"\n[DONE] Saved {out_path} ({aligned.shape[0]}x{aligned.shape[1]}) in {time.time()-t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
