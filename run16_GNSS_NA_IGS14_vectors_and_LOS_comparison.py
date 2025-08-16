#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 14:51:36 2025

@author: daniellelindsay
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pygmt
import h5py

from NC_ALOS2_filepaths import common_paths, paths_gps, paths_169
import insar_utils as utils

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
track       = "169"
ref_station = common_paths["ref_station"]
unit        = 1000

# Map region (edit if needed)
min_lon, max_lon = -124.63, -120.0
min_lat, max_lat =   36.17,   42.41
region = f"{min_lon}/{max_lon}/{min_lat}/{max_lat}"

# ---------------------------------------------------------------------
# Helpers (NO in-place mutation)
# ---------------------------------------------------------------------
def ref_enu_to_station(df_enu: pd.DataFrame, station: str) -> pd.DataFrame:
    """Subtract ENU components at `station` from ENU; returns a copy."""
    out = df_enu.copy(deep=True)
    mask = out["StaID"].str.fullmatch(station, case=False)
    if not mask.any():
        raise ValueError(f"Reference station '{station}' not found.")
    ve0 = float(out.loc[mask, "Ve"].iloc[0])
    vn0 = float(out.loc[mask, "Vn"].iloc[0])
    vu0 = float(out.loc[mask, "Vu"].iloc[0])
    out["Ve"] = out["Ve"] - ve0
    out["Vn"] = out["Vn"] - vn0
    out["Vu"] = out["Vu"] - vu0
    return out

def ref_los_to_station(df_los: pd.DataFrame, station: str) -> pd.DataFrame:
    """Subtract LOS at `station` from LOS; returns a copy."""
    out = df_los.copy(deep=True)
    mask = out["StaID"].str.fullmatch(station, case=False)
    if not mask.any():
        raise ValueError(f"Reference station '{station}' not found.")
    v0 = float(out.loc[mask, "LOS_Vel"].iloc[0])
    out["LOS_Vel"] = out["LOS_Vel"] - v0
    return out

def project_enu_to_los(df_enu: pd.DataFrame, insar_geo) -> pd.DataFrame:
    """Project ENU→LOS via utils; returns a copy to avoid aliasing."""
    return utils.project_gps2los_no_reference(df_enu.copy(deep=True), insar_geo)

def robust_limits(a, p=(5, 95)):
    """Percentile clip for color scale."""
    return tuple(np.nanpercentile(np.asarray(a, float), p))

def los_merge(a: pd.DataFrame, b: pd.DataFrame,
              col_a="LOS_Vel", col_b="LOS_Vel",
              colname_a="A", colname_b="B") -> pd.DataFrame:
    """
    Align by StaID and keep Lon/Lat from `a`. Value columns can be LOS_Vel or LOS_diff.
    """
    left  = a[["StaID", "Lon", "Lat", col_a]].rename(columns={col_a: colname_a})
    right = b[["StaID",           col_b]].rename(columns={col_b: colname_b})
    return left.merge(right, on="StaID", how="inner")

# ---------------------------------------------------------------------
# Load inputs
# ---------------------------------------------------------------------
insar_169 = utils.load_h5_data(
    paths_169["geo"]["geo_geometryRadar"],
    paths_169["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"],
    "velocity",
)

# RAW ENU (treat as immutable sources)
igs14_enu_raw = utils.load_UNR_gps(paths_gps[f"{track}_enu_IGS14"])
na_enu_raw    = utils.load_UNR_gps(paths_gps[f"{track}_enu_NA"])

lon_min, lon_max = insar_169["Lon"].min(), insar_169["Lon"].max()
lat_min, lat_max = insar_169["Lat"].min(), insar_169["Lat"].max()

igs14_enu_raw = igs14_enu_raw.loc[
    igs14_enu_raw["Lon"].between(lon_min, lon_max) &
    igs14_enu_raw["Lat"].between(lat_min, lat_max)
].copy()

na_enu_raw = na_enu_raw.loc[
    na_enu_raw["Lon"].between(lon_min, lon_max) &
    na_enu_raw["Lat"].between(lat_min, lat_max)
].copy()

# ---------------------------------------------------------------------
# Build LOS products with explicit order
# ---------------------------------------------------------------------
# 1) Proj LOS (no reference)
igs14_los_from_raw = project_enu_to_los(igs14_enu_raw, insar_169)
na_los_from_raw    = project_enu_to_los(na_enu_raw,    insar_169)

# 2) Proj→Ref (LOS then CASR)
igs14_los_from_raw_ref_casr = ref_los_to_station(igs14_los_from_raw, ref_station)
na_los_from_raw_ref_casr    = ref_los_to_station(na_los_from_raw,    ref_station)

# 3) Ref→Proj (CASR then LOS)
igs14_enu_ref_casr = ref_enu_to_station(igs14_enu_raw, ref_station)
na_enu_ref_casr    = ref_enu_to_station(na_enu_raw,    ref_station)

igs14_los_from_enu_ref_casr = project_enu_to_los(igs14_enu_ref_casr, insar_169)
na_los_from_enu_ref_casr    = project_enu_to_los(na_enu_ref_casr,    insar_169)

# ---------------------------------------------------------------------
# Differences & method deltas (aligned by StaID)
# ---------------------------------------------------------------------
# NA − IGS14 for each method
los_diff_from_raw = los_merge(na_los_from_raw, igs14_los_from_raw, colname_a="na", colname_b="igs")
los_diff_from_raw["LOS_diff"] = los_diff_from_raw["na"] - los_diff_from_raw["igs"]

los_diff_from_raw_ref_casr = los_merge(na_los_from_raw_ref_casr, igs14_los_from_raw_ref_casr, colname_a="na", colname_b="igs")
los_diff_from_raw_ref_casr["LOS_diff"] = los_diff_from_raw_ref_casr["na"] - los_diff_from_raw_ref_casr["igs"]

los_diff_from_enu_ref_casr = los_merge(na_los_from_enu_ref_casr, igs14_los_from_enu_ref_casr, colname_a="na", colname_b="igs")
los_diff_from_enu_ref_casr["LOS_diff"] = los_diff_from_enu_ref_casr["na"] - los_diff_from_enu_ref_casr["igs"]

# Δ(methods) = (Ref→Proj) − (Proj→Ref) for each row
igs14_delta_methods = los_merge(
    igs14_los_from_raw_ref_casr, igs14_los_from_enu_ref_casr,
    col_a="LOS_Vel", col_b="LOS_Vel", colname_a="proj_ref", colname_b="ref_proj"
)
igs14_delta_methods["Delta_methods"] = igs14_delta_methods["ref_proj"] - igs14_delta_methods["proj_ref"]

na_delta_methods = los_merge(
    na_los_from_raw_ref_casr, na_los_from_enu_ref_casr,
    col_a="LOS_Vel", col_b="LOS_Vel", colname_a="proj_ref", colname_b="ref_proj"
)
na_delta_methods["Delta_methods"] = na_delta_methods["ref_proj"] - na_delta_methods["proj_ref"]

# Δ(methods) on the DIFF row (use LOS_diff columns)
diff_delta_methods = los_merge(
    los_diff_from_raw_ref_casr, los_diff_from_enu_ref_casr,
    col_a="LOS_diff", col_b="LOS_diff", colname_a="proj_ref", colname_b="ref_proj"
)
diff_delta_methods["Delta_methods"] = diff_delta_methods["ref_proj"] - diff_delta_methods["proj_ref"]

# ---------------------------------------------------------------------
# 3×4 LOS color figure:
# Rows = [IGS14, NA, Diff]; Cols = [RAW, Proj→Ref, Ref→Proj, Δ(methods)]
# ---------------------------------------------------------------------
fig = pygmt.Figure()
pygmt.config(FONT=11, FONT_TITLE=11, MAP_HEADING_OFFSET="-7p",
             PS_MEDIA="A3", FORMAT_GEO_MAP="ddd", MAP_FRAME_TYPE="plain")

with fig.subplot(nrows=3, ncols=4, figsize=("25c", "25c"),
                 autolabel=True, sharex="b", sharey="l", frame="WSrt"):

    # ---------------- ROW 1: IGS14 ----------------
    # RAW
    fig.coast(region=region, shorelines=True, lakes=False, borders="2/thin", panel=True)
    vmin, vmax = robust_limits(igs14_los_from_raw["LOS_Vel"], (5,95))
    pygmt.makecpt(cmap="batlow", series=[vmin, vmax])
    with pygmt.config(FONT_ANNOT_PRIMARY="18p,black", FONT_ANNOT_SECONDARY="18p,black", FONT_LABEL="18p,black",):
        fig.colorbar(position="jBL+o0.3c/0.7c+w3.0c/0.4c", frame=["xa+lLOS (mm/yr)"])
    fig.plot(x=igs14_los_from_raw["Lon"], y=igs14_los_from_raw["Lat"],
             style="c.15c", fill=igs14_los_from_raw["LOS_Vel"], cmap=True)
    fig.text(text="IGS14 • ENU → LOS", position="BL", offset="0.2c/0.2c")

    # Proj→Ref
    fig.coast(region=region, shorelines=True, lakes=False, borders="2/thin", panel=True)
    vmin, vmax = robust_limits(igs14_los_from_raw_ref_casr["LOS_Vel"], (5,95))
    pygmt.makecpt(cmap="batlow", series=[vmin, vmax])
    with pygmt.config(FONT_ANNOT_PRIMARY="18p,black", FONT_ANNOT_SECONDARY="18p,black", FONT_LABEL="18p,black",):
        fig.colorbar(position="jBL+o0.3c/0.7c+w3.0c/0.4c", frame=["xa+lLOS (mm/yr)"])
    fig.plot(x=igs14_los_from_raw_ref_casr["Lon"], y=igs14_los_from_raw_ref_casr["Lat"],
             style="c.15c", fill=igs14_los_from_raw_ref_casr["LOS_Vel"], cmap=True)
    fig.text(text="IGS14 • Proj→Ref", position="BL", offset="0.2c/0.2c")

    # Ref→Proj
    fig.coast(region=region, shorelines=True, lakes=False, borders="2/thin", panel=True)
    vmin, vmax = robust_limits(igs14_los_from_enu_ref_casr["LOS_Vel"], (5,95))
    pygmt.makecpt(cmap="batlow", series=[vmin, vmax])
    with pygmt.config(FONT_ANNOT_PRIMARY="18p,black", FONT_ANNOT_SECONDARY="18p,black", FONT_LABEL="18p,black",):
        fig.colorbar(position="jBL+o0.3c/0.7c+w3.0c/0.4c", frame=["xa+lLOS (mm/yr)"])
    fig.plot(x=igs14_los_from_enu_ref_casr["Lon"], y=igs14_los_from_enu_ref_casr["Lat"],
             style="c.15c", fill=igs14_los_from_enu_ref_casr["LOS_Vel"], cmap=True)
    fig.text(text="IGS14 • Ref→Proj", position="BL", offset="0.2c/0.2c")

    # Δ(methods) = (Ref→Proj) − (Proj→Ref)
    fig.coast(region=region, shorelines=True, lakes=False, borders="2/thin", panel=True)
    amax = np.nanpercentile(np.abs(igs14_delta_methods["Delta_methods"]), 95)
    pygmt.makecpt(cmap="batlow", series=[-amax, amax])
    with pygmt.config(FONT_ANNOT_PRIMARY="18p,black", FONT_ANNOT_SECONDARY="18p,black", FONT_LABEL="18p,black",):
        fig.colorbar(position="jBL+o0.3c/0.7c+w3.0c/0.4c", frame=["xa+lΔ LOS (mm/yr)"])
    fig.plot(x=igs14_delta_methods["Lon"], y=igs14_delta_methods["Lat"],
             style="c.15c", fill=igs14_delta_methods["Delta_methods"], cmap=True)
    fig.text(text="IGS14 • Δ(methods)", position="BL", offset="0.2c/0.2c")

    # ---------------- ROW 2: NA ----------------
    # RAW
    fig.coast(region=region, shorelines=True, lakes=False, borders="2/thin", panel=True)
    vmin, vmax = robust_limits(na_los_from_raw["LOS_Vel"], (5,95))
    pygmt.makecpt(cmap="batlow", series=[vmin, vmax])
    with pygmt.config(FONT_ANNOT_PRIMARY="18p,black", FONT_ANNOT_SECONDARY="18p,black", FONT_LABEL="18p,black",):
        fig.colorbar(position="jBL+o0.3c/0.7c+w3.0c/0.4c", frame=["xa+lLOS (mm/yr)"])
    fig.plot(x=na_los_from_raw["Lon"], y=na_los_from_raw["Lat"],
             style="c.15c", fill=na_los_from_raw["LOS_Vel"], cmap=True)
    fig.text(text="NA • ENU → LOS", position="BL", offset="0.2c/0.2c")

    # Proj→Ref
    fig.coast(region=region, shorelines=True, lakes=False, borders="2/thin", panel=True)
    vmin, vmax = robust_limits(na_los_from_raw_ref_casr["LOS_Vel"], (5,95))
    pygmt.makecpt(cmap="batlow", series=[vmin, vmax])
    with pygmt.config(FONT_ANNOT_PRIMARY="18p,black", FONT_ANNOT_SECONDARY="18p,black", FONT_LABEL="18p,black",):
        fig.colorbar(position="jBL+o0.3c/0.7c+w3.0c/0.4c", frame=["xa+lLOS (mm/yr)"])
    fig.plot(x=na_los_from_raw_ref_casr["Lon"], y=na_los_from_raw_ref_casr["Lat"],
             style="c.15c", fill=na_los_from_raw_ref_casr["LOS_Vel"], cmap=True)
    fig.text(text="NA • Proj→Ref", position="BL", offset="0.2c/0.2c")

    # Ref→Proj
    fig.coast(region=region, shorelines=True, lakes=False, borders="2/thin", panel=True)
    vmin, vmax = robust_limits(na_los_from_enu_ref_casr["LOS_Vel"], (5,95))
    pygmt.makecpt(cmap="batlow", series=[vmin, vmax])
    with pygmt.config(FONT_ANNOT_PRIMARY="18p,black", FONT_ANNOT_SECONDARY="18p,black", FONT_LABEL="18p,black",):
        fig.colorbar(position="jBL+o0.3c/0.7c+w3.0c/0.4c", frame=["xa+lLOS (mm/yr)"])
    fig.plot(x=na_los_from_enu_ref_casr["Lon"], y=na_los_from_enu_ref_casr["Lat"],
             style="c.15c", fill=na_los_from_enu_ref_casr["LOS_Vel"], cmap=True)
    fig.text(text="NA • Ref→Proj", position="BL", offset="0.2c/0.2c")

    # Δ(methods)
    fig.coast(region=region, shorelines=True, lakes=False, borders="2/thin", panel=True)
    amax = np.nanpercentile(np.abs(na_delta_methods["Delta_methods"]), 95)
    pygmt.makecpt(cmap="batlow", series=[-amax, amax])
    with pygmt.config(FONT_ANNOT_PRIMARY="18p,black", FONT_ANNOT_SECONDARY="18p,black", FONT_LABEL="18p,black",):
        fig.colorbar(position="jBL+o0.3c/0.7c+w3.0c/0.4c", frame=["xa+lΔ LOS (mm/yr)"])
    fig.plot(x=na_delta_methods["Lon"], y=na_delta_methods["Lat"],
             style="c.15c", fill=na_delta_methods["Delta_methods"], cmap=True)
    fig.text(text="NA • Δ(methods)", position="BL", offset="0.2c/0.2c")

    # ---------------- ROW 3: DIFF (NA − IGS14) ----------------
    # RAW diff
    fig.coast(region=region, shorelines=True, lakes=False, borders="2/thin", panel=True)
    vmin, vmax = robust_limits(los_diff_from_raw["LOS_diff"], (5,95))
    pygmt.makecpt(cmap="batlow", series=[vmin, vmax])
    with pygmt.config(FONT_ANNOT_PRIMARY="18p,black", FONT_ANNOT_SECONDARY="18p,black", FONT_LABEL="18p,black",):
        fig.colorbar(position="jBL+o0.3c/0.7c+w3.0c/0.4c", frame=["xa+lLOS (mm/yr)"])
    fig.plot(x=los_diff_from_raw["Lon"], y=los_diff_from_raw["Lat"],
             style="c.15c", fill=los_diff_from_raw["LOS_diff"], cmap=True)
    fig.text(text="Diff ENU → LOS • (NA − IGS14)", position="BL", offset="0.2c/0.2c")

    # Proj→Ref diff
    fig.coast(region=region, shorelines=True, lakes=False, borders="2/thin", panel=True)
    vmin, vmax = robust_limits(los_diff_from_raw_ref_casr["LOS_diff"], (5,95))
    pygmt.makecpt(cmap="batlow", series=[vmin, vmax])
    with pygmt.config(FONT_ANNOT_PRIMARY="18p,black", FONT_ANNOT_SECONDARY="18p,black", FONT_LABEL="18p,black",):
        fig.colorbar(position="jBL+o0.3c/0.7c+w3.0c/0.4c", frame=["xa+lLOS (mm/yr)"])
    fig.plot(x=los_diff_from_raw_ref_casr["Lon"], y=los_diff_from_raw_ref_casr["Lat"],
             style="c.15c", fill=los_diff_from_raw_ref_casr["LOS_diff"], cmap=True)
    fig.text(text="Diff Proj→Ref • (NA − IGS14)", position="BL", offset="0.2c/0.2c")

    # Ref→Proj diff
    fig.coast(region=region, shorelines=True, lakes=False, borders="2/thin", panel=True)
    vmin, vmax = robust_limits(los_diff_from_enu_ref_casr["LOS_diff"], (5,95))
    pygmt.makecpt(cmap="batlow", series=[vmin, vmax])
    with pygmt.config(FONT_ANNOT_PRIMARY="18p,black", FONT_ANNOT_SECONDARY="18p,black", FONT_LABEL="18p,black",):
        fig.colorbar(position="jBL+o0.3c/0.7c+w3.0c/0.4c", frame=["xa+lLOS (mm/yr)"])
    fig.plot(x=los_diff_from_enu_ref_casr["Lon"], y=los_diff_from_enu_ref_casr["Lat"],
             style="c.15c", fill=los_diff_from_enu_ref_casr["LOS_diff"], cmap=True)
    fig.text(text="Diff Ref→Proj • (NA − IGS14)", position="BL", offset="0.2c/0.2c")



# Save
fig_path = common_paths["fig_dir"] + f"/Fig_{track}_LOS_maps_IGS14_NA_3x4.png"
fig.savefig(fig_path, transparent=False, crop=True, anti_alias=True, show=False)
fig.show()
print("Saved:", fig_path)



# ---------- helpers (only if not already defined above) ----------
def velo_df_from_enu(df):
    return pd.DataFrame({
        "x": df["Lon"], "y": df["Lat"],
        "east_velocity":  df["Ve"], "north_velocity": df["Vn"],
        "east_sigma":     df["Std_e"], "north_sigma": df["Std_n"],
    })

def enu_difference_velo_df(na_df, igs_df):
    m = (na_df[["StaID","Lon","Lat","Ve","Vn","Std_e","Std_n"]]
         .merge(igs_df[["StaID","Ve","Vn"]], on="StaID", suffixes=("_na","_igs")))
    return pd.DataFrame({
        "x": m["Lon"], "y": m["Lat"],
        "east_velocity":  m["Ve_na"] - m["Ve_igs"],
        "north_velocity": m["Vn_na"] - m["Vn_igs"],
        "east_sigma":     m["Std_e"],      # keep NA sigmas (or combine if you prefer)
        "north_sigma":    m["Std_n"],
    })
# ---------------------------------------------------------------

# Build vector tables
igs14_vec_raw      = velo_df_from_enu(igs14_enu_raw)
na_vec_raw         = velo_df_from_enu(na_enu_raw)
enu_diff_raw       = enu_difference_velo_df(na_enu_raw,       igs14_enu_raw)

igs14_vec_ref_casr = velo_df_from_enu(igs14_enu_ref_casr)
na_vec_ref_casr    = velo_df_from_enu(na_enu_ref_casr)
enu_diff_ref_casr  = enu_difference_velo_df(na_enu_ref_casr,  igs14_enu_ref_casr)

# Scale arrows
scale_30 = pd.DataFrame({"x":[-123.5], "y":[36.8],
                         "east_velocity":[30], "north_velocity":[0],
                         "east_sigma":[0], "north_sigma":[0]})
scale_01 = pd.DataFrame({"x":[-123.5], "y":[36.8],
                         "east_velocity":[1], "north_velocity":[0],
                         "east_sigma":[0], "north_sigma":[0]})

# ---------- ENU vector figure (2 x 3) ----------
fig = pygmt.Figure()
pygmt.config(FONT=11, FONT_TITLE=11, MAP_HEADING_OFFSET=0.1,
             PS_MEDIA="A3", FORMAT_GEO_MAP="ddd", MAP_FRAME_TYPE="plain")

with fig.subplot(nrows=2, ncols=3, figsize=("25c", "22c"),
                 autolabel=True, sharex="b", sharey="l", frame="WSrt"):

    # (a) IGS14 ENU (RAW)
    fig.coast(region=region, shorelines=True, lakes=False, borders="2/thin", panel=True)
    fig.velo(data=igs14_vec_raw, pen="0.6p,black", line=True,
             spec="e0.02/0.39/18", vector="0.2c+p1p+e+gblack")
    fig.velo(data=scale_30, pen="0.6p,black", line=True,
             spec="e0.02/0.39/18", vector="0.2c+p1p+e+gblack")
    fig.text(text="IGS14 • EN", position="BL", offset="0.2c/0.2c")
    fig.text(text="30 mm/yr", x=scale_30["x"], y=scale_30["y"], offset="-0.1c/0.0c", justify="MR")

    # (b) NA ENU (RAW)
    fig.coast(region=region, shorelines=True, lakes=False, borders="2/thin", panel=True)
    fig.velo(data=na_vec_raw, pen="0.6p,black", line=True,
             spec="e0.02/0.39/18", vector="0.2c+p1p+e+gblack")
    fig.velo(data=scale_30, pen="0.6p,black", line=True,
             spec="e0.02/0.39/18", vector="0.2c+p1p+e+gblack")
    fig.text(text="NA • EN", position="BL", offset="0.2c/0.2c")
    fig.text(text="30 mm/yr", x=scale_30["x"], y=scale_30["y"], offset="-0.1c/0.0c", justify="MR")

    # (c) Diff ENU (RAW): NA − IGS14
    fig.coast(region=region, shorelines=True, lakes=False, borders="2/thin", panel=True)    
    fig.velo(data=enu_diff_raw, pen="0.6p,black", line=True, spec="e0.02/0.39/18", vector="0.2c+p1p+e+gblack",)
    fig.velo(data=scale_30, pen="0.6p,black", line=True, spec="e0.02/0.39/18", vector="0.2c+p1p+e+gblack",)
    fig.text(text="Diff EN • NA − IGS14", position="BL", offset="0.2c/0.2c")
    fig.text(text="20 mm/yr", x=scale_01["x"], y=scale_01["y"], offset="-0.1c/0.0c", justify="MR")

    # (d) IGS14 ENU (CASR-referenced)
    fig.coast(region=region, shorelines=True, lakes=False, borders="2/thin", panel=True)
    fig.velo(data=igs14_vec_ref_casr, pen="0.6p,black", line=True,
             spec="e0.02/0.39/18", vector="0.2c+p1p+e+gblack")
    fig.velo(data=scale_30, pen="0.6p,black", line=True,
             spec="e0.02/0.39/18", vector="0.2c+p1p+e+gblack")
    fig.text(text="IGS14 • EN (CASR)", position="BL", offset="0.2c/0.2c")
    fig.text(text="30 mm/yr", x=scale_30["x"], y=scale_30["y"], offset="-0.1c/0.0c", justify="MR")

    # (e) NA ENU (CASR-referenced)
    fig.coast(region=region, shorelines=True, lakes=False, borders="2/thin", panel=True)
    fig.velo(data=na_vec_ref_casr, pen="0.6p,black", line=True,
             spec="e0.02/0.39/18", vector="0.2c+p1p+e+gblack")
    fig.velo(data=scale_30, pen="0.6p,black", line=True,
             spec="e0.02/0.39/18", vector="0.2c+p1p+e+gblack")
    fig.text(text="NA • EN (CASR)", position="BL", offset="0.2c/0.2c")
    fig.text(text="30 mm/yr", x=scale_30["x"], y=scale_30["y"], offset="-0.1c/0.0c", justify="MR")

    # (f) Diff ENU (CASR): NA − IGS14
    fig.coast(region=region, shorelines=True, lakes=False, borders="2/thin", panel=True)
    fig.velo(data=enu_diff_ref_casr, pen="0.6p,black", line=True,
             spec="e0.5/0.39/18", vector="0.2c+p1p+e+gblack")
    fig.velo(data=scale_01, pen="0.6p,black", line=True,
             spec="e0.5/0.39/18", vector="0.2c+p1p+e+gblack")
    fig.text(text="Diff EN (CASR) • NA − IGS14", position="BL", offset="0.2c/0.2c")
    fig.text(text="1 mm/yr", x=scale_01["x"], y=scale_01["y"], offset="-0.1c/0.0c", justify="MR")

# Save
vec_fig_path = common_paths["fig_dir"] + f"Fig_{track}_ENU_vectors_IGS14_NA_2x3.png"
fig.savefig(vec_fig_path, transparent=False, crop=True, anti_alias=True, show=False)
fig.show()
print("Saved:", vec_fig_path)
