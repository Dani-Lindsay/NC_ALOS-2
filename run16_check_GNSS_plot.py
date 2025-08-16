#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 09:29:54 2025

@author: daniellelindsay
"""

import pygmt
import pandas as pd
from NC_ALOS2_filepaths import (common_paths, paths_170, paths_170, paths_170)
import insar_utils as utils

from NC_ALOS2_filepaths import (paths_gps, paths_170, paths_170, paths_170, common_paths)
import insar_utils as utils
import numpy as np                # Matrix calculations
import pandas as pd               # Pandas for data
import pygmt
import h5py

dist = common_paths["dist"]
ref_station = common_paths["ref_station"]

unit = 1000
track = "170"

def _ref_to_station(df, ref_station):
    out = df.copy(deep=True)
    mask = out["StaID"].str.fullmatch(ref_station, case=False)
    if not mask.any():
        raise ValueError(f"Reference station '{ref_station}' not found in DataFrame.")
    ve0 = float(out.loc[mask, "Ve"].iloc[0])
    vn0 = float(out.loc[mask, "Vn"].iloc[0])
    vu0 = float(out.loc[mask, "Vu"].iloc[0])
    out["Ve"] = out["Ve"] - ve0
    out["Vn"] = out["Vn"] - vn0
    out["Vu"] = out["Vu"] - vu0
    return out

def _ref_to_station_LOS(df, ref_station):
    out = df.copy(deep=True)
    mask = out["StaID"].str.fullmatch(ref_station, case=False)
    if not mask.any():
        raise ValueError(f"Reference station '{ref_station}' not found in DataFrame.")
    v0 = float(out.loc[mask, "LOS_Vel"].iloc[0])
    out["LOS_Vel"] = out["LOS_Vel"] - v0
    return out

#####################
# Load in GPS and InSAR
#####################

# Load InSAR geometry and velocities
insar_170 = utils.load_h5_data(
    paths_170["geo"]["geo_geometryRadar"],
    paths_170["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"],
    "velocity",
)

# Load GNSS (raw ENU, keep these as the immutable sources)
gps_170_igs14 = utils.load_UNR_gps(paths_gps["170_enu_IGS14"])
gps_170_na    = utils.load_UNR_gps(paths_gps["170_enu_NA"])

#####################
# Project to LOS, NO adjustment for reference station
# (copy to avoid in-place mutation inside utils)
#####################
#1) Project to LOS
gps_170_igs14_LOS = utils.project_gps2los_no_reference(gps_170_igs14.copy(deep=True), insar_170)
gps_170_na_LOS    = utils.project_gps2los_no_reference(gps_170_na.copy(deep=True),    insar_170)

# 2) Reference to CASE
gps_170_igs14_LOS_CASR = _ref_to_station_LOS(gps_170_igs14_LOS.copy(deep=True), ref_station)
gps_170_na_LOS_CASR    = _ref_to_station_LOS(gps_170_na_LOS.copy(deep=True),    ref_station)
#####################
# Subtract reference station (on copies to keep raw ENU intact)
#####################
#1) Reference to CASR
gps_170_igs14_ref = _ref_to_station(gps_170_igs14, ref_station)
gps_170_na_ref    = _ref_to_station(gps_170_na,    ref_station)

#2) Project to LOS *after* referencing to CASR
gps_170_igs14_CASR_LOS = utils.project_gps2los_no_reference(gps_170_igs14_ref.copy(deep=True), insar_170)
gps_170_na_CASR_LOS    = utils.project_gps2los_no_reference(gps_170_na_ref.copy(deep=True),    insar_170)

# #####################
# # Correct Plate Motion (defensively copy in case util mutates)
# #####################
# gps_170_igs14_LOS_pm       = utils.gps_LOS_correction_plate_motion(
#     paths_170["geo"]["geo_geometryRadar"], paths_170["geo"]["ITRF_LOS"],
#     gps_170_igs14_LOS.copy(deep=True), ref_station, unit
# )
# gps_170_na_LOS_pm          = utils.gps_LOS_correction_plate_motion(
#     paths_170["geo"]["geo_geometryRadar"], paths_170["geo"]["ITRF_LOS"],
#     gps_170_na_LOS.copy(deep=True), ref_station, unit
# )
# gps_170_igs14_CASR_LOS_pm  = utils.gps_LOS_correction_plate_motion(
#     paths_170["geo"]["geo_geometryRadar"], paths_170["geo"]["ITRF_LOS"],
#     gps_170_igs14_CASR_LOS.copy(deep=True), ref_station, unit
# )
# gps_170_na_CASR_LOS_pm     = utils.gps_LOS_correction_plate_motion(
#     paths_170["geo"]["geo_geometryRadar"], paths_170["geo"]["ITRF_LOS"],
#     gps_170_na_CASR_LOS.copy(deep=True), ref_station, unit
# )
#####################
# Plot Results 
#####################

# Set lat and lon for plotting from the gps file. 
ref_lat = gps_170_na.loc[gps_170_na["StaID"] == ref_station, "Lat"].values
ref_lon = gps_170_na.loc[gps_170_na["StaID"] == ref_station, "Lon"].values

# Define region of interest 
min_lon=-124.63
max_lon=-120.0
min_lat=36.17
max_lat=42.41

region="%s/%s/%s/%s" % (min_lon, max_lon, min_lat, max_lat)
fig_size = "M8c"

scale_df = pd.DataFrame(
    data={
        "x": [-123.5],
        "y": [36.8],
        "east_velocity": [30],
        "north_velocity": [0],
        "east_sigma": [0],
        "north_sigma": [0],
    }
)

### Begin plotting ###
fig = pygmt.Figure()
pygmt.config(FONT=11, FONT_TITLE=11, MAP_HEADING_OFFSET=0.1, PS_MEDIA="A3", FORMAT_GEO_MAP="ddd", MAP_FRAME_TYPE="plain")

with fig.subplot(nrows=2, ncols=3, figsize=("25c", "22c"), autolabel=True, sharex="b", sharey="l", frame="WSrt",):
    
    # GPS igs14
    fig.coast(shorelines=True,lakes=False, borders="2/thin", region=region, panel=True)
    data = pd.DataFrame({
    "x": gps_170_igs14_LOS["Lon"],
    "y": gps_170_igs14_LOS["Lat"],
    "east_velocity": gps_170_igs14_LOS["Ve"],
    "north_velocity": gps_170_igs14_LOS["Vn"],
    "east_sigma": gps_170_igs14_LOS["Std_e"],
    "north_sigma": gps_170_igs14_LOS["Std_n"],
    })
    fig.velo(data=data, pen="0.6p,black", line=True, spec="e0.02/0.39/18", vector="0.2c+p1p+e+gblack",)
    fig.velo(data=scale_df, pen="0.6p,black", line=True, spec="e0.02/0.39/18", vector="0.2c+p1p+e+gblack",)
    fig.text(text='IGS14 Ref Frame', position="BL", offset="0.2c/0.2c")  
    fig.text(text='30 mm/yr', x=scale_df["x"], y=scale_df["y"], offset="-0.1c/0.0c", justify="MR")  
    
    # GPS NA
    fig.coast(shorelines=True,lakes=False, borders="2/thin", region=region, panel=True)
    data = pd.DataFrame({
    "x": gps_170_na_LOS["Lon"],
    "y": gps_170_na_LOS["Lat"],
    "east_velocity": gps_170_na_LOS["Ve"],
    "north_velocity": gps_170_na_LOS["Vn"],
    "east_sigma": gps_170_na_LOS["Std_e"],
    "north_sigma": gps_170_na_LOS["Std_n"],
    })
    fig.velo(data=data, pen="0.6p,black", line=True, spec="e0.02/0.39/18", vector="0.2c+p1p+e+gblack",)
    fig.velo(data=scale_df, pen="0.6p,black", line=True, spec="e0.02/0.39/18", vector="0.2c+p1p+e+gblack",)
    fig.text(text='NA Ref Frame', position="BL", offset="0.2c/0.2c")
    fig.text(text='30 mm/yr', x=scale_df["x"], y=scale_df["y"], offset="-0.1c/0.0c", justify="MR")  
    
    # GPS Different 
    fig.coast(shorelines=True,lakes=False, borders="2/thin", region=region, panel=True)
    data = pd.DataFrame({
    "x": gps_170_na_LOS["Lon"],
    "y": gps_170_na_LOS["Lat"],
    "east_velocity": gps_170_na_LOS["Ve"] - gps_170_igs14_LOS["Ve"],
    "north_velocity": gps_170_na_LOS["Vn"] - gps_170_igs14_LOS["Vn"],
    "east_sigma": gps_170_na_LOS["Std_e"],
    "north_sigma": gps_170_na_LOS["Std_n"],
    })
    fig.velo(data=data, pen="0.6p,black", line=True, spec="e0.02/0.39/18", vector="0.2c+p1p+e+gblack",)
    fig.velo(data=scale_df, pen="0.6p,black", line=True, spec="e0.02/0.39/18", vector="0.2c+p1p+e+gblack",)
    fig.text(text='Difference NA-IGS14', position="BL", offset="0.2c/0.2c")
    fig.text(text='30 mm/yr', x=scale_df["x"], y=scale_df["y"], offset="-0.1c/0.0c", justify="MR")  

    
    # GPS igs14
    fig.coast(shorelines=True,lakes=False, borders="2/thin", region=region, panel=True)
    data = pd.DataFrame({
    "x": gps_170_igs14_CASR_LOS["Lon"],
    "y": gps_170_igs14_CASR_LOS["Lat"],
    "east_velocity": gps_170_igs14_CASR_LOS["Ve"],
    "north_velocity": gps_170_igs14_CASR_LOS["Vn"],
    "east_sigma": gps_170_igs14_CASR_LOS["Std_e"],
    "north_sigma": gps_170_igs14_CASR_LOS["Std_n"],
    })
    fig.velo(data=data, pen="0.6p,black", line=True, spec="e0.02/0.39/18", vector="0.2c+p1p+e+gblack",)
    fig.velo(data=scale_df, pen="0.6p,black", line=True, spec="e0.02/0.39/18", vector="0.2c+p1p+e+gblack",)
    fig.text(text='IGS14 Ref Frame (CASR)', position="BL", offset="0.2c/0.2c")
    fig.text(text='30 mm/yr', x=scale_df["x"], y=scale_df["y"], offset="-0.1c/0.0c", justify="MR")  
    
    # GPS NA
    fig.coast(shorelines=True,lakes=False, borders="2/thin", region=region, panel=True)
    data = pd.DataFrame({
    "x": gps_170_na_CASR_LOS["Lon"],
    "y": gps_170_na_CASR_LOS["Lat"],
    "east_velocity": gps_170_na_CASR_LOS["Ve"],
    "north_velocity": gps_170_na_CASR_LOS["Vn"],
    "east_sigma": gps_170_na_CASR_LOS["Std_e"],
    "north_sigma": gps_170_na_CASR_LOS["Std_n"],
    })
    fig.velo(data=data, pen="0.6p,black", line=True, spec="e0.02/0.39/18", vector="0.2c+p1p+e+gblack",)
    fig.velo(data=scale_df, pen="0.6p,black", line=True, spec="e0.02/0.39/18", vector="0.2c+p1p+e+gblack",)
    fig.text(text='NA Ref Frame (CASR)', position="BL", offset="0.2c/0.2c")
    fig.text(text='30 mm/yr', x=scale_df["x"], y=scale_df["y"], offset="-0.1c/0.0c", justify="MR")  
    
    # GPS Different 
    fig.coast(shorelines=True,lakes=False, borders="2/thin", region=region, panel=True)
    
    scale_df = pd.DataFrame(
        data={
            "x": [-123.5],
            "y": [36.8],
            "east_velocity": [1],
            "north_velocity": [0],
            "east_sigma": [0],
            "north_sigma": [0],
        }
    )
    
    data = pd.DataFrame({
    "x": gps_170_na_CASR_LOS["Lon"],
    "y": gps_170_na_CASR_LOS["Lat"],
    "east_velocity": gps_170_na_CASR_LOS["Ve"] - gps_170_igs14_CASR_LOS["Ve"],
    "north_velocity": gps_170_na_CASR_LOS["Vn"] - gps_170_igs14_CASR_LOS["Vn"],
    "east_sigma": gps_170_na_CASR_LOS["Std_e"],
    "north_sigma": gps_170_na_CASR_LOS["Std_n"],
    })
    fig.velo(data=data, pen="0.6p,black", line=True, spec="e0.5/0.39/18", vector="0.2c+p1p+e+gblack",)
    fig.velo(data=scale_df, pen="0.6p,black", line=True, spec="e0.5/0.39/18", vector="0.2c+p1p+e+gblack",)
    fig.text(text='Difference (CASR) NA-IGS14', position="BL", offset="0.2c/0.2c")
    fig.text(text='1 mm/yr', x=scale_df["x"], y=scale_df["y"], offset="-0.1c/0.0c", justify="MR")  
    
fig.savefig(common_paths["fig_dir"]+f'Fig_14_{ref_station}_{track}_IDS15_NA_referenceFrame_comparison.jpg', transparent=False, crop=True, anti_alias=True, show=False)
fig.show()


### Begin plotting ###
fig = pygmt.Figure()
pygmt.config(FONT=11, FONT_TITLE=11, MAP_HEADING_OFFSET=0.1, PS_MEDIA="A3", FORMAT_GEO_MAP="ddd", MAP_FRAME_TYPE="plain")

with fig.subplot(nrows=3, ncols=3, figsize=("25c", "33c"), autolabel=True, sharex="b", sharey="l", frame="WSrt",):
    
    #-----------------------
    # GPS igs14
    #-----------------------
    fig.coast(shorelines=True,lakes=False, borders="2/thin", region=region, panel=True)
    
    vmin, vmax = np.nanpercentile(gps_170_igs14_LOS['LOS_Vel'], [5, 95])   # choose your percentiles
    pygmt.makecpt(cmap="batlow", series=[vmin, vmax])
    with pygmt.config(
        FONT_ANNOT_PRIMARY="18p,black", 
        FONT_ANNOT_SECONDARY="18p,black",
        FONT_LABEL="18p,black",
        ):
        fig.colorbar(position="jBL+o0.3c/0.7c+w3.0c/0.4c", frame=["xa+lLOS (mm/yr)"])
        
    fig.plot(y=gps_170_igs14_LOS["Lat"], x=gps_170_igs14_LOS["Lon"], style="c.15c", fill=gps_170_igs14_LOS['LOS_Vel'], cmap=True)
    fig.text(text='IGS14 Ref Frame', position="BL", offset="0.2c/0.2c")
   
    #-----------------------
    # GPS NA
    #-----------------------
    fig.coast(shorelines=True,lakes=False, borders="2/thin", region=region, panel=True)
    
    vmin, vmax = np.nanpercentile(gps_170_na_LOS['LOS_Vel'], [5, 95])   # choose your percentiles
    pygmt.makecpt(cmap="batlow", series=[vmin, vmax])
    with pygmt.config(
        FONT_ANNOT_PRIMARY="18p,black", 
        FONT_ANNOT_SECONDARY="18p,black",
        FONT_LABEL="18p,black",
        ):
        fig.colorbar(position="jBL+o0.3c/0.7c+w3.0c/0.4c", frame=["xa+lLOS (mm/yr)"])
    
    fig.plot(y=gps_170_na_LOS["Lat"], x=gps_170_na_LOS["Lon"], style="c.15c", fill=gps_170_na_LOS['LOS_Vel'], cmap=True)
    fig.text(text='NA Ref Frame', position="BL", offset="0.2c/0.2c")
    
    #-----------------------
    # GPS Different 
    #-----------------------
    fig.coast(shorelines=True,lakes=False, borders="2/thin", region=region, panel=True)
    
    pygmt.makecpt(cmap="batlow", series=[np.nanmin(gps_170_na_LOS['LOS_Vel']-gps_170_igs14_LOS['LOS_Vel']), np.nanmax(gps_170_na_LOS['LOS_Vel']-gps_170_igs14_LOS['LOS_Vel'])])
    with pygmt.config(
        FONT_ANNOT_PRIMARY="18p,black", 
        FONT_ANNOT_SECONDARY="18p,black",
        FONT_LABEL="18p,black",
        ):
        fig.colorbar(position="jBL+o0.3c/0.7c+w3.0c/0.4c", frame=["xa+lLOS (mm/yr)"])
    
    fig.plot(y=gps_170_na_LOS["Lat"], x=gps_170_na_LOS["Lon"], style="c.15c", fill=gps_170_na_LOS['LOS_Vel']-gps_170_igs14_LOS['LOS_Vel'], cmap=True)
    fig.text(text='Difference NA-IGS14', position="BL", offset="0.2c/0.2c")
   
    #-----------------------
    # GPS igs14
    #-----------------------
    fig.coast(shorelines=True,lakes=False, borders="2/thin", region=region, panel=True)
    
    vmin, vmax = np.nanpercentile(gps_170_igs14_LOS_CASR['LOS_Vel'], [5, 95])   # choose your percentiles
    pygmt.makecpt(cmap="batlow", series=[vmin, vmax])
    with pygmt.config(
        FONT_ANNOT_PRIMARY="18p,black", 
        FONT_ANNOT_SECONDARY="18p,black",
        FONT_LABEL="18p,black",
        ):
        fig.colorbar(position="jBL+o0.3c/0.7c+w3.0c/0.4c", frame=["xa+lLOS (mm/yr)"])
    
    fig.plot(y=gps_170_igs14_LOS_CASR["Lat"], x=gps_170_igs14_LOS_CASR["Lon"], style="c.15c", fill=gps_170_igs14_LOS_CASR['LOS_Vel'], cmap=True)
    fig.text(text='IGS14 1) Proj LOS 2) Ref CASR', position="BL", offset="0.2c/0.2c")
    
    #-----------------------
    # GPS NA
    #-----------------------
    fig.coast(shorelines=True,lakes=False, borders="2/thin", region=region, panel=True)
    
    vmin, vmax = np.nanpercentile(gps_170_na_LOS_CASR['LOS_Vel'], [5, 95])   # choose your percentiles
    pygmt.makecpt(cmap="batlow", series=[vmin, vmax])
    with pygmt.config(
        FONT_ANNOT_PRIMARY="18p,black", 
        FONT_ANNOT_SECONDARY="18p,black",
        FONT_LABEL="18p,black",
        ):
        fig.colorbar(position="jBL+o0.3c/0.7c+w3.0c/0.4c", frame=["xa+lLOS (mm/yr)"])
        
    fig.plot(y=gps_170_na_LOS_CASR["Lat"], x=gps_170_na_LOS_CASR["Lon"], style="c.15c", fill=gps_170_na_LOS_CASR['LOS_Vel'], cmap=True)
    fig.text(text='NA 1) Proj LOS 2) Ref CASR', position="BL", offset="0.2c/0.2c")
    
    #-----------------------
    # GPS Different 
    #-----------------------
    fig.coast(shorelines=True,lakes=False, borders="2/thin", region=region, panel=True)
    
    pygmt.makecpt(cmap="batlow", series=[np.nanmin(gps_170_na_LOS_CASR['LOS_Vel']-gps_170_igs14_LOS_CASR['LOS_Vel']), np.nanmax(gps_170_na_LOS_CASR['LOS_Vel']-gps_170_igs14_LOS_CASR['LOS_Vel'])])
    with pygmt.config(
        FONT_ANNOT_PRIMARY="18p,black", 
        FONT_ANNOT_SECONDARY="18p,black",
        FONT_LABEL="18p,black",
        ):
        fig.colorbar(position="jBL+o0.3c/0.7c+w3.0c/0.4c", frame=["xa+lLOS (mm/yr)"])
        
    fig.plot(y=gps_170_na_LOS_CASR["Lat"], x=gps_170_na_LOS_CASR["Lon"], style="c.15c", fill=gps_170_na_LOS_CASR['LOS_Vel']-gps_170_igs14_LOS_CASR['LOS_Vel'], cmap=True)
    fig.text(text='Difference (CASR) NA-IGS14', position="BL", offset="0.2c/0.2c") 
   
    #-----------------------
    # GPS igs14
    #-----------------------
    fig.coast(shorelines=True,lakes=False, borders="2/thin", region=region, panel=True)
    
    vmin, vmax = np.nanpercentile(gps_170_igs14_CASR_LOS['LOS_Vel'], [5, 95])   # choose your percentiles
    pygmt.makecpt(cmap="batlow", series=[vmin, vmax])
    with pygmt.config(
        FONT_ANNOT_PRIMARY="18p,black", 
        FONT_ANNOT_SECONDARY="18p,black",
        FONT_LABEL="18p,black",
        ):
        fig.colorbar(position="jBL+o0.3c/0.7c+w3.0c/0.4c", frame=["xa+lLOS (mm/yr)"])
    
    fig.plot(y=gps_170_igs14_CASR_LOS["Lat"], x=gps_170_igs14_CASR_LOS["Lon"], style="c.15c", fill=gps_170_igs14_CASR_LOS['LOS_Vel'], cmap=True)
    fig.text(text='IGS14 1) Ref CASR 2) Proj LOS', position="BL", offset="0.2c/0.2c")
    
    #-----------------------
    # GPS NA
    #-----------------------
    fig.coast(shorelines=True,lakes=False, borders="2/thin", region=region, panel=True)
    
    vmin, vmax = np.nanpercentile(gps_170_na_CASR_LOS['LOS_Vel'], [5, 95])   # choose your percentiles
    pygmt.makecpt(cmap="batlow", series=[vmin, vmax])
    with pygmt.config(
        FONT_ANNOT_PRIMARY="18p,black", 
        FONT_ANNOT_SECONDARY="18p,black",
        FONT_LABEL="18p,black",
        ):
        fig.colorbar(position="jBL+o0.3c/0.7c+w3.0c/0.4c", frame=["xa+lLOS (mm/yr)"])
        
    fig.plot(y=gps_170_na_CASR_LOS["Lat"], x=gps_170_na_CASR_LOS["Lon"], style="c.15c", fill=gps_170_na_CASR_LOS['LOS_Vel'], cmap=True)
    fig.text(text='NA 1) Ref CASR 2) Proj LOS', position="BL", offset="0.2c/0.2c")
    
    #-----------------------
    # GPS Different 
    #-----------------------
    fig.coast(shorelines=True,lakes=False, borders="2/thin", region=region, panel=True)
    
    pygmt.makecpt(cmap="batlow", series=[np.nanmin(gps_170_na_CASR_LOS['LOS_Vel']-gps_170_igs14_CASR_LOS['LOS_Vel']), np.nanmax(gps_170_na_CASR_LOS['LOS_Vel']-gps_170_igs14_CASR_LOS['LOS_Vel'])])
    with pygmt.config(
        FONT_ANNOT_PRIMARY="18p,black", 
        FONT_ANNOT_SECONDARY="18p,black",
        FONT_LABEL="18p,black",
        ):
        fig.colorbar(position="jBL+o0.3c/0.7c+w3.0c/0.4c", frame=["xa+lLOS (mm/yr)"])
        
    fig.plot(y=gps_170_na_CASR_LOS["Lat"], x=gps_170_na_CASR_LOS["Lon"], style="c.15c", fill=gps_170_na_CASR_LOS['LOS_Vel']-gps_170_igs14_CASR_LOS['LOS_Vel'], cmap=True)
    fig.text(text='Difference (CASR) NA-IGS14', position="BL", offset="0.2c/0.2c")

fig.savefig(common_paths["fig_dir"]+f'Fig_14_{ref_station}_{track}_IGS15_NA_LOS_comparison.jpg', transparent=False, crop=True, anti_alias=True, show=False)
fig.show()


