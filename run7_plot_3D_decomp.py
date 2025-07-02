#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 01:17:36 2025

@author: daniellelindsay
"""

import insar_utils as utils
from NC_ALOS2_filepaths import (paths_gps, paths_068, paths_169, paths_170, common_paths, decomp)
import numpy as np
import pygmt 
import pandas as pd 

dist = common_paths["dist"]
ref_station = common_paths["ref_station"]
ref_lat = common_paths["ref_lat"]
ref_lon = common_paths["ref_lon"]

# Input files 
asc_068_grd = paths_068["grd_mm"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"]
des_170_grd = paths_170["grd_mm"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"]
east_grd = paths_gps["visr"]["east"]
north_grd = paths_gps["visr"]["north"]

# Vertical files 
asc_up_grd = decomp["grd"]["asc_semi"]
des_up_grd = decomp["grd"]["des_semi"]
insar_up_grd = decomp["grd"]["insar_only_up"]
insar_gps_up_grd = decomp["grd"]["gps_insar_up"]


#####################
# Load InSAR
#####################

asc_up_df = utils.load_h5_data(paths_170["geo"]["geo_geometryRadar"], decomp["CASR"]["asc_semi"], "velocity")
des_up_df = utils.load_h5_data(paths_170["geo"]["geo_geometryRadar"], decomp["CASR"]["des_semi"], "velocity")
insar_up_df = utils.load_h5_data(paths_170["geo"]["geo_geometryRadar"], decomp["CASR"]["insar_only_up"], "velocity")
insar_gps_up = utils.load_h5_data(paths_170["geo"]["geo_geometryRadar"], decomp["CASR"]["gps_insar_up"], "velocity")

#####################
# Load in GPS and project UNR enu --> los
#####################

gps_df = utils.load_UNR_gps(paths_gps["170_enu"], ref_station)


#####################
# Find average InSAR velocity for each GPS point 
#####################
asc_up = utils.calculate_average_insar_velocity(gps_df, asc_up_df, dist)
des_up = utils.calculate_average_insar_velocity(gps_df, des_up_df, dist)
insar_up = utils.calculate_average_insar_velocity(gps_df, insar_up_df, dist)
insar_gps_up = utils.calculate_average_insar_velocity(gps_df, insar_gps_up, dist)

# Calculate residuals 
asc_up["residuals"] = asc_up['Vu'] - asc_up['insar_Vel']
des_up["residuals"] = des_up['Vu'] - des_up['insar_Vel']
insar_up["residuals"] = insar_up['Vu'] - insar_up['insar_Vel']
insar_gps_up["residuals"] = insar_gps_up['Vu'] - insar_gps_up['insar_Vel']

#####################
# Calculate rmse, r2 and linear fit
#####################
rmse_asc_up, r2_asc_up, slope_asc_up, intercept_asc_up = utils.calculate_rmse_r2_and_linear_fit(asc_up['Vu'], asc_up['insar_Vel'])
rmse_des_up, r2_des_up, slope_des_up, intercept_des_up = utils.calculate_rmse_r2_and_linear_fit(des_up['Vu'], des_up['insar_Vel'])
rmse_insar_up, r2_insar_up, slope_insar_up, intercept_insar_up = utils.calculate_rmse_r2_and_linear_fit(insar_up['Vu'], insar_up['insar_Vel'])
rmse_insar_gps_up, r2_insar_gps_up, slope_insar_gps_up, intercept_insar_gps_up = utils.calculate_rmse_r2_and_linear_fit(insar_gps_up['Vu'], insar_gps_up['insar_Vel'])

# ------------------------
# Print 2D‐ramp validation metrics and residual‐percent
# ------------------------

# 1) Ascending pass
print("\nAscending Semi-Up (Vu vs insar_Vel):")
print(f"RMSE:      {rmse_asc_up:.3f}")
print(f"R²:        {r2_asc_up:.3f}")
print(f"Slope:     {slope_asc_up:.3f}")
print(f"Intercept: {intercept_asc_up:.3f}")

# 2) Descending pass
print("\nDescending Semi-Up (Vu vs insar_Vel):")
print(f"RMSE:      {rmse_des_up:.3f}")
print(f"R²:        {r2_des_up:.3f}")
print(f"Slope:     {slope_des_up:.3f}")
print(f"Intercept: {intercept_des_up:.3f}")

# 3) InSAR only
print("\nInSAR only (Vu vs insar_Vel):")
print(f"RMSE:      {rmse_insar_up:.3f}")
print(f"R²:        {r2_insar_up:.3f}")
print(f"Slope:     {slope_insar_up:.3f}")
print(f"Intercept: {intercept_insar_up:.3f}")

# 4) InSAR+GPS combined
print("\nInSAR+GPS combined (Vu vs insar_Vel):")
print(f"RMSE:      {rmse_insar_gps_up:.3f}")
print(f"R²:        {r2_insar_gps_up:.3f}")
print(f"Slope:     {slope_insar_gps_up:.3f}")
print(f"Intercept: {intercept_insar_gps_up:.3f}")
 

###########################
# Plot Results 4 x 2 
###########################

size = "M3.4c"
fig_region="-125/-121.0/37.0/42.25" 
style="c.03c"
res_style="c.1c"
vmin, vmax = -20, 20

fig = pygmt.Figure()
# Begin plot
pygmt.config(FORMAT_GEO_MAP="ddd", MAP_FRAME_TYPE="plain", FONT=9, FONT_TITLE=10, FONT_SUBTITLE = 10, MAP_TITLE_OFFSET= "-7p")

with fig.subplot(nrows=2, ncols=2, figsize=("7.5c", "12.0c"), autolabel=True,sharex="b", sharey="l",
                  frame=["WSrt", "xa", "ya"], margins=["0.2c", "0.2c"]):
    pygmt.makecpt(cmap="vik", series=[vmin, vmax])
    
    fig.basemap(    region=[fig_region], projection=size, panel=True, frame=["Wsrt", "xa", "ya"])
    fig.grdimage(   region=[fig_region], projection=size, grid=asc_068_grd, cmap=True, nan_transparent=True)
    fig.plot(x=ref_lon, y=ref_lat, style="s.15c", fill="black", pen="0.8p,black", region=[fig_region],projection= size)
    fig.coast(area_thresh=5000, borders=1,  shorelines=True, region=[fig_region], projection=size, )
    fig.text(text="Ascending", position="BL", offset="0.2c/0.2c", justify="BL", fill="white", region=[fig_region], projection=size)
    with pygmt.config(FONT_ANNOT_PRIMARY="16p,black", FONT_ANNOT_SECONDARY="16p,black", FONT_LABEL="16p,black"):
        fig.colorbar(position="jBL+o0.2c/0.7c+w1.8c/0.25c", frame=["xa","y+lmm/yr"], projection=size)
    
    fig.text(
    text="Input Data",  # "@@" gives "@" in GMT or PyGMT
    font="10p,Helvetica-Bold,black", 
    position="TR",  # Top Center
    justify="MC",  # Middle Center
    offset="0.25c/0.35c",
    no_clip=True,  # Allow plotting outside of the map or plot frame
    )
    
    fig.text(
    text="InSAR",  # "@@" gives "@" in GMT or PyGMT
    font="10p,Helvetica-Bold,black", 
    position="ML",  # Top Center
    justify="MC",  # Middle Center
    offset="-1.0c/0.0c",
    no_clip=True,  # Allow plotting outside of the map or plot frame
    angle = 90)
    
    fig.basemap(region=[fig_region], projection=size, panel=True, frame=["wsrt", "xa", "ya"])
    fig.grdimage(   region=[fig_region], projection=size, grid=des_170_grd, cmap=True, nan_transparent=True)
    fig.plot(x=ref_lon, y=ref_lat, style="s.15c", fill="black", pen="0.8p,black", region=[fig_region],projection= size)
    fig.coast(area_thresh=5000, borders=1, shorelines=True, region=[fig_region], projection=size, )
    fig.text(text="Descending", position="BL", offset="0.2c/0.2c", justify="BL",  fill="white", region=[fig_region], projection=size)
    with pygmt.config(FONT_ANNOT_PRIMARY="16p,black", FONT_ANNOT_SECONDARY="16p,black", FONT_LABEL="16p,black"):
        fig.colorbar(position="jBL+o0.2c/0.7c+w1.8c/0.25c", frame=["xa","y+lmm/yr"], projection=size)
        
    fig.basemap(region=[fig_region], projection=size, panel=True, frame=["WSrt", "xa", "ya"])
    fig.grdimage(   region=[fig_region], projection=size, grid=north_grd, cmap=True, nan_transparent=True)
    
    fig.plot(x=ref_lon, y=ref_lat, style="s.15c", fill="black", pen="0.8p,black", region=[fig_region],projection= size)
    fig.coast(area_thresh=5000, borders=1,   shorelines=True, region=[fig_region], projection=size, )
    fig.text(text="Interp. North", position="BL", offset="0.2c/0.2c", justify="BL",  fill="white", region=[fig_region], projection=size)
    with pygmt.config(FONT_ANNOT_PRIMARY="16p,black", FONT_ANNOT_SECONDARY="16p,black", FONT_LABEL="16p,black"):
        fig.colorbar(position="jBL+o0.2c/0.7c+w1.8c/0.25c", frame=["xa","y+lmm/yr"], projection=size)
    
    fig.text(
    text="GNSS",  # "@@" gives "@" in GMT or PyGMT
    font="10p,Helvetica-Bold,black", 
    position="ML",  # Top Center
    justify="MC",  # Middle Center
    offset="-1.0c/0.0c",
    no_clip=True,  # Allow plotting outside of the map or plot frame
    angle = 90)
    
    fig.basemap(region=[fig_region], projection=size, panel=True, frame=["wSrt", "xa", "ya"])
    fig.grdimage(   region=[fig_region], projection=size, grid=east_grd, cmap=True, nan_transparent=True)
    fig.plot(x=ref_lon, y=ref_lat, style="s.15c", fill="black", pen="0.8p,black", region=[fig_region],projection= size)
    fig.coast(area_thresh=5000, borders=1,   shorelines=True, region=[fig_region], projection=size,)
    fig.text(text="Interp. East", position="BL", offset="0.2c/0.2c", justify="BL",  fill="white", region=[fig_region], projection=size)
    with pygmt.config(FONT_ANNOT_PRIMARY="16p,black", FONT_ANNOT_SECONDARY="16p,black", FONT_LABEL="16p,black"):
        fig.colorbar(position="jBL+o0.2c/0.7c+w1.8c/0.25c", frame=["xa","y+lmm/yr"], projection=size)
    
fig.shift_origin(yshift="0c", xshift="8.7c")

with fig.subplot(nrows=2, ncols=4, figsize=("15c", "12.0c"), autolabel="e)",sharex="b", sharey="l",
                  frame=["WSrt", "xa", "ya"], margins=["0.2c", "0.2c"]):
    
    pygmt.makecpt(cmap="vik", series=[-10, 10])
    
    fig.basemap(region=[fig_region], projection=size, panel=True, frame=["Wsrt", "xa", "ya"])
    fig.grdimage(   region=[fig_region], projection=size, grid=asc_up_grd, cmap=True, nan_transparent=True)
    fig.plot(x=des_up["Lon"], y=des_up["Lat"], fill=des_up["Vu"], pen="black", cmap=True, style=res_style,  region=[fig_region],projection= size)
    fig.plot(x=ref_lon, y=ref_lat, style="s.15c", fill="black", pen="0.8p,black", region=[fig_region],projection= size)
    fig.text(x=ref_lon, y=ref_lat, text="%s" % ref_station,  font="10p,Helvetica,black", offset="-0.5c/-0.25c+v", justify="RM", region=[fig_region],projection= size)
    fig.coast(area_thresh=5000, borders=1,  shorelines=True, region=[fig_region], projection=size, frame = ["+tSemi-vertical Asc."])
    
    fig.text(
    text="Vertical Velocity",  # "@@" gives "@" in GMT or PyGMT
    font="10p,Helvetica-Bold,black", 
    position="ML",  # Top Center
    justify="MC",  # Middle Center
    offset="-1.0c/0.0c",
    no_clip=True,  # Allow plotting outside of the map or plot frame
    angle = 90)
    
    fig.basemap(    region=[fig_region], projection=size, panel=True, frame=["wsrt", "xa", "ya"])
    fig.grdimage(   region=[fig_region], projection=size, grid=des_up_grd, cmap=True, nan_transparent=True)
    fig.plot(x=des_up["Lon"], y=des_up["Lat"], fill=des_up["Vu"], pen="black", cmap=True, style=res_style,  region=[fig_region],projection= size)
    fig.plot(x=ref_lon, y=ref_lat, style="s.15c", fill="black", pen="0.8p,black", region=[fig_region],projection= size)
    fig.coast(area_thresh=5000, borders=1, shorelines=True, region=[fig_region], projection=size, frame = ["+tSemi-vertical Des."])
        
    fig.basemap(    region=[fig_region], projection=size, panel=True, frame=["wsrt", "xa", "ya"])
    fig.grdimage(   region=[fig_region], projection=size, grid=insar_up_grd, cmap=True, nan_transparent=True)
    fig.plot(x=des_up["Lon"], y=des_up["Lat"], fill=des_up["Vu"], pen="black", cmap=True, style=res_style,  region=[fig_region],projection= size)
    fig.plot(x=ref_lon, y=ref_lat, style="s.15c", fill="black", pen="0.8p,black", region=[fig_region],projection= size)
    fig.coast(area_thresh=5000, borders=1,   shorelines=True, region=[fig_region], projection=size, frame = ["+tInSAR-only"])
    
    fig.basemap(region=[fig_region], projection=size, panel=True, frame=["wsrt", "xa", "ya"])
    fig.grdimage(   region=[fig_region], projection=size, grid=insar_gps_up_grd, cmap=True, nan_transparent=True)
    fig.plot(x=des_up["Lon"], y=des_up["Lat"], fill=des_up["Vu"], pen="black", cmap=True, style=res_style,  region=[fig_region],projection= size)
    fig.plot(x=ref_lon, y=ref_lat, style="s.15c", fill="black", pen="0.8p,black", region=[fig_region],projection= size)
    fig.coast(area_thresh=5000, borders=1,   shorelines=True, region=[fig_region], projection=size, frame = ["+tInSAR+GNSS"])

    with pygmt.config(
        FONT_ANNOT_PRIMARY="18p,black", 
        FONT_ANNOT_SECONDARY="18p,black",
        FONT_LABEL="18p,black",
        ):
        fig.colorbar(position="jCL+o3.7c/0c+w4c/0.4c", frame=["xa+lVelocity (mm/yr)", "y"], projection = size)
    
    fig.basemap(region=[fig_region], projection=size, panel=True, frame=["WSrt", "xa", "ya"])
    pygmt.makecpt(cmap="roma", series=[ -5, 5, 1])
    fig.plot(x=asc_up["Lon"], y=asc_up["Lat"], fill=asc_up["residuals"], pen="black", cmap=True, style=res_style,  region=[fig_region],projection= size)
    fig.plot(x=ref_lon, y=ref_lat, style="s.15c", fill="black", pen="0.8p,black", region=[fig_region],projection= size)
    fig.coast(area_thresh=5000, borders=1, shorelines=True, region=[fig_region], projection=size, )
    fig.text(text=f'{rmse_asc_up:.2f} mm/yr', position="BL", offset="0.2c/0.2c", region=[fig_region], projection=size)
    
    fig.text(
    text="Residuals",  # "@@" gives "@" in GMT or PyGMT
    font="10p,Helvetica-Bold,black", 
    position="ML",  # Top Center
    justify="MC",  # Middle Center
    offset="-1.0c/0.0c",
    no_clip=True,  # Allow plotting outside of the map or plot frame
    angle = 90)
    
    fig.basemap(region=[fig_region], projection=size, panel=True, frame=["wSrt", "xa", "ya"])
    pygmt.makecpt(cmap="roma", series=[ -5, 5, 1])
    fig.plot(x=des_up["Lon"], y=des_up["Lat"], fill=des_up["residuals"], pen="black", cmap=True, style=res_style,  region=[fig_region],projection= size)
    fig.plot(x=ref_lon, y=ref_lat, style="s.15c", fill="black", pen="0.8p,black", region=[fig_region],projection= size)
    fig.coast(area_thresh=5000, borders=1, shorelines=True, region=[fig_region], projection=size, )
    fig.text(text=f'{rmse_des_up:.2f} mm/yr', position="BL", offset="0.2c/0.2c", region=[fig_region], projection=size)    
    
    fig.basemap(region=[fig_region], projection=size, panel=True, frame=["wSrt", "xa", "ya"])
    pygmt.makecpt(cmap="roma", series=[ -5, 5, 1])
    fig.plot(x=insar_up["Lon"], y=insar_up["Lat"], fill=insar_up["residuals"], pen="black", cmap=True, style=res_style,  region=[fig_region],projection= size)
    fig.text(text=f'{rmse_insar_up:.2f} mm/yr', position="BL", offset="0.2c/0.2c", region=[fig_region], projection=size)
    fig.coast(area_thresh=5000, borders=1,  shorelines=True, region=[fig_region], projection=size,)
    
    fig.basemap(region=[fig_region], projection=size, panel=True, frame=["wSrt", "xa", "ya"])
    pygmt.makecpt(cmap="roma", series=[ -5, 5, 1])
    fig.plot(x=insar_gps_up["Lon"], y=insar_gps_up["Lat"], fill=insar_gps_up["residuals"], pen="black", cmap=True, style=res_style, region=[fig_region],projection= size)
    fig.text(text=f'{rmse_insar_gps_up:.2f} mm/yr', position="BL", offset="0.2c/0.2c", region=[fig_region], projection=size)
    fig.coast(area_thresh=5000, borders=1,   shorelines=True, region=[fig_region], projection=size,)
    
    pygmt.makecpt(cmap="roma", series=[ -5, 5, 1])
    with pygmt.config(
        FONT_ANNOT_PRIMARY="18p,black", 
        FONT_ANNOT_SECONDARY="18p,black",
        FONT_LABEL="18p,black",
        ):
        fig.colorbar(position="jCL+o3.7c/0.0c+w4.0c/0.4c", frame=["xa+lResidual (mm/yr)", "y"], projection = size)
        
fig.savefig(common_paths['fig_dir']+'Fig_4_3D_results_170_068.png', transparent=False, crop=True, anti_alias=True, show=False)
fig.savefig(common_paths['fig_dir']+'Fig_4_3D_results_170_068.pdf', transparent=False, crop=True, anti_alias=True, show=False)
fig.show()
