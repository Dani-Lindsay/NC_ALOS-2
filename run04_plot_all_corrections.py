#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 22:03:37 2024

@author: daniellelindsay
"""
import insar_utils as utils
import pygmt
import numpy as np
import math
from NC_ALOS2_filepaths import (common_paths, paths_068, paths_169, paths_170, paths_gps)

ref_station = common_paths["ref_station"]

# Define the distance threshold for averaging InSAR velocities
distance_threshold = common_paths["dist"]

# Pixel size for geocoding
lat_step = common_paths["lat_step"]
lon_step = common_paths["lon_step"]

# Scaling unit.
unit = 1000

# ------------------------##
# Load all corrections. 
# ------------------------#

geo_169 = paths_169["geo"]["geo_geometryRadar"]
geo_170 = paths_170["geo"]["geo_geometryRadar"]
geo_068 = paths_068["geo"]["geo_geometryRadar"]

# ------------------------
# Track 068
# ------------------------
vel_068_df       = utils.load_h5_data(geo_068, paths_068["geo"]["geo_velocity_msk"], "velocity")
vel_SET_068_df   = utils.load_h5_data(geo_068, paths_068["geo"]["geo_velocity_SET_msk"], "velocity")
vel_ERA5_068_df  = utils.load_h5_data(geo_068, paths_068["geo"]["geo_velocity_SET_ERA5_msk"], "velocity")
vel_demErr_068_df = utils.load_h5_data(geo_068, paths_068["geo"]["geo_velocity_SET_ERA5_demErr_msk"], "velocity")
vel_ITRF14_068_df = utils.load_h5_data(geo_068, paths_068["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14_msk"], "velocity")
vel_deramp_068_df = utils.load_h5_data(geo_068, paths_068["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"], "velocity")

SET_068_df    = utils.load_h5_data(geo_068, paths_068["geo"]["diff_SET"], "velocity")
ERA5_068_df   = utils.load_h5_data(geo_068, paths_068["geo"]["diff_ERA5"], "velocity")
demErr_068_df = utils.load_h5_data(geo_068, paths_068["geo"]["diff_demErr"], "velocity")
ITRF14_068_df = utils.load_h5_data(geo_068, paths_068["geo"]["diff_ITRF14"], "velocity")
deramp_068_df = utils.load_h5_data(geo_068, paths_068["geo"]["diff_deramp"], "velocity")

vel_068_grd       = paths_068["grd_mm"]["geo_velocity_msk"]
vel_SET_068_grd   = paths_068["grd_mm"]["geo_velocity_SET_msk"]
vel_ERA5_068_grd  = paths_068["grd_mm"]["geo_velocity_SET_ERA5_msk"]
vel_demErr_068_grd = paths_068["grd_mm"]["geo_velocity_SET_ERA5_demErr_msk"]
vel_ITRF14_068_grd = paths_068["grd_mm"]["geo_velocity_SET_ERA5_demErr_ITRF14_msk"]
vel_deramp_068_grd = paths_068["grd_mm"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"]

SET_068_grd    = paths_068["grd_mm"]["diff_SET"]
ERA5_068_grd   = paths_068["grd_mm"]["diff_ERA5"]
demErr_068_grd = paths_068["grd_mm"]["diff_demErr"]
ITRF14_068_grd = paths_068["grd_mm"]["diff_ITRF14"]
deramp_068_grd = paths_068["grd_mm"]["diff_deramp"]

slantRange_068_df = utils.load_h5_data(geo_068, geo_068, "slantRangeDistance")

# ------------------------
# Track 169
# ------------------------
vel_169_df       = utils.load_h5_data(geo_169, paths_169["geo"]["geo_velocity_msk"], "velocity")
vel_SET_169_df   = utils.load_h5_data(geo_169, paths_169["geo"]["geo_velocity_SET_msk"], "velocity")
vel_ERA5_169_df  = utils.load_h5_data(geo_169, paths_169["geo"]["geo_velocity_SET_ERA5_msk"], "velocity")
vel_demErr_169_df = utils.load_h5_data(geo_169, paths_169["geo"]["geo_velocity_SET_ERA5_demErr_msk"], "velocity")
vel_ITRF14_169_df = utils.load_h5_data(geo_169, paths_169["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14_msk"], "velocity")
vel_deramp_169_df = utils.load_h5_data(geo_169, paths_169["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"], "velocity")

SET_169_df    = utils.load_h5_data(geo_169, paths_169["geo"]["diff_SET"], "velocity")
ERA5_169_df   = utils.load_h5_data(geo_169, paths_169["geo"]["diff_ERA5"], "velocity")
demErr_169_df = utils.load_h5_data(geo_169, paths_169["geo"]["diff_demErr"], "velocity")
ITRF14_169_df = utils.load_h5_data(geo_169, paths_169["geo"]["diff_ITRF14"], "velocity")
deramp_169_df = utils.load_h5_data(geo_169, paths_169["geo"]["diff_deramp"], "velocity")

vel_169_grd        = paths_169["grd_mm"]["geo_velocity_msk"]
vel_SET_169_grd    = paths_169["grd_mm"]["geo_velocity_SET_msk"]
vel_ERA5_169_grd   = paths_169["grd_mm"]["geo_velocity_SET_ERA5_msk"]
vel_demErr_169_grd = paths_169["grd_mm"]["geo_velocity_SET_ERA5_demErr_msk"]
vel_ITRF14_169_grd = paths_169["grd_mm"]["geo_velocity_SET_ERA5_demErr_ITRF14_msk"]
vel_deramp_169_grd = paths_169["grd_mm"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"]

SET_169_grd    = paths_169["grd_mm"]["diff_SET"]
ERA5_169_grd   = paths_169["grd_mm"]["diff_ERA5"]
demErr_169_grd = paths_169["grd_mm"]["diff_demErr"]
ITRF14_169_grd = paths_169["grd_mm"]["diff_ITRF14"]
deramp_169_grd = paths_169["grd_mm"]["diff_deramp"]

slantRange_169_df = utils.load_h5_data(geo_169, geo_169, "slantRangeDistance")

# ------------------------
# Track 170
# ------------------------
vel_170_df       = utils.load_h5_data(geo_170, paths_170["geo"]["geo_velocity_msk"], "velocity")
vel_SET_170_df   = utils.load_h5_data(geo_170, paths_170["geo"]["geo_velocity_SET_msk"], "velocity")
vel_ERA5_170_df  = utils.load_h5_data(geo_170, paths_170["geo"]["geo_velocity_SET_ERA5_msk"], "velocity")
vel_demErr_170_df = utils.load_h5_data(geo_170, paths_170["geo"]["geo_velocity_SET_ERA5_demErr_msk"], "velocity")
vel_ITRF14_170_df = utils.load_h5_data(geo_170, paths_170["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14_msk"], "velocity")
vel_deramp_170_df = utils.load_h5_data(geo_170, paths_170["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"], "velocity")

SET_170_df    = utils.load_h5_data(geo_170, paths_170["geo"]["diff_SET"], "velocity")
ERA5_170_df   = utils.load_h5_data(geo_170, paths_170["geo"]["diff_ERA5"], "velocity")
demErr_170_df = utils.load_h5_data(geo_170, paths_170["geo"]["diff_demErr"], "velocity")
ITRF14_170_df = utils.load_h5_data(geo_170, paths_170["geo"]["diff_ITRF14"], "velocity")
deramp_170_df = utils.load_h5_data(geo_170, paths_170["geo"]["diff_deramp"], "velocity")

vel_170_grd        = paths_170["grd_mm"]["geo_velocity_msk"]
vel_SET_170_grd    = paths_170["grd_mm"]["geo_velocity_SET_msk"]
vel_ERA5_170_grd   = paths_170["grd_mm"]["geo_velocity_SET_ERA5_msk"]
vel_demErr_170_grd = paths_170["grd_mm"]["geo_velocity_SET_ERA5_demErr_msk"]
vel_ITRF14_170_grd = paths_170["grd_mm"]["geo_velocity_SET_ERA5_demErr_ITRF14_msk"]
vel_deramp_170_grd = paths_170["grd_mm"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"]

SET_170_grd    = paths_170["grd_mm"]["diff_SET"]
ERA5_170_grd   = paths_170["grd_mm"]["diff_ERA5"]
demErr_170_grd = paths_170["grd_mm"]["diff_demErr"]
ITRF14_170_grd = paths_170["grd_mm"]["diff_ITRF14"]
deramp_170_grd = paths_170["grd_mm"]["diff_deramp"]

slantRange_170_df = utils.load_h5_data(geo_170, geo_170, "slantRangeDistance")

insar_068_dfs = [
    (vel_068_df, 'vel_068'),
    (vel_SET_068_df, 'vel_SET_068'),
    (vel_ERA5_068_df, 'vel_ERA5_068'),
    (vel_demErr_068_df, 'vel_demErr_068'),
    (vel_ITRF14_068_df, 'vel_ITRF14_068'),
    (vel_deramp_068_df, 'vel_deramp_068'), 
    (slantRange_068_df, 'slantRange_068')]

insar_169_dfs = [
    (vel_169_df, 'vel_169'),
    (vel_SET_169_df, 'vel_SET_169'),
    (vel_ERA5_169_df, 'vel_ERA5_169'),
    (vel_demErr_169_df, 'vel_demErr_169'),
    (vel_ITRF14_169_df, 'vel_ITRF14_169'),
    (vel_deramp_169_df, 'vel_deramp_169'), 
    (slantRange_169_df, 'slantRange_169')]

insar_170_dfs = [
    (vel_170_df, 'vel_170'),
    (vel_SET_170_df, 'vel_SET_170'),
    (vel_ERA5_170_df, 'vel_ERA5_170'),
    (vel_demErr_170_df, 'vel_demErr_170'),
    (vel_ITRF14_170_df, 'vel_ITRF14_170'),
    (vel_deramp_170_df, 'vel_deramp_170'), 
    (slantRange_170_df, 'slantRange_170')]

itrf_LOS_169 = paths_169["geo"]["ITRF_LOS"]
itrf_LOS_170 = paths_170["geo"]["ITRF_LOS"]
itrf_LOS_068 = paths_068["geo"]["ITRF_LOS"]

# ------------------------
# Load in GPS, correct plate motion, project UNR enu --> los
# ------------------------

gps_169 = utils.load_UNR_gps(paths_gps["169_enu_ISG14"], ref_station)
gps_170 = utils.load_UNR_gps(paths_gps["170_enu_ISG14"], ref_station)
gps_068 = utils.load_UNR_gps(paths_gps["068_enu_ISG14"], ref_station)

# Set lat and lon for plotting from the gps file. 
ref_lat = gps_169.loc[gps_169["StaID"] == ref_station, "Lat"].values
ref_lon = gps_169.loc[gps_169["StaID"] == ref_station, "Lon"].values

# Projecy GPS to LOS
gps_169 = utils.calculate_gps_los(gps_169, vel_169_df)
gps_170 = utils.calculate_gps_los(gps_170, vel_170_df)
gps_068 = utils.calculate_gps_los(gps_068, vel_068_df)

# Correction GPS for plate motion
gps_169 = utils.gps_LOS_correction_plate_motion(geo_169, itrf_LOS_169, gps_169, ref_station, unit)
gps_170 = utils.gps_LOS_correction_plate_motion(geo_170, itrf_LOS_170, gps_170, ref_station, unit)
gps_068 = utils.gps_LOS_correction_plate_motion(geo_068, itrf_LOS_068, gps_068, ref_station, unit)

# ------------------------
# Find average InSAR velocity for each GPS point 
# ------------------------
# Initialize an empty dictionary to store RMSE and R² values
results_169_dict = {}
results_170_dict = {}
results_068_dict = {}

# Loop through all InSAR datasets
for insar_df, insar_name in insar_169_dfs:
    
    # Step 1: Calculate the average InSAR velocity for each GPS point
    gps_169 = utils.calculate_average_insar_velocity(gps_169, insar_df, distance_threshold)
    gps_169['insar_Vel'] = gps_169['insar_Vel'] * unit
    
    # Step 2: Calculate RMSE and R² between 'UNR_Vel' and the calculated InSAR velocity    
    rmse, r2, slope, intercept = utils.calculate_rmse_r2_and_linear_fit(gps_169['LOS_Vel'], gps_169['insar_Vel'])
    results_169_dict[insar_name] = {
        'rmse': rmse,
        'r2': r2}
        
    # Step 3: Add the InSAR velocity as a new column in the GPS DataFrame
    gps_169[insar_name] = gps_169['insar_Vel']

# Loop through all InSAR datasets
for insar_df, insar_name in insar_170_dfs:
    
    # Step 1: Calculate the average InSAR velocity for each GPS point
    gps_170 = utils.calculate_average_insar_velocity(gps_170, insar_df, distance_threshold)
    gps_170['insar_Vel'] = gps_170['insar_Vel'] * unit
        
    # Step 2: Calculate RMSE and R² between 'UNR_Vel' and the calculated InSAR velocity    
    rmse, r2, slope, intercept = utils.calculate_rmse_r2_and_linear_fit(gps_170['LOS_Vel'], gps_170['insar_Vel'])
    results_170_dict[insar_name] = {
        'rmse': rmse,
        'r2': r2}
        
    # Step 3: Add the InSAR velocity as a new column in the GPS DataFrame
    gps_170[insar_name] = gps_170['insar_Vel']
    
# Loop through all InSAR datasets
for insar_df, insar_name in insar_068_dfs:
    
    # Step 1: Calculate the average InSAR velocity for each GPS point
    gps_068 = utils.calculate_average_insar_velocity(gps_068, insar_df, distance_threshold)
    gps_068['insar_Vel'] = gps_068['insar_Vel'] * unit
        
    # Step 2: Calculate RMSE and R² between 'UNR_Vel' and the calculated InSAR velocity    
    rmse, r2, slope, intercept = utils.calculate_rmse_r2_and_linear_fit(gps_068['LOS_Vel'], gps_068['insar_Vel'])
    results_068_dict[insar_name] = {
        'rmse': rmse,
        'r2': r2}
        
    # Step 3: Add the InSAR velocity as a new column in the GPS DataFrame
    gps_068[insar_name] = gps_068['insar_Vel']
    

# ------------------------
# Residual Ramp stats
# ------------------------

def slope_km_to_mm100km_and_swath(slope_mm_per_km, swath_km=350):
    """
    Convert a slope in mm per km into:
      1. mm per 100 km
      2. mm over a swath (default 350 km)
    """
    mm_per_100km = slope_mm_per_km * 100
    mm_over_swath = slope_mm_per_km * swath_km
    return mm_per_100km, mm_over_swath

gps_169["residual_ITRF14_169"] = gps_169["LOS_Vel"] - gps_169["vel_ITRF14_169"]
gps_170["residual_ITRF14_170"] = gps_170["LOS_Vel"] - gps_170["vel_ITRF14_170"]
gps_068["residual_ITRF14_068"] = gps_068["LOS_Vel"] - gps_068["vel_ITRF14_068"]

gps_169['slantRange_169_km'] = gps_169['slantRange_169']/unit/1000 # In loop it mutiplies by unit, then 1000 to go from meters to km
gps_170['slantRange_170_km'] = gps_170['slantRange_170']/unit/1000 # In loop it mutiplies by unit, then 1000 to go from meters to km
gps_068['slantRange_068_km'] = gps_068['slantRange_068']/unit/1000 # In loop it mutiplies by unit, then 1000 to go from meters to km

# slantRange in metres (or mm), Inc in degrees
gps_169['groundRange_169'] = (gps_169['slantRange_169_km'] * np.sin(np.deg2rad(gps_169['Inc'])))
gps_170['groundRange_170'] = (gps_170['slantRange_170_km'] * np.sin(np.deg2rad(gps_170['Inc'])))
gps_068['groundRange_068'] = (gps_068['slantRange_068_km'] * np.sin(np.deg2rad(gps_068['Inc'])))

res_rmse_169, res_r2_169, res_slope_169, res_intercept_169 = utils.calculate_rmse_r2_and_linear_fit(gps_169['groundRange_169'], gps_169['residual_ITRF14_169'])
res_rmse_170, res_r2_170, res_slope_170, res_intercept_170 = utils.calculate_rmse_r2_and_linear_fit(gps_170['groundRange_170'], gps_170['residual_ITRF14_170'])
res_rmse_068, res_r2_068, res_slope_068, res_intercept_068 = utils.calculate_rmse_r2_and_linear_fit(gps_068['groundRange_068'], gps_068['residual_ITRF14_068'])

# — print longitude stats —
print(f"groundRange — Site 169: R² = {res_r2_169:.3f} ({res_r2_169*100:.1f}% of variance)")
print(f"groundRange — Site 170: R² = {res_r2_170:.3f} ({res_r2_170*100:.1f}% of variance)")
print(f"groundRange — Site 068: R² = {res_r2_068:.3f} ({res_r2_068*100:.1f}% of variance)")

mean_res_169 = gps_169['residual_ITRF14_169'].mean()
mean_res_170 = gps_170['residual_ITRF14_170'].mean()
mean_res_068 = gps_068['residual_ITRF14_068'].mean()

print(f"Mean residual — Site 169: {mean_res_169:.2f} mm/yr")
print(f"Mean residual — Site 170: {mean_res_170:.2f} mm/yr")
print(f"Mean residual — Site 068: {mean_res_068:.2f} mm/yr")

mean_lat_169 = gps_169['Lat'].mean()
mean_lat_170 = gps_170['Lat'].mean()
mean_lat_068 = gps_068['Lat'].mean()

ramp100_169, ramp350_169 = slope_km_to_mm100km_and_swath(res_slope_169, swath_km=350)
ramp100_170, ramp350_170 = slope_km_to_mm100km_and_swath(res_slope_170, swath_km=350)
ramp100_068, ramp350_068 = slope_km_to_mm100km_and_swath(res_slope_068, swath_km=350)

print(f"Track 169: Slope {res_slope_169:.4f} mm/° → {ramp100_169:.2f} mm/100 km → {ramp350_169:.2f} mm over 350 km")
print(f"Track 170: Slope {res_slope_170:.4f} mm/° → {ramp100_170:.2f} mm/100 km → {ramp350_170:.2f} mm over 350 km")
print(f"Track 068: Slope {res_slope_068:.4f} mm/° → {ramp100_068:.2f} mm/100 km → {ramp350_068:.2f} mm over 350 km")

print(f"Site 169: RMSE improved by {((results_169_dict['vel_ITRF14_169']['rmse'] - results_169_dict['vel_deramp_169']['rmse']) / results_169_dict['vel_ITRF14_169']['rmse'] * 100):.1f}% "
      f"(from {results_169_dict['vel_ITRF14_169']['rmse']:.2f} to {results_169_dict['vel_deramp_169']['rmse']:.2f} mm/yr)")

print(f"Site 170: RMSE improved by {((results_170_dict['vel_ITRF14_170']['rmse'] - results_170_dict['vel_deramp_170']['rmse']) / results_170_dict['vel_ITRF14_170']['rmse'] * 100):.1f}% "
      f"(from {results_170_dict['vel_ITRF14_170']['rmse']:.2f} to {results_170_dict['vel_deramp_170']['rmse']:.2f} mm/yr)")

print(f"Site 068: RMSE improved by {((results_068_dict['vel_ITRF14_068']['rmse'] - results_068_dict['vel_deramp_068']['rmse']) / results_068_dict['vel_ITRF14_068']['rmse'] * 100):.1f}% "
      f"(from {results_068_dict['vel_ITRF14_068']['rmse']:.2f} to {results_068_dict['vel_deramp_068']['rmse']:.2f} mm/yr)")


# ------------------------
# Plotting function
# ------------------------
def subplot(vel_grd, diff_grd, rmse, frame_title):
    # main velocity panel
    fig.basemap(    region=[fig_region], projection=size, panel=True)
    pygmt.makecpt(  cmap="vik", series=[vel_min, vel_max])
    fig.grdimage(   region=[fig_region], projection=size, grid=vel_grd, cmap=True, nan_transparent=True)
    fig.plot(       region=[fig_region], projection=size, x=ref_lon, y=ref_lat, style="s.15c", fill="black", pen="0.8p,black")
    
    fig.text(       region=[fig_region], projection=size, text=f"{rmse:.2f} mm/yr", position="BR", offset="-0.15c/0.15c", fill="white", font="8p")

    # inset difference panel
    fig.basemap(    region=[fig_region], projection=sub_size, frame="+t")
    fig.grdimage(   region=[fig_region], projection=sub_size, grid=diff_grd, cmap="plasma", nan_transparent=True)
    fig.coast(      region=[fig_region], projection=sub_size, shorelines=True, area_thresh=5000)
    with pygmt.config(FONT_ANNOT_PRIMARY="14p,black", FONT_ANNOT_SECONDARY="14p,black", FONT_LABEL="14p,black"):
        fig.colorbar(position="jBL+o0.2c/2c+w1.5c/0.2c", frame=["xa","y+lmm/yr"], projection=size)
        
    fig.coast(      region=[fig_region], projection=size, borders=1, shorelines=True, frame=[frame_title])
        

# ------------------------
# Plot Figure 
# ------------------------

size = "M4c"
sub_size = "M1.5c"
fig_region="-125.5/-119.224/36.172/42.408" 
style="s.15c"
vel_min, vel_max = -20, 20

fig = pygmt.Figure()
# Begin plot
pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain", FONT=9, FONT_TITLE=10, FONT_SUBTITLE = 10, MAP_TITLE_OFFSET= "-7p")

with fig.subplot(nrows=3, ncols=7, figsize=("29c", "16.5c"), autolabel=True,sharex="b", sharey="l",
                  frame=["WSrt", "xa", "ya"], margins=["0.2c", "0.2c"]):
    
    # ------------------------
    # Track 169 
    # ------------------------
    
    # Row for Track 169 - velocity
    fig.basemap(    region=[fig_region], projection=size, panel=True)
    pygmt.makecpt(  cmap="vik", series=[vel_min, vel_max ])
    fig.grdimage(   region=[fig_region], projection=size, grid=vel_169_grd , cmap=True, nan_transparent=True)
    fig.plot(       region=[fig_region], projection=size, y=ref_lat, x=ref_lon, style="s.15c", fill="black", pen="0.8p,black", )
    fig.coast(      region=[fig_region], projection=size, borders=1, shorelines=True,  frame = ["+tLinear Velocity"])
    fig.text(       region=[fig_region], projection=size, text=f'{results_169_dict['vel_169']['rmse']:.2f} mm/yr',  
             position="BR", offset="-0.15c/0.15c", fill="white", font="8p", )
        
    fig.text(       region=[fig_region], projection=size, text="Track 169", font="11p,Helvetica-Bold,black",  
             position="ML", justify="MC", offset="-1.5c/0.0c", no_clip=True, angle = 90)
    
    with pygmt.config(FONT_ANNOT_PRIMARY="14p,black", FONT_ANNOT_SECONDARY="14p,black", FONT_LABEL="14p,black"):
        fig.colorbar(position="jBL+o0.2c/0.2c+w2.5c/0.3c", frame=["xa+lVelocity", "y+lmm/yr"], projection = size)
    
    # Row for Track 169 - velocity_SET
    subplot(vel_SET_169_grd, SET_169_grd, results_169_dict['vel_SET_169']['rmse'], "+t- Solid Earth Tides")
    
    # Row for Track 169 - velocity_SET_ERA5
    subplot(vel_ERA5_169_grd, ERA5_169_grd, results_169_dict['vel_ERA5_169']['rmse'], "+t- Troposphere")
    
    # Row for Track 169 - velocity_SET_ERA5_demErr
    # main velocity panel
    fig.basemap(    region=[fig_region], projection=size, panel=True)
    pygmt.makecpt(  cmap="vik", series=[vel_min, vel_max])
    fig.grdimage(   region=[fig_region], projection=size, grid=vel_demErr_169_grd, cmap=True, nan_transparent=True)
    fig.plot(       region=[fig_region], projection=size, x=ref_lon, y=ref_lat, style=style, fill="black", pen="0.8p,black")
    
    fig.text(       region=[fig_region], projection=size, text=f"{results_169_dict['vel_demErr_169']['rmse']:.2f} mm/yr", position="BR", offset="-0.15c/0.15c", fill="white", font="8p")

    # inset difference panel
    fig.basemap(    region=[fig_region], projection=sub_size, frame="+t")
    pygmt.makecpt(  cmap="plasma", series=[-1, 1])
    fig.grdimage(   region=[fig_region], projection=sub_size, grid=demErr_169_grd, cmap=True, nan_transparent=True)
    fig.coast(      region=[fig_region], projection=sub_size, shorelines=True, area_thresh=5000)
    with pygmt.config(FONT_ANNOT_PRIMARY="14p,black", FONT_ANNOT_SECONDARY="14p,black", FONT_LABEL="14p,black"):
        fig.colorbar(position="jBL+o0.2c/2c+w1.5c/0.2c", frame=["xa","y+lmm/yr"], projection=size)
    fig.coast(      region=[fig_region], projection=size, borders=1, shorelines=True, frame=["+t- DEM Error"])
    
    
    
    # Row for Track 169 - velocity_SET_ERA5_demErr_ITRF
    subplot(vel_ITRF14_169_grd, ITRF14_169_grd, results_169_dict['vel_ITRF14_169']['rmse'], "+t- Bulk Plate Motion")
    
    # Row for Track 068 - residuals
    # main velocity panel
    fig.basemap(    region=[fig_region], projection=size, panel=True)
    pygmt.makecpt(cmap="roma", series=[-5+mean_res_169, 5+mean_res_169, 1])
    fig.plot(y=gps_169["Lat"], x=gps_169["Lon"], style="c.12c", fill=gps_169['residual_ITRF14_169'], cmap=True, pen="0.3p,black", region=[fig_region],projection= size)
    fig.plot(y=ref_lat, x=ref_lon, style="s.15c", fill="black", pen="0.8p,black", region=[fig_region],projection= size)
    fig.coast(      region=[fig_region], projection=size, borders=1, shorelines=True, frame=["+t Residuals"])
    fig.text(       region=[fig_region], projection=size, text=f"Mean {mean_res_169:.1f} mm/yr", position="BL", offset="0.15c/0.5c", fill="white", font="8p")
    fig.text(       region=[fig_region], projection=size, text=f"Range {ramp100_169:.2f} mm/yr/100 km", position="BL", offset="0.15c/0.15c", fill="white", font="8p")
    #fig.text(       region=[fig_region], projection=size, text=f"Ramp {res_r2_169*100:.2f}% of var.", position="BL", offset="0.15c/1.0c", fill="white", font="8p")

    with pygmt.config(FONT_ANNOT_PRIMARY="14p,black", FONT_ANNOT_SECONDARY="14p,black", FONT_LABEL="14p,black"):
        fig.colorbar(position="jBL+o0.2c/2c+w1.5c/0.2c", frame=["xa","y+lmm/yr"], projection=size)
    
    
    # Row for Track 169 - velocity_SET_ERA5_demErr_ITRF_ramp
    subplot(vel_deramp_169_grd, deramp_169_grd, results_169_dict['vel_deramp_169']['rmse'], "+t- Quad. Ramp")

    # ------------------------
    # Track 170 
    # ------------------------
    
    # Row for Track 170 - velocity
    fig.basemap(    region=[fig_region], projection=size, panel=True)
    pygmt.makecpt(  cmap="vik", series=[vel_min, vel_max ])
    fig.grdimage(   region=[fig_region], projection=size, grid=vel_170_grd , cmap=True, nan_transparent=True)
    fig.plot(       region=[fig_region], projection=size, y=ref_lat, x=ref_lon, style="s.15c", fill="black", pen="0.8p,black", )
    fig.coast(      region=[fig_region], projection=size, borders=1, shorelines=True,  frame = ["+t "])
    fig.text(       region=[fig_region], projection=size, text=f'{results_170_dict['vel_170']['rmse']:.2f} mm/yr',  
             position="BR", offset="-0.15c/0.15c", fill="white", font="8p", )
        
    fig.text(       region=[fig_region], projection=size, text="Track 170", font="11p,Helvetica-Bold,black",  
             position="ML", justify="MC", offset="-1.5c/0.0c", no_clip=True, angle = 90)
    
    fig.text(region=[fig_region], projection=size, x=ref_lon, y=ref_lat, text="%s" % ref_station,  font="10p,Helvetica,black", offset="0.5c/-0.3c+v", justify="LM", fill="white", transparency=50)
    fig.text(region=[fig_region], projection=size, x=ref_lon, y=ref_lat, text="%s" % ref_station,  font="10p,Helvetica,black", offset="0.5c/-0.3c+v", justify="LM")
    
    with pygmt.config(FONT_ANNOT_PRIMARY="14p,black", FONT_ANNOT_SECONDARY="14p,black", FONT_LABEL="14p,black"):
        fig.colorbar(position="jBL+o0.2c/0.2c+w2.5c/0.3c", frame=["xa+lVelocity", "y+lmm/yr"], projection = size)
    
    # Row for Track 170 - velocity_SET
    subplot(vel_SET_170_grd, SET_170_grd, results_170_dict['vel_SET_170']['rmse'], "+t ")
    
    # Row for Track 170 - velocity_SET_ERA5
    subplot(vel_ERA5_170_grd, ERA5_170_grd, results_170_dict['vel_ERA5_170']['rmse'], "+t ")
    
    # Row for Track 170 - velocity_SET_ERA5_demErr
    # main velocity panel
    fig.basemap(    region=[fig_region], projection=size, panel=True)
    pygmt.makecpt(  cmap="vik", series=[vel_min, vel_max])
    fig.grdimage(   region=[fig_region], projection=size, grid=vel_demErr_170_grd, cmap=True, nan_transparent=True)
    fig.plot(       region=[fig_region], projection=size, x=ref_lon, y=ref_lat, style=style, fill="black", pen="0.8p,black")
    
    fig.text(       region=[fig_region], projection=size, text=f"{results_170_dict['vel_demErr_170']['rmse']:.2f} mm/yr", position="BR", offset="-0.15c/0.15c", fill="white", font="8p")

    # inset difference panel
    fig.basemap(    region=[fig_region], projection=sub_size, frame="+t")
    pygmt.makecpt(  cmap="plasma", series=[-0.5, 0.5])
    fig.grdimage(   region=[fig_region], projection=sub_size, grid=demErr_170_grd, cmap=True, nan_transparent=True)
    fig.coast(      region=[fig_region], projection=sub_size, shorelines=True, area_thresh=5000)
    with pygmt.config(FONT_ANNOT_PRIMARY="14p,black", FONT_ANNOT_SECONDARY="14p,black", FONT_LABEL="14p,black"):
        fig.colorbar(position="jBL+o0.2c/2c+w1.5c/0.2c", frame=["xa","y+lmm/yr"], projection=size)
    fig.coast(      region=[fig_region], projection=size, borders=1, shorelines=True, frame=["+t "])
    
    
    # Row for Track 170 - velocity_SET_ERA5_demErr_ITRF
    subplot(vel_ITRF14_170_grd, ITRF14_170_grd, results_170_dict['vel_ITRF14_170']['rmse'], "+t ")
    
    # Row for Track 068 - residuals
    # main velocity panel
    fig.basemap(    region=[fig_region], projection=size, panel=True)
    pygmt.makecpt(cmap="roma", series=[-5+mean_res_170, 5+mean_res_170, 1])
    fig.plot(y=gps_170["Lat"], x=gps_170["Lon"], style="c.12c", fill=gps_170['residual_ITRF14_170'], cmap=True, pen="0.3p,black", region=[fig_region],projection= size)
    fig.plot(y=ref_lat, x=ref_lon, style="s.15c", fill="black", pen="0.8p,black", region=[fig_region],projection= size)
    fig.coast(      region=[fig_region], projection=size, borders=1, shorelines=True, frame=["+t "])
    fig.text(       region=[fig_region], projection=size, text=f"Mean {mean_res_170:.1f} mm/yr", position="BL", offset="0.15c/0.5c", fill="white", font="8p")
    fig.text(       region=[fig_region], projection=size, text=f"Range {ramp100_170:.2f} mm/yr/100 km", position="BL", offset="0.15c/0.15c", fill="white", font="8p")
    #fig.text(       region=[fig_region], projection=size, text=f"Ramp {res_r2_170*100:.2f}% of var.", position="BL", offset="0.15c/1.0c", fill="white", font="8p")

    with pygmt.config(FONT_ANNOT_PRIMARY="14p,black", FONT_ANNOT_SECONDARY="14p,black", FONT_LABEL="14p,black"):
        fig.colorbar(position="jBL+o0.2c/2c+w1.5c/0.2c", frame=["xa","y+lmm/yr"], projection=size)
        
    # Row for Track 170 - velocity_SET_ERA5_demErr_ITRF_ramp
    subplot(vel_deramp_170_grd, deramp_170_grd, results_170_dict['vel_deramp_170']['rmse'], "+t ")

    # ------------------------
    # Track 068 
    # ------------------------
    
    # Row for Track 068 - velocity
    fig.basemap(    region=[fig_region], projection=size, panel=True)
    pygmt.makecpt(  cmap="vik", series=[vel_min, vel_max ])
    fig.grdimage(   region=[fig_region], projection=size, grid=vel_068_grd , cmap=True, nan_transparent=True)
    fig.plot(       region=[fig_region], projection=size, y=ref_lat, x=ref_lon, style="s.15c", fill="black", pen="0.8p,black", )
    fig.coast(      region=[fig_region], projection=size, borders=1, shorelines=True,  frame=["+t "])
    fig.text(       region=[fig_region], projection=size, text=f'{results_068_dict['vel_068']['rmse']:.2f} mm/yr',  
             position="BR", offset="-0.15c/0.15c", fill="white", font="8p", )
        
    fig.text(       region=[fig_region], projection=size, text="Track 068", font="11p,Helvetica-Bold,black",  
             position="ML", justify="MC", offset="-1.5c/0.0c", no_clip=True, angle = 90)
    
    with pygmt.config(FONT_ANNOT_PRIMARY="14p,black", FONT_ANNOT_SECONDARY="14p,black", FONT_LABEL="14p,black"):
        fig.colorbar(position="jBL+o0.2c/0.2c+w2.5c/0.3c", frame=["xa+lVelocity", "y+lmm/yr"], projection = size)
    
    # Row for Track 068 - velocity_SET
    subplot(vel_SET_068_grd, SET_068_grd, results_068_dict['vel_SET_068']['rmse'], "+t ")
    
    # Row for Track 068 - velocity_SET_ERA5
    subplot(vel_ERA5_068_grd, ERA5_068_grd, results_068_dict['vel_ERA5_068']['rmse'], "+t ")
    
    # Row for Track 068 - velocity_SET_ERA5_demErr
    # main velocity panel
    fig.basemap(    region=[fig_region], projection=size, panel=True)
    pygmt.makecpt(  cmap="vik", series=[vel_min, vel_max])
    fig.grdimage(   region=[fig_region], projection=size, grid=vel_demErr_068_grd, cmap=True, nan_transparent=True)
    fig.plot(       region=[fig_region], projection=size, x=ref_lon, y=ref_lat, style=style, fill="black", pen="0.8p,black")
    
    fig.text(       region=[fig_region], projection=size, text=f"{results_068_dict['vel_demErr_068']['rmse']:.2f} mm/yr", position="BR", offset="-0.15c/0.15c", fill="white", font="8p")

    # inset difference panel
    fig.basemap(    region=[fig_region], projection=sub_size, frame="+t")
    pygmt.makecpt(  cmap="plasma", series=[-2, 2])
    fig.grdimage(   region=[fig_region], projection=sub_size, grid=demErr_068_grd, cmap=True, nan_transparent=True)
    fig.coast(      region=[fig_region], projection=sub_size, shorelines=True, area_thresh=5000)
    with pygmt.config(FONT_ANNOT_PRIMARY="14p,black", FONT_ANNOT_SECONDARY="14p,black", FONT_LABEL="14p,black"):
        fig.colorbar(position="jBL+o0.2c/2c+w1.5c/0.2c", frame=["xa","y+lmm/yr"], projection=size)
    fig.coast(      region=[fig_region], projection=size, borders=1, shorelines=True, frame=["+t "])
    
    
    # Row for Track 068 - velocity_SET_ERA5_demErr_ITRF
    subplot(vel_ITRF14_068_grd, ITRF14_068_grd, results_068_dict['vel_ITRF14_068']['rmse'], "+t ")
    
    # Row for Track 068 - residuals
    # main velocity panel
    fig.basemap(    region=[fig_region], projection=size, panel=True)
    pygmt.makecpt(cmap="roma", series=[-5+mean_res_068, 5+mean_res_068, 1])
    fig.plot(y=gps_068["Lat"], x=gps_068["Lon"], style="c.12c", fill=gps_068['residual_ITRF14_068'], cmap=True, pen="0.3p,black", region=[fig_region],projection= size)
    fig.plot(y=ref_lat, x=ref_lon, style="s.15c", fill="black", pen="0.8p,black", region=[fig_region],projection= size)
    fig.coast(      region=[fig_region], projection=size, borders=1, shorelines=True, frame=["+t "])
    fig.text(       region=[fig_region], projection=size, text=f"Mean {mean_res_068:.1f} mm/yr", position="BL", offset="0.15c/0.5c", fill="white", font="8p")
    fig.text(       region=[fig_region], projection=size, text=f"Range {ramp100_068:.2f} mm/yr/100 km", position="BL", offset="0.15c/0.15c", fill="white", font="8p")


    with pygmt.config(FONT_ANNOT_PRIMARY="14p,black", FONT_ANNOT_SECONDARY="14p,black", FONT_LABEL="14p,black"):
        fig.colorbar(position="jBL+o0.2c/2c+w1.5c/0.2c", frame=["xa","y+lmm/yr"], projection=size)
    

    # Row for Track 068 - velocity_SET_ERA5_demErr_ITRF_ramp
    subplot(vel_deramp_068_grd, deramp_068_grd, results_068_dict['vel_deramp_068']['rmse'], "+t ")

fig.savefig(common_paths['fig_dir']+f'Fig_3_{ref_station}_InSAR_vel_all_corrections_dist{distance_threshold}_latstep{lat_step}_lonstep{lon_step}QuadRammp_Residuals_GPS_platemotioncorrectionLOS.png', transparent=False, crop=True, anti_alias=True, show=False)
fig.savefig(common_paths['fig_dir']+f'Fig_3_{ref_station}_InSAR_vel_all_corrections_dist{distance_threshold}_latstep{lat_step}_lonstep{lon_step}QuadRammp_Residuals_GPS_platemotioncorrectionLOS.jpg', transparent=False, crop=True, anti_alias=True, show=False)
#fig.savefig(common_paths['fig_dir']+f'Fig_3_{ref_station}_InSAR_vel_all_corrections_dist{distance_threshold}_latstep{lat_step}_lonstep{lon_step}QuadRammp_Residuals.pdf', transparent=False, crop=True, anti_alias=True, show=False)
fig.show()  