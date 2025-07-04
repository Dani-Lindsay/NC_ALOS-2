#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 01:29:21 2025

@author: daniellelindsay
"""

from NC_ALOS2_filepaths import (paths_gps, paths_170, common_paths, decomp)
import insar_utils as utils
import numpy as np                # Matrix calculations
import pandas as pd               # Pandas for data
import pygmt
import h5py
from scipy import stats

dic_170 = {"ORBIT_DIRECTION" : "descending", 
           "PATH" : "170", 
           "Platform" : "ALOS-2", 
           "geo_file" : paths_170["geo"]["geo_geometryRadar"], 
           "vel_file" : paths_170["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"], 
           "ts_file" :  paths_170["geo"]["geo_timeseries"], 
           "vel_grd" :  paths_170["grd"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"], 
           }

insar_170 = utils.load_insar_vel_ts_as_dictionary(dic_170)

points = [ (-122.79015, 38.82104), (-122.721909, 38.771748)]

radius = 0.01
unit=1000

insar_170["ts"] = insar_170["ts"] * unit

ts_170 =  utils.extract_averaged_time_series(insar_170["ts"], insar_170["lons"], insar_170["lats"], points, radius)
inc_170 = utils.extract_incidence_at_points(insar_170["inc"], insar_170["lons"], insar_170["lats"], points, radius)



# # ts_170[i] is the 1D time‐series at point i; inc_170[i] is its incidence angle
# ts_170_up = ts_170 #[ts * np.cos(np.deg2rad(inc_170[i])) for i, ts in enumerate(ts_170)]
# ts_169_up = ts_169 #[ts * np.cos(np.deg2rad(inc_169[i])) for i, ts in enumerate(ts_169)]



# pt1_la, pt1_lo = 38.82104, -122.79015
# pt1_170_inc = 32.5
# pt1_169_inc = 42.5
# pt1_ts_170 = np.array([11.05, 12.31, 10.09, 10.37, 12.6, 11.16, 15.69, 11.17, 10.29, 9.4, 7.57, 9.67, 9.67, 9.39, 8.61, 6.82, 6.09, 7.19, 8.77, 6.66, 4.0, 6.39, 4.65, 6.0, 5.7, 6.58, 6.86, 7.23, 4.97, 5.46, 5.25, 6.03, 4.61, 5.81, 7.73, 5.18, 0.02, 2.31, 4.32, 3.71, 3.53, 3.22, 2.73, 1.28, -3.76, 2.31, 3.68, 3.32, 3.48, 3.25, 3.21, 3.6, 2.7, 3.78, 0.66, 2.01, -0.73, 1.04, 2.18, 3.09, 1.28, -0.14, 1.28, 2.68, 1.54, 0.19, -0.61, -0.26, 0.72, 0.47, -0.1, -2.86, -0.85, 0.84, 0.1, -1.11, -2.78, -0.72, -1.35, 0.09, -0.43, 0.0, -0.25, -1.04, 1.01, -0.13, -1.53, -5.04, -4.03, -1.71, -1.8])
#pt1_vel_170 = "-14.3 +/-  0.6 mm/yr"
# pt1_ts_169 = np.array([0.26, 0.69, -3.43, -1.06, 0.25, 0.01, -3.41, -0.21, -0.23, -2.76, -0.6, -2.06, -2.19, -1.9, 0.0, -0.66, -0.05, -2.92, -3.93, 1.02, -0.2, -0.89, -0.43, 0.3, -1.97, -2.14, 1.98, 0.11, 1.09, 0.36, -2.29, -0.98, -0.94, -2.41, -0.7, -2.05, -1.33, -2.02, -2.5, -3.2, -2.67, -2.08, -1.64, -4.1, -2.76, -2.0, -3.48, -4.96, -4.22, -5.1, -5.14, -2.53, -2.54, -4.55, -1.38, -4.4, -5.06, -2.83])
#pt1_vel_169 = "-5.5 +/-  1.2 mm/year"


# pt2_la, pt2_lo = 38.76891, -122.70944
# pt2_170_inc = 32.0
# pt2_169_inc = 42.0
# pt2_ts_170 = np.array([9.79, 11.12, 8.84, 8.13, 11.6, 10.34, 16.88, 10.98, 9.98, 11.02, 8.17, 10.38, 11.35, 10.45, 8.2, 6.98, 6.35, 7.21, 8.77, 6.14, 4.46, 6.18, 6.08, 7.29, 6.93, 5.76, 6.09, 6.75, 4.69, 6.3, 5.05, 4.77, 3.75, 5.38, 7.07, 4.89, 2.24, 2.06, 5.64, 3.26, 2.77, 2.52, 3.67, 1.17, -0.7, 1.86, 2.51, 2.49, 3.19, 2.03, 2.12, 2.57, 2.84, 2.8, -0.83, 2.46, 1.54, 1.49, 3.29, 2.82, 1.01, 0.55, 2.65, 2.17, 1.55, 0.48, 1.39, 1.17, 2.15, -0.02, -0.16, 2.14, -0.07, 0.55, 0.22, 0.48, -1.95, 1.16, 0.48, 0.75, 0.08, 0.0, 0.59, -0.24, 1.95, 0.8, 1.11, -1.49, -1.39, 0.28, -0.96])
#pt2_vel_170 = "-12.8 +/-  0.5 mm/yr"
# pt2_ts_169 = np.array([1.12, 1.74, -3.19, 0.1, -0.31, 0.06, -1.06, 1.05, -0.46, -1.93, -0.85, -0.14, -2.51, -2.06, 0.0, -1.33, 0.06, -1.31, -2.72, -0.8, 0.08, -1.0, -0.53, -0.14, -2.35, -0.38, 0.56, -1.65, 0.93, -0.18, -1.21, -1.56, -0.26, -3.29, 0.21, -2.83, -1.97, -2.34, -2.07, -2.59, -2.58, -2.44, -1.73, -3.24, -4.27, -1.72, -2.34, -4.5, -4.03, -3.54, -4.03, -1.65, -1.92, -2.97, -1.91, -3.86, -4.61, -3.08])
#pt2_vel_169 = "-6.1 +/-  1.0 mm/year"

# points = [(-122.79015, 38.82104), # north
#           (-122.70944, 38.76891)] # south 

# # Define ts_170 as a list of two 1D numpy arrays (one per point).
# ts_170 = [
#     np.array([11.05, 12.31, 10.09, 10.37, 12.6, 11.16, 15.69, 11.17, 10.29, 9.4, 7.57, 9.67, 9.67, 9.39, 8.61, 6.82, 6.09, 7.19, 8.77, 6.66, 4.0, 6.39, 4.65, 6.0, 5.7, 6.58, 6.86, 7.23, 4.97, 5.46, 5.25, 6.03, 4.61, 5.81, 7.73, 5.18, 0.02, 2.31, 4.32, 3.71, 3.53, 3.22, 2.73, 1.28, -3.76, 2.31, 3.68, 3.32, 3.48, 3.25, 3.21, 3.6, 2.7, 3.78, 0.66, 2.01, -0.73, 1.04, 2.18, 3.09, 1.28, -0.14, 1.28, 2.68, 1.54, 0.19, -0.61, -0.26, 0.72, 0.47, -0.1, -2.86, -0.85, 0.84, 0.1, -1.11, -2.78, -0.72, -1.35, 0.09, -0.43, 0.0, -0.25, -1.04, 1.01, -0.13, -1.53, -5.04, -4.03, -1.71, -1.8]),
#     np.array([9.79, 11.12, 8.84, 8.13, 11.6, 10.34, 16.88, 10.98, 9.98, 11.02, 8.17, 10.38, 11.35, 10.45, 8.2, 6.98, 6.35, 7.21, 8.77, 6.14, 4.46, 6.18, 6.08, 7.29, 6.93, 5.76, 6.09, 6.75, 4.69, 6.3, 5.05, 4.77, 3.75, 5.38, 7.07, 4.89, 2.24, 2.06, 5.64, 3.26, 2.77, 2.52, 3.67, 1.17, -0.7, 1.86, 2.51, 2.49, 3.19, 2.03, 2.12, 2.57, 2.84, 2.8, -0.83, 2.46, 1.54, 1.49, 3.29, 2.82, 1.01, 0.55, 2.65, 2.17, 1.55, 0.48, 1.39, 1.17, 2.15, -0.02, -0.16, 2.14, -0.07, 0.55, 0.22, 0.48, -1.95, 1.16, 0.48, 0.75, 0.08, 0.0, 0.59, -0.24, 1.95, 0.8, 1.11, -1.49, -1.39, 0.28, -0.96])
# ]

# # Define ts_169 similarly, using np.array instead of np.arange
# ts_169 = [
#     np.array([0.26, 0.69, -3.43, -1.06, 0.25, 0.01, -3.41, -0.21, -0.23, -2.76, -0.6, -2.06, -2.19, -1.9, 0.0, -0.66, -0.05, -2.92, -3.93, 1.02, -0.2, -0.89, -0.43, 0.3, -1.97, -2.14, 1.98, 0.11, 1.09, 0.36, -2.29, -0.98, -0.94, -2.41, -0.7, -2.05, -1.33, -2.02, -2.5, -3.2, -2.67, -2.08, -1.64, -4.1, -2.76, -2.0, -3.48, -4.96, -4.22, -5.1, -5.14, -2.53, -2.54, -4.55, -1.38, -4.4, -5.06, -2.83]),
#     np.array([1.12, 1.74, -3.19, 0.1, -0.31, 0.06, -1.06, 1.05, -0.46, -1.93, -0.85, -0.14, -2.51, -2.06, 0.0, -1.33, 0.06, -1.31, -2.72, -0.8, 0.08, -1.0, -0.53, -0.14, -2.35, -0.38, 0.56, -1.65, 0.93, -0.18, -1.21, -1.56, -0.26, -3.29, 0.21, -2.83, -1.97, -2.34, -2.07, -2.59, -2.58, -2.44, -1.73, -3.24, -4.27, -1.72, -2.34, -4.5, -4.03, -3.54, -4.03, -1.65, -1.92, -2.97, -1.91, -3.86, -4.61, -3.08])
# ]

# ts_170[i] is the 1D time‐series at point i; inc_170[i] is its incidence angle
# ts_170_up = [ts * np.cos(np.deg2rad(inc_170[i])) for i, ts in enumerate(ts_170)]
# ts_169_up = [ts * np.cos(np.deg2rad(inc_169[i])) for i, ts in enumerate(ts_169)]

# ts_170_up = ts_170 #[ts * np.cos(np.deg2rad(inc_170[i])) for i, ts in enumerate(ts_170)]
# ts_169_up = ts_169 #[ts * np.cos(np.deg2rad(inc_169[i])) for i, ts in enumerate(ts_169)]

vel_t1 = 2015.0
vel_t2 = 2019.2

vel_t3 = 2021.5
vel_t4 = 2024.215

vel_pt1_15_20, err_pt1_15_20 = utils.compute_velocity(insar_170['ts_dates'], ts_170[0], start=vel_t1, stop=vel_t2)
vel_pt1_21_24, err_pt1_21_24 = utils.compute_velocity(insar_170['ts_dates'], ts_170[0], start=vel_t3, stop=vel_t4)
vel_pt2_15_20, err_pt2_15_20 = utils.compute_velocity(insar_170['ts_dates'], ts_170[1], start=vel_t1, stop=vel_t2)
vel_pt2_21_24, err_pt2_21_24 = utils.compute_velocity(insar_170['ts_dates'], ts_170[1], start=vel_t3, stop=vel_t4)

###########################
# Load InSAR
###########################

gey_center_la_east, gey_center_lo_east = 38.831758,  -122.785778
gey_center_la_up, gey_center_lo_up =  38.831758,  -122.785778

gey_center_la_east, gey_center_lo_east = 38.764905, -122.722718
gey_center_la_up, gey_center_lo_up,  =  38.764905, -122.722718


ref_station, ref_lat, ref_lon = "CASR", 38.441,  -122.747

# Define region of interest 
gey_min_lon=-123.2
gey_max_lon=-122.5
gey_min_lat=38.575
gey_max_lat=39.0

gey_region = "%s/%s/%s/%s" % (gey_min_lon, gey_max_lon, gey_min_lat, gey_max_lat)

east_df = utils.load_h5_data(paths_170["geo"]["geo_geometryRadar"], decomp["CASR"]["gps_insar_east"], "velocity")
up_df = utils.load_h5_data(paths_170["geo"]["geo_geometryRadar"], decomp["CASR"]["gps_insar_up"], "velocity")

east = east_df.drop(columns=['Az', 'Inc'])
up = up_df.drop(columns=['Az', 'Inc'])

east['Vel'] = east['Vel']
up['Vel'] = up['Vel']

east = east[
    (east['Lon'] >= gey_min_lon) & 
    (east['Lon'] <= gey_max_lon) & 
    (east['Lat'] >= gey_min_lat) & 
    (east['Lat'] <= gey_max_lat)
]

up = up[
    (up['Lon'] >= gey_min_lon) & 
    (up['Lon'] <= gey_max_lon) & 
    (up['Lat'] >= gey_min_lat) & 
    (up['Lat'] <= gey_max_lat)
]

###########################
# Load GNSS
###########################
gps_df = utils.load_UNR_gps(paths_gps["170_enu"], ref_station)

###########################
# Extract Profiles
###########################

## define profile variables
gey_azi = 45

# profile distances and width in km. 
p_dist = 15
width = 1

# Download DEM
grid = pygmt.datasets.load_earth_relief(region=gey_region, resolution="01s")
dgrid = pygmt.grdgradient(grid=grid, radiance=[315, 30], region=gey_region)

### Define x distance bin spacing to closer near to zero profiles
dist_bins =np.arange(-p_dist, p_dist, 0.5)

### Get start and end points for plotting
A_start_lon, A_start_lat, A_end_lon, A_end_lat = utils.get_start_end_points(gey_center_lo_up, gey_center_la_up, gey_azi, p_dist)
up_points, up_mean, up_offset = utils.extract_profiles(up, gey_center_lo_up, gey_center_la_up, gey_azi, p_dist, width, dist_bins)
dem_track_df = pygmt.project(center=f"{A_start_lon}/{A_start_lat}", endpoint=f"{A_end_lon}/{A_end_lat}", generate="0.002")
dem_track_df = pygmt.grdtrack(grid=grid, points=dem_track_df, newcolname="elevation")
dem_xyz = dem_track_df[["r", "s", "elevation"]]
gey_dem_a_points, dem_a_mean, dem_a_offset = utils.extract_profiles(dem_xyz, gey_center_lo_up, gey_center_la_up, gey_azi, p_dist, width, dist_bins)

B_start_lon, B_start_lat, B_end_lon, B_end_lat = utils.get_start_end_points(gey_center_lo_east, gey_center_la_east, gey_azi, p_dist)
east_points, east_mean, east_offset = utils.extract_profiles(east, gey_center_lo_east, gey_center_la_east, gey_azi, p_dist, width, dist_bins)
dem_track_df = pygmt.project(center=f"{B_start_lon}/{B_start_lat}", endpoint=f"{B_end_lon}/{B_end_lat}", generate="0.002")
dem_track_df = pygmt.grdtrack(grid=grid, points=dem_track_df, newcolname="elevation")
dem_xyz = dem_track_df[["r", "s", "elevation"]]
get_dem_b_points, dem_b_mean, dem_b_offset = utils.extract_profiles(dem_xyz, gey_center_lo_east, gey_center_la_east, gey_azi, p_dist, width, dist_bins)

###########################
# Find maximum subsidence 
###########################

# Compute the mean "stable" value in the specified distance range
mean_up = up_mean.loc[(up_mean['distance'] >= -15) & (up_mean['distance'] <= -5), 'z'].mean()
mean_east = east_mean.loc[(east_mean['distance'] >= -15) & (east_mean['distance'] <= -5), 'z'].mean()
#mean_east_e = east_mean.loc[(east_mean['distance'] >= -30) & (east_mean['distance'] <= -15), 'z'].mean()


# Find the minimum (maximum subsidence) values for each profile
min_up = np.nanmin(up_mean["z"])

min_east = east_mean.loc[(east_mean['distance'] >= -5) & (east_mean['distance'] <= 5), 'z'].min()
max_east = east_mean.loc[(east_mean['distance'] >= -5) & (east_mean['distance'] <= 5), 'z'].max()

#min_east = np.nanmin(east_mean["z"])  # corrected from up_a_mean to up_b_mean
#max_east = np.nanmax(east_mean["z"]) 

# Calculate subsidence relative to the stable "zero" level
subsidence = min_up - mean_up
east_peak2trough = max_east - min_east

print("Subsidence:", subsidence)
print("Contraction:", east_peak2trough)



###########################
# Plot 
###########################

style="c.03c"
res_style="c.1c"
size = "M5.01c"

vmin = 20

sub_map_size = "M5.6c"
sub_proj = "X7/4.35c"
#NC_grid = pygmt.datasets.load_earth_relief(region=NC_fig_region, resolution="15s")
#NC_dgrid = pygmt.grdgradient(grid=NC_grid, radiance=[315, 30], region=NC_fig_region)

fig = pygmt.Figure()
pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain", FONT=10, FONT_TITLE=11, FONT_SUBTITLE = 11, MAP_TITLE_OFFSET= "-7p")

with fig.subplot(nrows=2, ncols=1, figsize=("6.1c", "9.35c"), autolabel="a)",sharex="b", sharey="l",
                  frame=["WSrt", "xa", "ya"], margins=["0.2c", "0.2c"]):
    
    # ## ******** gey Map ********* 
    fig.basemap(region=gey_region, projection=sub_map_size, frame=["Wbtr", "xa", "ya"], panel=True,)
    pygmt.makecpt(cmap="vik", series=[-12, 12])
    fig.grdimage(grid=decomp["grd"]["gps_insar_up"], cmap=True, nan_transparent=True, region=gey_region, projection= sub_map_size)
    #fig.grdimage(grid=grid, projection=sub_map_size, frame=["lbrt", "xa", "ya"], cmap="grey", shading=dgrid, region=gey_region)
    for fault_file in common_paths["fault_files"]:
        fig.plot(data=fault_file, pen="0.5p,black", transparency=50, region=gey_region, projection= sub_map_size)
    
    fig.plot(x=gps_df["Lon"], y=gps_df["Lat"],  style="c.2c", fill=gps_df["Vu"], pen="0.8p,black", cmap=True, region=gey_region, projection= sub_map_size)
    fig.plot(x=points[0][0], y=points[0][1],  style="+.2c", fill="black", pen="0.8p,black", region=gey_region, projection= sub_map_size)
    fig.plot(x=points[1][0], y=points[1][1],  style="+.2c", fill="black", pen="0.8p,black", region=gey_region, projection= sub_map_size)
    fig.plot(x=[A_start_lon, A_end_lon], y=[A_start_lat, A_end_lat], pen="1.2p,black", region=gey_region, projection=sub_map_size, transparency=40)
    fig.text(text="X", x=A_start_lon, y=A_start_lat, justify="RM", offset="-0.1c/-0.1c", font="10p,Helvetica,black", region=gey_region, projection=sub_map_size, fill="white", transparency=30  )
    fig.text(text="X'", x=A_end_lon, y=A_end_lat, justify="LM", offset="0.1c/0.1c", font="10p,Helvetica,black", region=gey_region, projection=sub_map_size, fill="white", transparency=30  )
    fig.text(text="X", x=A_start_lon, y=A_start_lat, justify="RM", offset="-0.1c/-0.1c", font="10p,Helvetica,black", region=gey_region, projection=sub_map_size )
    fig.text(text="X'", x=A_end_lon, y=A_end_lat, justify="LM", offset="0.1c/0.1c", font="10p,Helvetica,black", region=gey_region, projection=sub_map_size )
    fig.text(text="Up Velocity", position="TC", offset="0c/-0.2c", region=gey_region, projection= sub_map_size, fill="white", transparency=30)
    fig.text(text="Up Velocity", position="TC", offset="0c/-0.2c", region=gey_region, projection= sub_map_size)

    fig.text(text="MA", x=-122.916817, y=38.776721, justify="CM", font="9p,black" , fill="white", angle=320, transparency=30)
    fig.text(text="CO", x=-122.763552, y=38.866036, justify="CM", font="9p,black" , fill="white", angle=320, transparency=30)
    
    fig.text(text="MA", x=-122.916817, y=38.776721, justify="CM", font="9p,black" , angle=320)
    fig.text(text="CO", x=-122.763552, y=38.866036, justify="CM", font="9p,black" , angle=320)
    
    with pygmt.config(
            FONT_ANNOT_PRIMARY="18p,black", 
            FONT_ANNOT_SECONDARY="18p,black",
            FONT_LABEL="18p,black",
            ):
        #pygmt.makecpt(cmap="vik", series=[-vmin, vmin])
        fig.colorbar(position="jBL+o0.2c/0.2c+w2c/0.3c", frame=["xaf", "y+lmm/yr"],)
    
    fig.basemap(region=gey_region, projection=sub_map_size, frame=["Wbtr", "xa", "ya"], map_scale="jBR+w10k+o0.2c/0.5c")
    
    # ## ******** gey DEM Map ********* 
    fig.basemap(region=gey_region, projection=sub_map_size, frame=["WStr", "xa", "ya"], panel=True,)
    #fig.grdimage(grid=grid, projection=sub_map_size, frame=["lbrt", "xa", "ya"], cmap="grey", shading=dgrid, region=gey_region)
    pygmt.makecpt(cmap="vik", series=[-6, 6])
    fig.grdimage(grid=decomp["grd"]["gps_insar_east"], cmap=True, nan_transparent=True, region=gey_region, projection= sub_map_size)
    #fig.grdimage(grid="/Users/daniellelindsay/NC_Manuscript_Data_local/P784_068_170_Hz_Up/Up_068_170_P784.grd", cmap=True, nan_transparent=True, region=gey_region, projection= sub_map_size)
    for fault_file in common_paths["fault_files"]:
        fig.plot(data=fault_file, pen="0.5p,black", transparency=50, region=gey_region, projection= sub_map_size)
    fig.plot(x=gps_df["Lon"], y=gps_df["Lat"],  style="c.2c", fill=gps_df["Ve"], pen="0.8p,black", cmap=True, region=gey_region, projection= sub_map_size)
    fig.plot(x=points[0][0], y=points[0][1],  style="+.2c", fill="black", pen="0.8p,black", region=gey_region, projection= sub_map_size)
    fig.plot(x=points[1][0], y=points[1][1],  style="+.2c", fill="black", pen="0.8p,black", region=gey_region, projection= sub_map_size)
    fig.plot(x=[B_start_lon, B_end_lon], y=[B_start_lat, B_end_lat], pen="1.2p,black", region=gey_region, projection=sub_map_size, transparency=40)
    fig.text(text="Y", x=B_start_lon, y=B_start_lat, justify="RM", offset="-0.1c/-0.1c", font="10p,Helvetica,black", region=gey_region, projection=sub_map_size, fill="white", transparency=30 )
    fig.text(text="Y'", x=B_end_lon, y=B_end_lat, justify="LM", offset="0.1c/0.1c", font="10p,Helvetica,black", region=gey_region, projection=sub_map_size, fill="white", transparency=30 )
    fig.text(text="Y", x=B_start_lon, y=B_start_lat, justify="RM", offset="-0.1c/-0.1c", font="10p,Helvetica,black", region=gey_region, projection=sub_map_size )
    fig.text(text="Y'", x=B_end_lon, y=B_end_lat, justify="LM", offset="0.1c/0.1c", font="10p,Helvetica,black", region=gey_region, projection=sub_map_size )
    #fig.text(text="X", x=B_start_lon, y=B_start_lat, justify="RM", offset="-0.1c/-0.1c", font="10p,Helvetica,black", region=gey_region, projection=sub_map_size )
    fig.text(text="Pt 1", x=points[0][0], y=points[0][1], justify="LB", offset="0.1c/0.1c", font="10p,Helvetica,black", region=gey_region, projection=sub_map_size, fill="white", transparency=30 )
    fig.text(text="Pt 2", x=points[1][0], y=points[1][1], justify="LT", offset="0.1c/-0.1c", font="10p,Helvetica,black", region=gey_region, projection=sub_map_size, fill="white", transparency=30 )
    fig.text(text="Pt 1", x=points[0][0], y=points[0][1], justify="LB", offset="0.1c/0.1c", font="10p,Helvetica,black", region=gey_region, projection=sub_map_size )
    fig.text(text="Pt 2", x=points[1][0], y=points[1][1], justify="LT", offset="0.1c/-0.1c", font="10p,Helvetica,black", region=gey_region, projection=sub_map_size )
    fig.text(text="East Velocity", position="TC", offset="0c/-0.2c", region=gey_region, projection= sub_map_size, fill="white", transparency=30)
    fig.text(text="East Velocity", position="TC", offset="0c/-0.2c", region=gey_region, projection= sub_map_size)
    #fig.plot(x=[gey_center_lo], y=[gey_center_la], size=[20], style="E-", pen="1.5p,white", region=gey_region, projection= sub_map_size)
    
    with pygmt.config(
            FONT_ANNOT_PRIMARY="18p,black", 
            FONT_ANNOT_SECONDARY="18p,black",
            FONT_LABEL="18p,black",
            ):
        #pygmt.makecpt(cmap="vik", series=[-vmin, vmin])
        fig.colorbar(position="jBL+o0.2c/0.2c+w2c/0.3c", frame=["xaf", "y+lmm/yr"],)
        
    fig.basemap(region=gey_region, projection=sub_map_size, frame=["WStr", "xa", "ya"], map_scale="jBR+w10k+o0.2c/0.5c")

fig.shift_origin(xshift="wc")
with fig.subplot(nrows=2, ncols=1, figsize=("7c", "9.35c"), autolabel="c)",sharex="b", sharey="l",
                  frame=["WSrt", "xa", "ya"], margins=["0.2c", "0.2c"]):
    
    # ## ******** gey Profile ********* 
    up_region =[-p_dist-0.5, p_dist+0.5, -23, 23]
    topo_region=[-p_dist-0.5, p_dist+0.5, -3000, 2000]
    fig.basemap(region=up_region, projection=sub_proj, panel=True, frame=["lsEt" , "xaf+l Profile Distance (km)", "yaf+lVelocity Up (mm/yr)"])    
    pygmt.makecpt(cmap="vik", series=[-20, 20])
    fig.plot(x=up_points.p, y=up_points.z, style="c0.1c", fill=up_points.z, cmap=True, transparency=70, projection=sub_proj)
    fig.plot(x=up_mean.p,   y=up_mean.z, pen="1.2p,black", projection=sub_proj) #style="c0.1c", fill="black")
    fig.plot(x=dem_a_mean.p,   y=dem_a_mean.z, pen="1.2p,black", projection=sub_proj, region=topo_region) #style="c0.1c", fill="black")
    fig.text(text="X → X' Up Velocity", position="TC", offset="0c/-0.2c", region=up_region, projection=sub_proj)
    fig.plot(x=-8, y=subsidence, style="v0.3c+b+e+h0.15", direction=([90], [np.abs(subsidence)/10]), pen="0.8p,black", fill="black", region=up_region, projection=sub_proj)
    fig.plot(x=[-12, 0], y=[mean_up, mean_up],  pen="0.8p,black,--", region=up_region, projection=sub_proj)
    fig.plot(x=[-12, 0], y=[subsidence, subsidence],  pen="0.8p,black,--", region=up_region, projection=sub_proj)
    fig.text(text="%s mm/yr" % np.round(subsidence,1), x=-8, y=subsidence-2, justify="TC", region=up_region, projection=sub_proj)
    
    # ## ******** gey Profile ********* 
    up_region =[-p_dist-0.5, p_dist+0.5, -4, 7]
    topo_region=[-p_dist-0.5, p_dist+0.5, -3000, 2000]
    fig.basemap(region=up_region, projection=sub_proj, panel=True, frame=["lSEt" , "xaf+l Profile Distance (km)", "yaf+lVelocity East (mm/yr)"])    
    pygmt.makecpt(cmap="vik", series=[-6, 6])
    fig.plot(x=east_points.p, y=east_points.z-mean_east, style="c0.1c", fill=east_points.z-mean_east, cmap=True, projection=sub_proj)
    fig.plot(x=east_mean.p,   y=east_mean.z-mean_east, pen="1.2p,black", projection=sub_proj) #style="c0.1c", fill="black")
    fig.plot(x=dem_b_mean.p,  y=dem_b_mean.z, pen="1.2p,black", projection=sub_proj, region=topo_region) #style="c0.1c", fill="black")
    fig.text(text="Y → Y' East Velocity", position="TC", offset="0c/-0.2c", region=up_region, projection=sub_proj)
    fig.plot(x=[-5, 5], y=[max_east-mean_east, max_east-mean_east],  pen="0.8p,black,--", region=up_region, projection=sub_proj)
    fig.plot(x=[-5, 5], y=[min_east-mean_east, min_east-mean_east],  pen="0.8p,black,--", region=up_region, projection=sub_proj)
    fig.plot(x=0, y=min_east-mean_east, style="v0.3c+b+e+h0.15", direction=([90], [east_peak2trough/2.5]), pen="0.8p,black", fill="black", region=up_region, projection=sub_proj)
    fig.text(text="%s mm/yr" % np.round(east_peak2trough,1), x=0, y=min_east-mean_east-0.5, justify="TC", region=up_region, projection=sub_proj)
    
fig.shift_origin(xshift="8.5c")
with fig.subplot(nrows=2, ncols=1, figsize=("7c", "9.35c"), autolabel="e)",sharex="b", sharey="l",
                  frame=["WSrt", "xa", "ya"], margins=["0.2c", "0.2c"]):
    
    sub_proj = "X7/4.35c"
       
    # ## ******** gey Profile ********* 
    ts_region =[insar_170['ts_dates'][0]-0.6, insar_170['ts_dates'][-1]+0.6, -125, 125]
    y_text = -100
    
    fig.basemap(region=ts_region, projection=sub_proj, panel=True, frame=["lsEt" , "xaf", "yaf+lDisplacement LOS (mm)"]) 
    fig.plot(x=insar_170['ts_dates'], y=(ts_170[0]-np.nanmean(ts_170[0])), style="c0.1c", fill="dodgerblue4", region=ts_region, projection=sub_proj) # , label = "Track 170")
    fig.text(text="Point 1", position="TC", offset="0c/-0.2c", region=up_region, projection=sub_proj)
    
    fig.plot(x=[vel_t1, vel_t2], y=[y_text, y_text],  pen="2p,darkorange", transparency=50, region=ts_region, projection=sub_proj)
    fig.plot(x=[vel_t3, vel_t4], y=[y_text, y_text],  pen="2p,darkorange", transparency=50, region=ts_region, projection=sub_proj)
    fig.text(text=f"{vel_pt1_15_20:.1f}±{err_pt1_15_20:.1f} mm/yr", x=(vel_t1+vel_t2)/2, y=y_text, offset="0c/0.3c", region=ts_region, projection=sub_proj)
    fig.text(text=f"{vel_pt1_21_24:.1f}±{err_pt1_21_24:.1f} mm/yr", x=(vel_t3+vel_t4)/2, y=y_text, offset="0c/0.3c", region=ts_region, projection=sub_proj)
    
    # ## ******** gey Profile ********* 
    
    y_text = -100
    fig.basemap(region=ts_region, projection=sub_proj, panel=True, frame=["lSEt" , "xaf", "yaf+lDisplacement LOS (mm)"])    
    fig.plot(x=insar_170['ts_dates'],  y=(ts_170[1]-np.nanmean(ts_170[1])), style="c0.1c", fill="dodgerblue4", region=ts_region, projection=sub_proj)
    fig.text(text="Point 2", position="TC", offset="0c/-0.2c", region=up_region, projection=sub_proj)
    
    fig.plot(x=[vel_t1, vel_t2], y=[y_text, y_text],  pen="2p,darkorange", transparency=50, region=ts_region, projection=sub_proj)
    fig.plot(x=[vel_t3, vel_t4], y=[y_text, y_text],  pen="2p,darkorange", transparency=50, region=ts_region, projection=sub_proj)
    fig.text(text=f"{vel_pt2_15_20:.1f}±{err_pt2_15_20:.1f} mm/yr", x=(vel_t1+vel_t2)/2, y=y_text, offset="0c/0.3c", region=ts_region, projection=sub_proj)
    fig.text(text=f"{vel_pt2_21_24:.1f}±{err_pt2_21_24:.1f} mm/yr", x=(vel_t3+vel_t4)/2, y=y_text, offset="0c/0.3c", region=ts_region, projection=sub_proj)
       
fig.savefig(common_paths["fig_dir"]+"Fig_10_Geysers_Vertial_example_southernprofiles.png", transparent=False, crop=True, anti_alias=True, show=False)
fig.savefig(common_paths["fig_dir"]+"Fig_10_Geysers_Vertial_example_southernprofiles.pdf", transparent=False, crop=True, anti_alias=True, show=False)
fig.show()  


# fig = pygmt.Figure()
# scat_region = [np.nanmin(dem_a_mean.p), np.nanmax(dem_a_mean.p), np.nanmin(dem_b_mean.z)-200, np.nanmax(dem_b_mean.z)]
# fig.basemap(region=scat_region, projection="X10/5c", frame=["lSEt" , "xaf+l Velocity Up (mm/yr)", "ya100f100g+lElevation (m)"])
# fig.plot(x=dem_a_mean.p,   y=(dem_a_mean.z), pen="1.2p,black", projection=sub_proj) #style="c0.1c", fill="black")
# fig.plot(x=dem_b_mean.p,   y=(dem_b_mean.z), pen="1.2p,black", projection=sub_proj) #style="c0.1c", fill="black")
# fig.show()  