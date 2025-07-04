#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 01:29:21 2025

@author: daniellelindsay
"""

from NC_ALOS2_filepaths import (common_paths, paths_gps, decomp)
import insar_utils as utils
import numpy as np                # Matrix calculations
import pandas as pd               # Pandas for data
import pygmt

dist = common_paths["dist"]

lat_step = common_paths["lat_step"]
lon_step = common_paths["lon_step"]

shasta_lat, shasta_lon =  41.410519, -122.194231

radius = 20 

###########################
# Load GNSS
###########################
# MLV GPS
columns = ['Lon', 'Lat', 'Ve', 'Vn', 'Vu', 'Std_e', 'Std_n', 'Std_u', 'StaID']
gps_df = pd.read_csv(paths_gps['visr']['gps_enu'] ,delim_whitespace=True, comment='#', names=columns)
Vu_ref, Ve_ref, Vn_ref = gps_df.loc[gps_df['StaID'] == 'P784', ['Vu', 'Ve', 'Vn']].values[0]
gps_df[['Vu', 'Ve', 'Vn']] = gps_df[['Vu', 'Ve', 'Vn']] - [Vu_ref, Ve_ref, Vn_ref]
mlv_gps_df = gps_df

###########################
# Load InSAR
###########################

mlv_center_la, mlv_center_lo,  =  41.578394, -121.591573
ref_station, ref_lat, ref_lon = "P784", 41.831, -122.420

mlv_min_lon = -122.5
mlv_max_lon = -121.2
mlv_min_lat = 41.14
mlv_max_lat = 41.90

mlv_region = "%s/%s/%s/%s" % (mlv_min_lon, mlv_max_lon, mlv_min_lat, mlv_max_lat)

data = utils.load_h5_data(decomp["P784"]["geo"], decomp["P784"]["insar_only_up"], 'velocity')
data['Vel'] = data['Vel'] 
data = data[['Lon', 'Lat', 'Vel']]
mlv_data = data[
    (data['Lon'] >= mlv_min_lon) & 
    (data['Lon'] <= mlv_max_lon) & 
    (data['Lat'] >= mlv_min_lat) & 
    (data['Lat'] <= mlv_max_lat)
]

height = utils.load_h5_generalData_df(decomp["P784"]["geo"], decomp["P784"]["geo"], 'height')
height = height[
    (height['Lon'] >= mlv_min_lon) & 
    (height['Lon'] <= mlv_max_lon) & 
    (height['Lat'] >= mlv_min_lat) & 
    (height['Lat'] <= mlv_max_lat)
]

###########################
# Load InSAR
###########################

# 1) merge on Lon & Lat
df = pd.merge( mlv_data[['Lon','Lat','Vel']], height[['Lon','Lat','height']], on=['Lon','Lat'])


# 2) compute geodesic distance (km) for each row
df['dist2ref'] = df.apply(lambda row: utils.calculate_distance(row['Lat'], row['Lon'], mlv_center_la, mlv_center_lo), axis=1)

# 3) mask out everything within 10 km
df_filt = df[df['dist2ref'] > radius].copy()
df_inside = df[df['dist2ref'] <= radius].copy()

# 4) Pearson‐R between height and Vel (outside radius)
r = df_filt['Vel'].corr(df_filt['height'])
r2 = r**2
r2_var = r2 * 100
print(f"Pearson r (Vel vs height) beyond {radius} km: {r:.3f}")
print(f"R^2 (Vel vs height) beyond {radius} km: {r2:.3f}, explains {r2_var:.3f}% of variance")

# Pearson‐R for points inside the radius
r_inside = df_inside['Vel'].corr(df_inside['height'])
r2_inside = r_inside**2
r2_var_inside = r2_inside * 100
print(f"Pearson r (Vel vs height) inside {radius} km: {r_inside:.3f}")
print(f"R^2 (Vel vs height) inside {radius} km: {r2_inside:.3f}, explains {r2_var_inside:.3f}% of variance")

df.sort_values(by='dist2ref', ascending=False, inplace=True)


###########################
# Extract Profiles
###########################

## define profile variables
mlv_azi = 135
gey_azi = 45

# profile distances and width in km. 
p_dist = 30
width = 1

# Download DEM
grid = pygmt.datasets.load_earth_relief(region=mlv_region, resolution="03s")
dgrid = pygmt.grdgradient(grid=grid, radiance=[315, 30], region=mlv_region)

### Define x distance bin spacing to closer near to zero profiles
dist_bins =np.arange(-p_dist, p_dist, 0.5)

### Get start and end points for plotting
A_start_lon, A_start_lat, A_end_lon, A_end_lat = utils.get_start_end_points(mlv_center_lo, mlv_center_la, mlv_azi, p_dist)
up_a_points, up_a_mean, up_a_offset = utils.extract_profiles(mlv_data, mlv_center_lo, mlv_center_la, mlv_azi, p_dist, width, dist_bins)
dem_track_df = pygmt.project(center=f"{A_start_lon}/{A_start_lat}", endpoint=f"{A_end_lon}/{A_end_lat}", generate="0.002")
dem_track_df = pygmt.grdtrack(grid=grid, points=dem_track_df, newcolname="elevation")
dem_xyz = dem_track_df[["r", "s", "elevation"]]
mlv_dem_a_points, dem_a_mean, dem_a_offset = utils.extract_profiles(dem_xyz, mlv_center_lo, mlv_center_la, mlv_azi, p_dist, width, dist_bins)

B_start_lon, B_start_lat, B_end_lon, B_end_lat = utils.get_start_end_points(mlv_center_lo, mlv_center_la, gey_azi, p_dist)
up_b_points, up_b_mean, up_b_offset = utils.extract_profiles(mlv_data, mlv_center_lo, mlv_center_la, gey_azi, p_dist, width, dist_bins)
dem_track_df = pygmt.project(center=f"{B_start_lon}/{B_start_lat}", endpoint=f"{B_end_lon}/{B_end_lat}", generate="0.002")
dem_track_df = pygmt.grdtrack(grid=grid, points=dem_track_df, newcolname="elevation")
dem_xyz = dem_track_df[["r", "s", "elevation"]]
get_dem_b_points, dem_b_mean, dem_b_offset = utils.extract_profiles(dem_xyz, mlv_center_lo, mlv_center_la, gey_azi, p_dist, width, dist_bins)

###########################
# Find maximum subsidence 
###########################

# Compute the mean "stable" value in the specified distance range
mean_up_a = up_a_mean.loc[(up_a_mean['distance'] >= -30) & (up_a_mean['distance'] <= -15), 'z'].mean()
mean_up_b = up_b_mean.loc[(up_b_mean['distance'] >= 15) & (up_b_mean['distance'] <= 30), 'z'].mean()
mean_up_bb = up_b_mean.loc[(up_b_mean['distance'] >= -30) & (up_b_mean['distance'] <= -15), 'z'].mean()


# Find the minimum (maximum subsidence) values for each profile
min_up_a = np.nanmin(up_a_mean["z"])
min_up_b = np.nanmin(up_b_mean["z"])  # corrected from up_a_mean to up_b_mean

# Calculate subsidence relative to the stable "zero" level
subsidence_a = min_up_a - mean_up_a
subsidence_b = min_up_b - mean_up_b
subsidence_bb = min_up_b - mean_up_bb

print("Subsidence A:", subsidence_a)
print("Subsidence B east:", subsidence_b)
print("Subsidence BB west:", subsidence_bb)

###########################
# Plot 
###########################

style="c.03c"
res_style="c.1c"
size = "M5.01c"

vmin = 20

sub_map_size = "M5.6c"
sub_proj = "X8/4.35c"
# NC_grid = pygmt.datasets.load_earth_relief(region=NC_fig_region, resolution="15s")
# NC_dgrid = pygmt.grdgradient(grid=NC_grid, radiance=[315, 30], region=NC_fig_region)

fig = pygmt.Figure()
pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain", FONT=10, FONT_TITLE=11, FONT_SUBTITLE = 11, MAP_TITLE_OFFSET= "-7p")

with fig.subplot(nrows=2, ncols=1, figsize=("6c", "9.35c"), autolabel="a)",sharex="b", sharey="l",
                  frame=["WSrt", "xa", "ya"], margins=["0.2c", "0.2c"]):
    
    # ## ******** MLV Map ********* 
    fig.basemap(region=mlv_region, projection=sub_map_size, frame=["lbtr", "xa", "ya"], panel=True,)
    fig.grdimage(grid=grid, projection=sub_map_size, frame=["lbrt", "xa", "ya"], cmap="grey", shading=dgrid, region=mlv_region)
    pygmt.makecpt(cmap="vik", series=[-(vmin)+(mean_up_a+mean_up_b)/2, (vmin)+(mean_up_a+mean_up_b)/2])
    fig.grdimage(grid=decomp["P784"]["insar_only_up_grd"], cmap=True, nan_transparent=True, region=mlv_region, projection= sub_map_size)
    
    fig.text(text="Shasta", x=shasta_lon, y=shasta_lat, justify="LM", offset="0.1c/0.1c", font="10p,Helvetica,black", region=mlv_region, projection= sub_map_size, fill="white", transparency=50)
    fig.text(text="Shasta", x=shasta_lon, y=shasta_lat, justify="LM", offset="0.1c/0.1c", font="10p,Helvetica,black", region=mlv_region, projection= sub_map_size)
    fig.plot(x=shasta_lon, y=shasta_lat,  style="t.25c", fill="black", pen="0.8p,black", region=mlv_region, projection= sub_map_size)
    
    for fault_file in common_paths["fault_files"]:
        fig.plot(data=fault_file, pen="0.5p,black", transparency=50, region=mlv_region, projection= sub_map_size)
    pygmt.makecpt(cmap="vik", series=[-vmin, vmin])
    fig.plot(x=mlv_gps_df["Lon"], y=mlv_gps_df["Lat"],  style="c.2c", fill=mlv_gps_df["Vu"], pen="0.8p,black", cmap=True, region=mlv_region, projection= sub_map_size)
    fig.plot(data=common_paths["MLV_level"], pen="1.2p,darkorange", region=mlv_region, projection=sub_map_size, transparency=40)
    fig.plot(x=[A_start_lon, A_end_lon], y=[A_start_lat, A_end_lat], pen="1.2p,black", region=mlv_region, projection=sub_map_size, transparency=40)
    fig.text(text="X", x=A_start_lon, y=A_start_lat, justify="RM", offset="-0.1c/0.1c", font="10p,Helvetica,black" , region=mlv_region, projection=sub_map_size, fill="white", transparency=50)
    fig.text(text="X'", x=A_end_lon, y=A_end_lat, justify="LM", offset="0.1c/-0.1c", font="10p,Helvetica,black" , region=mlv_region, projection=sub_map_size, fill="white", transparency=50)
    fig.text(text="X", x=A_start_lon, y=A_start_lat, justify="RM", offset="-0.1c/0.1c", font="10p,Helvetica,black" , region=mlv_region, projection=sub_map_size)
    fig.text(text="X'", x=A_end_lon, y=A_end_lat, justify="LM", offset="0.1c/-0.1c", font="10p,Helvetica,black" , region=mlv_region, projection=sub_map_size)
    
    fig.plot(x=[B_start_lon, B_end_lon], y=[B_start_lat, B_end_lat], pen="1.2p,black", region=mlv_region, projection=sub_map_size, transparency=40)
    fig.text(text="Y", x=B_start_lon, y=B_start_lat, justify="RM", offset="-0.1c/-0.1c", font="10p,Helvetica,black", region=mlv_region, projection=sub_map_size,  fill="white", transparency=50 )
    fig.text(text="Y'", x=B_end_lon, y=B_end_lat, justify="LM", offset="0.1c/0.1c", font="10p,Helvetica,black", region=mlv_region, projection=sub_map_size,  fill="white", transparency=50 )
    fig.text(text="Y", x=B_start_lon, y=B_start_lat, justify="RM", offset="-0.1c/-0.1c", font="10p,Helvetica,black", region=mlv_region, projection=sub_map_size )
    fig.text(text="Y'", x=B_end_lon, y=B_end_lat, justify="LM", offset="0.1c/0.1c", font="10p,Helvetica,black", region=mlv_region, projection=sub_map_size )
    
    fig.basemap(region=mlv_region, projection=sub_map_size, frame=["lbtr", "xa", "ya"], map_scale="jBL+w15k+o0.2c/0.5c")
    
    # ## ******** MLV DEM Map ********* 
    fig.basemap(region=mlv_region, projection=sub_map_size, frame=["lbtr", "xa", "ya"], panel=True,)
    fig.grdimage(grid=grid, projection=sub_map_size, frame=["lbrt", "xa", "ya"], cmap="grey", shading=dgrid, region=mlv_region)
    pygmt.makecpt(cmap="vik", series=[-vmin, vmin])
    #fig.grdimage(grid=decomp["P784"]["insar_only_grd"], cmap=True, nan_transparent=True, region=mlv_region, projection= sub_map_size)
    #fig.grdimage(grid="/Users/daniellelindsay/NC_Manuscript_Data_local/P784_068_170_Hz_Up/Up_068_170_P784.grd", cmap=True, nan_transparent=True, region=mlv_region, projection= sub_map_size)
    for fault_file in common_paths["fault_files"]:
        fig.plot(data=fault_file, pen="0.5p,black", transparency=50, region=mlv_region, projection= sub_map_size)
    pygmt.makecpt(cmap="vik", series=[-vmin, vmin])
    #fig.plot(x=mlv_gps_df["Lon"], y=mlv_gps_df["Lat"],  style="c.2c", fill=mlv_gps_df["Vu"], pen="0.8p,black", cmap=True, region=mlv_region, projection= sub_map_size)
    fig.plot(data=common_paths["MLV_level"], pen="1.2p,darkorange", region=mlv_region, projection=sub_map_size, transparency=40)
    fig.plot(x=[A_start_lon, A_end_lon], y=[A_start_lat, A_end_lat], pen="1.2p,black", region=mlv_region, projection=sub_map_size, transparency=40)
    fig.text(text="X", x=A_start_lon, y=A_start_lat, justify="RM", offset="-0.1c/0.1c", font="10p,Helvetica,black" , region=mlv_region, projection=sub_map_size, fill="white", transparency=50)
    fig.text(text="X'", x=A_end_lon, y=A_end_lat, justify="LM", offset="0.1c/-0.1c", font="10p,Helvetica,black" , region=mlv_region, projection=sub_map_size, fill="white", transparency=50)
    fig.text(text="X", x=A_start_lon, y=A_start_lat, justify="RM", offset="-0.1c/0.1c", font="10p,Helvetica,black" , region=mlv_region, projection=sub_map_size)
    fig.text(text="X'", x=A_end_lon, y=A_end_lat, justify="LM", offset="0.1c/-0.1c", font="10p,Helvetica,black" , region=mlv_region, projection=sub_map_size)
    
    fig.plot(x=[B_start_lon, B_end_lon], y=[B_start_lat, B_end_lat], pen="1.2p,black", region=mlv_region, projection=sub_map_size, transparency=40)
    fig.text(text="Y", x=B_start_lon, y=B_start_lat, justify="RM", offset="-0.1c/-0.1c", font="10p,Helvetica,black", region=mlv_region, projection=sub_map_size, fill="white", transparency=50 )
    fig.text(text="Y'", x=B_end_lon, y=B_end_lat, justify="LM", offset="0.1c/0.1c", font="10p,Helvetica,black", region=mlv_region, projection=sub_map_size, fill="white", transparency=50 )
    fig.text(text="Y", x=B_start_lon, y=B_start_lat, justify="RM", offset="-0.1c/-0.1c", font="10p,Helvetica,black", region=mlv_region, projection=sub_map_size )
    fig.text(text="Y'", x=B_end_lon, y=B_end_lat, justify="LM", offset="0.1c/0.1c", font="10p,Helvetica,black", region=mlv_region, projection=sub_map_size )
    
    fig.plot(x=[mlv_center_lo], y=[mlv_center_la], size=[radius*2], style="E-", pen="1.5p,white", region=mlv_region, projection= sub_map_size)
    
    with pygmt.config(
            FONT_ANNOT_PRIMARY="18p,black", 
            FONT_ANNOT_SECONDARY="18p,black",
            FONT_LABEL="18p,black",
            ):
        pygmt.makecpt(cmap="vik", series=[-vmin, vmin])
        fig.colorbar(position="jBL+o0.2c/0.2c+w2c/0.3c", frame=["xaf", "y+lmm/yr"],)

fig.shift_origin(xshift="wc")
with fig.subplot(nrows=2, ncols=1, figsize=("6c", "9.35c"), autolabel="c)",sharex="b", sharey="l",
                  frame=["WSrt", "xa", "ya"], margins=["0.2c", "0.2c"]):
    
    # ## ******** MLV Profile ********* 
    up_region =[-p_dist-0.5, p_dist+0.5, -23, 18]
    
    fig.basemap(region=up_region, projection=sub_proj, panel=True, frame=["lsEt" , "xaf+l Profile Distance (km)", "yaf+lVelocity Up (mm/yr)"])    
    pygmt.makecpt(cmap="vik", series=[-20, 20])
    fig.plot(x=up_a_points.p, y=up_a_points.z-mean_up_a, style="c0.1c", fill=up_a_points.z-mean_up_a, cmap=True, transparency=70, projection=sub_proj)
    fig.plot(x=up_a_mean.p,   y=up_a_mean.z-mean_up_a, pen="1.2p,black", projection=sub_proj) #style="c0.1c", fill="black")
    fig.plot(x=dem_a_mean.p,   y=((dem_a_mean.z-1000)/100), pen="1.2p,black", projection=sub_proj) #style="c0.1c", fill="black")
    fig.text(text="X → X'", position="TC", offset="0c/-0.2c", region=up_region, projection=sub_proj)
    fig.plot(x=-20, y=0, style="v0.3c+b+e+h0.15", direction=([90], [subsidence_a/10]), pen="0.8p,black", fill="black", region=up_region, projection=sub_proj)
    fig.plot(x=[-25, 0], y=[subsidence_a, subsidence_a],  pen="0.8p,black,--", region=up_region, projection=sub_proj)
    fig.text(text="%s mm/yr" % np.round(subsidence_a,1), x=-20, y=subsidence_a-2, justify="TC", region=up_region, projection=sub_proj)
    
    # ## ******** MLV Profile ********* 
    fig.basemap(region=up_region, projection=sub_proj, panel=True, frame=["lSEt" , "xaf+l Profile Distance (km)", "yaf+lVelocity Up (mm/yr)"])    
    pygmt.makecpt(cmap="vik", series=[-20, 20])
    fig.plot(x=up_b_points.p, y=up_b_points.z-mean_up_b, style="c0.1c", fill=up_b_points.z-mean_up_b, cmap=True, transparency=70, projection=sub_proj)
    fig.plot(x=up_b_mean.p,   y=up_b_mean.z-mean_up_b, pen="1.2p,black", projection=sub_proj) #style="c0.1c", fill="black")
    fig.plot(x=dem_b_mean.p,   y=((dem_b_mean.z-1000)/100)+0 , pen="1.2p,black", projection=sub_proj) #style="c0.1c", fill="black")
    fig.text(text="Y → Y'", position="TC", offset="0c/-0.2c", region=up_region, projection=sub_proj)
    fig.plot(x=[-25, 25], y=[min_up_b-mean_up_b, min_up_b-mean_up_b],  pen="0.8p,black,--", region=up_region, projection=sub_proj)
    fig.plot(x=20, y=min_up_b-mean_up_b, style="v0.3c+b+e+h0.15", direction=([90], [np.abs(subsidence_b)/10]), pen="0.8p,black", fill="black", region=up_region, projection=sub_proj)
    fig.text(text="%s mm/yr" % np.round(subsidence_b,1), x=20, y=subsidence_b-mean_up_b-4, justify="TC", region=up_region, projection=sub_proj)
    fig.plot(x=-20, y=min_up_b-mean_up_b, style="v0.3c+b+e+h0.15", direction=([90], [np.abs(subsidence_bb)/10]), pen="0.8p,black", fill="black", region=up_region, projection=sub_proj)
    fig.text(text="%s mm/yr" % np.round(subsidence_bb,1), x=-20, y=subsidence_b-mean_up_b-4, justify="TC", region=up_region, projection=sub_proj)
    
fig.shift_origin(xshift="9.75c")   

pygmt.makecpt(cmap="acton", series=[0, 100, radius])

scat_region = [np.nanmin(df['Vel']), np.nanmax(df['Vel']), np.nanmin(df['height']), np.nanmax(df['height']), ]
fig.basemap(region=scat_region, projection="X5c/5c", frame=["lSEt" , "xaf+l Velocity Up (mm/yr)", "yaf+lElevation (m)"])
fig.plot(y=df['height'], x=df['Vel']-(mean_up_a+mean_up_b)/2, style="c.08c", fill=df['dist2ref'], cmap=True, region=scat_region, projection="X5c/5c", transparency=70)
fig.text(text=f"R² > 20km {r2:.2f}", position="TL", offset="0.2c/-0.2c", region=scat_region, projection="X5c/5c")
fig.text(text=f"R² ≤ 20km {r2_inside:.2f}", position="TL", offset="0.2c/-0.7c", region=scat_region, projection="X5c/5c")
fig.text(text="e)", position="TR", offset="-0.2c/-0.2c", region=scat_region, projection="X5c/5c")
fig.text(text="Shasta", x=0, y=3000, region=scat_region, projection="X5c/5c")
fig.text(text="MLV", x=-6, y=1900, region=scat_region, projection="X5c/5c", font="white")
fig.text(text="Agriculture", x=-15, y=1000, region=scat_region, projection="X5c/5c")

with pygmt.config(
        FONT_ANNOT_PRIMARY="18p,black", 
        FONT_ANNOT_SECONDARY="18p,black",
        FONT_LABEL="18p,black",
        ):
    fig.colorbar(cmap=True, position="JCT+o0c/0.5c+w5c/0.5c", frame=["x+lDistance from MLV", "y+lkm"],)

fig.savefig(common_paths["fig_dir"]+f"Fig_9_MLV_Vertial_example_{radius}.png", transparent=False, crop=True, anti_alias=True, show=False)
fig.savefig(common_paths["fig_dir"]+f"Fig_9_MLV_Vertial_example_{radius}.pdf", transparent=False, crop=True, anti_alias=True, show=False)
fig.show()  

# fig = pygmt.Figure()
# scat_region = [np.nanmin(dem_a_mean.p), np.nanmax(dem_a_mean.p), np.nanmin(dem_b_mean.z)-200, np.nanmax(dem_b_mean.z)]
# fig.basemap(region=scat_region, projection="X10/5c", frame=["lSEt" , "xaf+l Velocity Up (mm/yr)", "ya100f100g+lElevation (m)"])
# fig.plot(x=dem_a_mean.p,   y=(dem_a_mean.z), pen="1.2p,black", projection=sub_proj) #style="c0.1c", fill="black")
# fig.plot(x=dem_b_mean.p,   y=(dem_b_mean.z), pen="1.2p,black", projection=sub_proj) #style="c0.1c", fill="black")
# fig.show()  
