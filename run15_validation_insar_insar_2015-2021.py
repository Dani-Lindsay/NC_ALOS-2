#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 16:02:40 2025

@author: daniellelindsay
"""

 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 11:55:15 2025

@author: daniellelindsay
"""

from NC_ALOS2_filepaths import (common_paths, paths_068, paths_169, paths_170, paths_170_5_28, paths_115, paths_gps)
import insar_utils as utils
import pygmt
import pandas as pd
import numpy as np

unit = 100

# set your bin size in cm/yr
#bin_size = 0.01

ref_station = common_paths["ref_station"]




asc_des_min_lat = 38.9 - 0.1
asc_des_max_lat = 39.1 + 0.1
asc_des_min_lon = -122.2 - 0.07
asc_des_max_lon = -122.0 + 0.13
region_asc = f"{asc_des_min_lon}/{asc_des_max_lon}/{asc_des_min_lat}/{asc_des_max_lat}"
vel_region_asc = [-10, 1, -10, 1]  # for scatter plot axes

# For Descending comparison
map_size_des = "M2.6c"
scatter_size_des = "X5.1/5.1c"

des_min_lat = 37.0
des_max_lat = 42.2
des_min_lon = -124.4
des_max_lon = -121.0
region_des = f"{des_min_lon}/{des_max_lon}/{des_min_lat}/{des_max_lat}"
vel_region_des = [-3.5, 3.5, -3.5, 3.5]



def prepare_heatmap_xyz(df, x_col, y_col, bin_size=0.1):
    min_val = df[[x_col, y_col]].min().min(); max_val = df[[x_col, y_col]].max().max()
    edges = np.arange(np.floor(min_val/bin_size)*bin_size,
                      np.ceil(max_val/bin_size)*bin_size + bin_size,
                      bin_size)
    counts, xedges, yedges = np.histogram2d(df[x_col], df[y_col], bins=[edges, edges])
    x_centers = (xedges[:-1] + xedges[1:]) / 2; y_centers = (yedges[:-1] + yedges[1:]) / 2
    heatmap_df = pd.DataFrame(counts,
                              index=pd.Index(x_centers, name=f"{x_col}_bin"),
                              columns=pd.Index(y_centers, name=f"{y_col}_bin"))
    return (
        heatmap_df
        .stack()
        .reset_index(name='count')
        .loc[lambda d: d['count'] > 0]
        .assign(count=lambda d: d['count'] / d['count'].max())
        .sort_values('count')
    )

# Load UNR data
gps_df = utils.load_UNR_gps(paths_gps["170_enu_IGS14"])
ref = gps_df.loc[gps_df['StaID']==ref_station]
if ref.empty:
    raise ValueError(f"Reference station '{ref_station}' not in gps_df")
ref_lon, ref_lat, ref_Ve, ref_Vn, ref_Vu = ref[['Lon','Lat', 'Ve', 'Vn', 'Vu']].iloc[0]

gps_df['Ve'] = gps_df['Ve'] - ref_Ve
gps_df['Vn'] = gps_df['Vn'] - ref_Vn
gps_df['Vu'] = gps_df['Vu'] - ref_Vu

##############################
# Prepare 170, 169 comparison
##############################

# Load deramped descending data for tracks 169 and 170
des169_lon, des169_lat, des169_vel, des169_azi, des169_inc = utils.load_insar_vel_data_as_2Darrays(
    paths_169["geo"]["geo_geometryRadar"],
    paths_169["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"]
)
des170_lon, des170_lat, des170_vel, des170_azi, des170_inc = utils.load_insar_vel_data_as_2Darrays(
    "/Volumes/WD2TB_Phd/NC_ALOS-2/Data_Zenodo/170/CASR_15_21/geo/geo_geometryRadar.h5",
    "/Volumes/WD2TB_Phd/NC_ALOS-2/Data_Zenodo/170/CASR_15_21/geo/geo_velocity_SET_ERA5_demErr_ITRF14_msk_deramp.h5"
)


# Convert velocities
des169_vel = des169_vel * unit
des170_vel = des170_vel * unit

# Build DataFrame for descending comparison and filter valid data
df_des_diff = pd.DataFrame({
    'des169_vel': des169_vel.ravel(),
    'des170_vel': des170_vel.ravel(),
    })
df_des_diff = df_des_diff.dropna(subset=['des169_vel', 'des170_vel'])

rmse_des, r2_des, slope_des, intercept_des = utils.calculate_rmse_r2_and_linear_fit(
    df_des_diff['des169_vel'], df_des_diff['des170_vel']
)
print(f"Des 169/170: RMSE {rmse_des:.2f}, R² {r2_des:.2f}, slope {slope_des:.2f}")

# Generate 2D histogram for plotting
xyz_df_des_des = prepare_heatmap_xyz(df_des_diff, 'des169_vel', 'des170_vel', bin_size=0.01)



############################
# Start Plotting 
#############################3
# Define plotting parameters for each block
# For Ascending/Descending comparison
map_size_asc = "M4c"
scatter_size_asc = "X5.1/5.1c"



fig = pygmt.Figure()
pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain", FONT=9, FONT_TITLE=10, MAP_TITLE_OFFSET="-7p")


# --- Second subplot block: 4 panels for Descending Comparison (Tracks 169 vs. 170) ---
with fig.subplot(nrows=1, ncols=4, figsize=("11c", "5.1c"), autolabel="d)", sharex="l", frame=["WSrt"], margins=["0.3c", "0.3c"]):
    
    ref_station = common_paths["ref_station"]
    ref_lat = gps_df.loc[gps_df["StaID"] == ref_station, "Lat"].values
    ref_lon = gps_df.loc[gps_df["StaID"] == ref_station, "Lon"].values
    
    vmin, vmax = -0.02, 0.02
    
    # Panel 1: Descending 169 map
    fig.basemap(frame=["WSrt", "xa", "ya"], region=region_des, projection=map_size_des, panel=True)
    pygmt.makecpt(cmap="vik", series=[vmin, vmax])
    fig.grdimage(grid=paths_169["grd"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"], cmap=True, region=[region_des], projection=map_size_des)
    fig.coast(shorelines=True, region=[region_des], projection=map_size_des, frame=["+t169: 2015-2021"])
    
    fig.plot(y=ref_lat, x=ref_lon, style="s.1c", fill="black", pen="0.8p,black", region=[region_des], projection=map_size_des,)
    
    df = pd.DataFrame(
        data={
            "x": [-123.5,-123.5 ],
            "y": [38.2,38.2],
            "east_velocity": [-0.173*0.8,(-0.9848/2.5)*0.8],
            "north_velocity": [-0.9848*0.8, (0.173/2.5)*0.8],
            "east_sigma": [0,0],
            "north_sigma": [0,0],
            "correlation_EN": [0,0],
            "SITE": ["",""],
            })
    fig.velo(data=df, pen="0.5p,black", line=True, spec="e1.2/1/1", vector="0.3c+p2p+e+gblack",region=region_des, projection=map_size_des)

    fig.basemap(frame=["WSrt", "xa", "ya"], region=region_des, projection=map_size_des)
    
    # Panel 2: Descending 170 map
    fig.basemap(frame=["wSrt", "xa", "ya"], region=region_des, projection=map_size_des, panel=True)
    pygmt.makecpt(cmap="vik", series=[vmin, vmax])
    fig.grdimage(grid="/Volumes/WD2TB_Phd/NC_ALOS-2/Data_Zenodo/170/CASR_15_21/geo/geo_velocity_SET_ERA5_demErr_ITRF14_msk_deramp.grd", cmap=True, region=[region_des], projection=map_size_des)
    fig.coast(shorelines=True, region=[region_des], projection=map_size_des, frame=["+t170: 2015-2021"])
    fig.plot(y=ref_lat, x=ref_lon, style="s.1c", fill="black", pen="0.8p,black", region=[region_des], projection=map_size_des,)
    
    df = pd.DataFrame(
        data={
            "x": [-123.5,-123.5 ],
            "y": [38.2,38.2],
            "east_velocity": [-0.173*0.8,(-0.9848/2.5)*0.8],
            "north_velocity": [-0.9848*0.8, (0.173/2.5)*0.8],
            "east_sigma": [0,0],
            "north_sigma": [0,0],
            "correlation_EN": [0,0],
            "SITE": ["",""],
            })
    fig.velo(data=df, pen="0.5p,black", line=True, spec="e1.2/1/1", vector="0.3c+p2p+e+gblack",region=region_des, projection=map_size_des)
    
    fig.basemap(frame=["wSrt", "xa", "ya"],
                region=region_des, projection=map_size_des)
    
    # Panel 3: Difference map (Des 169 minus Des 170)
    fig.basemap(frame=["wSrt", "xa", "ya"], region=region_des, projection=map_size_des, panel=True)
    pygmt.makecpt(cmap="vik", series=[vmin, vmax])
    fig.grdimage(grid="/Volumes/WD2TB_Phd/NC_ALOS-2/Data_Zenodo/170/CASR_15_21/geo/diff_169_170_2015_2021.grd", cmap=True, region=[region_des], projection=map_size_des)
    fig.coast(shorelines=True, region=[region_des], projection=map_size_des, frame=["+tDifference"])

    fig.plot(y=ref_lat, x=ref_lon, style="s.1c", fill="black", pen="0.8p,black", region=[region_des], projection=map_size_des,)
    
    fig.plot(x = [asc_des_min_lon, asc_des_min_lon, asc_des_max_lon, asc_des_max_lon, asc_des_min_lon], 
             y = [asc_des_min_lat, asc_des_max_lat, asc_des_max_lat, asc_des_min_lat, asc_des_min_lat], 
             pen="0.5p,black", transparency=0, region=[region_des], projection=map_size_des,)
    
    # Inset map of frames
    # Define region of interest 
    min_lon=-126.5
    max_lon=-119.0
    min_lat=35.5
    max_lat=43

    sub_region="%s/%s/%s/%s" % (min_lon, max_lon, min_lat, max_lat)
    
    fig.coast(shorelines=True, area_thresh=5000, frame="+t", projection="M1.1c", region=[sub_region])
    fig.plot(data=common_paths['frames']['170_2800'] , pen="0.5p,black", transparency=30, projection="M1.1c", region=[sub_region])
    fig.plot(data=common_paths['frames']['170_2850'] , pen="0.5p,black", transparency=30, projection="M1.1c", region=[sub_region])
    fig.plot(data=common_paths['frames']['169_2800'] , pen="0.5p,black", transparency=30, projection="M1.1c", region=[sub_region])
    fig.plot(data=common_paths['frames']['169_2850'] , pen="0.5p,black", transparency=30, projection="M1.1c", region=[sub_region])
    fig.plot(data=common_paths['frames']['169_170_overlap'] , pen="0.5p,black", transparency=30, fill="grey", projection="M1.1c", region=[sub_region])
    fig.coast(shorelines=True, area_thresh=5000, frame="+t", projection="M1.1c", region=[sub_region])
    fig.basemap(frame=["wSrt", "xa", "ya"], region=region_des, projection=map_size_des)
    
    # # Panel 4: Scatter plot (Des 169 vs. Des 170)
    fig.basemap(frame=["lSEt", "xaf+lDescending 169 (cm/yr)", "yaf+lDescending LOS 170 (cm/yr)"],
                region=vel_region_des, projection=scatter_size_des, panel=True)
    
    pygmt.makecpt(cmap="gray", series=[-0.1, 0.3, 0.05], reverse=True)
    fig.plot(x=xyz_df_des_des['des169_vel_bin'], y=xyz_df_des_des['des170_vel_bin'], style="s0.05c",
              fill=xyz_df_des_des["count"], cmap=True, region=vel_region_des, projection=scatter_size_des)
    fig.plot(x=[-3.5,3.5], y=[-3.5,3.5], pen="0.5p,black", region=vel_region_des, projection=scatter_size_des)
    
    fig.text(text="RMSE {} cm/yr".format(np.round(rmse_des,2)), position="BR",
             offset="-0.2c/0.65c", font="9p,Helvetica,black", region=vel_region_des, projection=scatter_size_des)
    fig.text(text="R² {}, Slope {}".format(np.round(r2_des,2), np.round(slope_des,2)), position="BR",
             offset="-0.2c/0.2c", font="9p,Helvetica,black", region=vel_region_des, projection=scatter_size_des)
    
    with pygmt.config(FONT_ANNOT_PRIMARY="18p,black", FONT_ANNOT_SECONDARY="18p,black", FONT_LABEL="18p,black"):
        pygmt.makecpt(cmap="vik", series=[-2, 2])
        fig.colorbar(position="jCL+o0.25c/0.5c+w2.7c/0.3c",
                     frame=["xaf", "y+lcm/yr"], region=region_des, projection=map_size_des)
        
    fig.basemap(frame=["lSEt", "xaf+lDescending 169 (cm/yr)", "yaf+lDescending LOS 170 (cm/yr)"],
                region=vel_region_des, projection=scatter_size_des)

# Save and display the final figure
fig.savefig(common_paths["fig_dir"]+f'Fig_7_insar_insar_validation_{ref_station}_2015-2021.png', transparent=False, crop=True, anti_alias=True, show=False)
fig.savefig(common_paths["fig_dir"]+f'Fig_7_insar_insar_validation_{ref_station}_2015-2021.pdf', transparent=False, crop=True, anti_alias=True, show=False)
fig.savefig(common_paths["fig_dir"]+f'Fig_7_insar_insar_validation_{ref_station}_2015-2021.jpg', dpi=600, transparent=False, crop=True, anti_alias=True, show=False)
fig.show()