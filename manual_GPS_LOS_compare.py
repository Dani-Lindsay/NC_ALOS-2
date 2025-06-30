#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 10:26:56 2025

@author: daniellelindsay
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 13:26:10 2025

@author: daniellelindsay
"""

#from NC_manuscript_filepaths_June2 import (paths_068, paths_169, paths_170, paths_gps, fig_dir, common_paths)

from NC_ALOS2_filepaths import (paths_gps, paths_068, paths_169, paths_170, common_paths)
import insar_utils as utils
import numpy as np                # Matrix calculations
import pandas as pd               # Pandas for data
import pygmt

dist = common_paths["dist"]
ref_station = common_paths["ref_station"]
ref_lat = common_paths["ref_lat"]
ref_lon = common_paths["ref_lon"]

#lat_step = common_paths["lat_step"]
#lon_step = common_paths["lon_step"]

unit = 1000
#####################
# Load InSAR
#####################

insar_169 = utils.load_h5_data(paths_169["geo"]["geo_geometryRadar"], paths_169["geo"]["geo_velocity_msk"], "velocity")
insar_170 = utils.load_h5_data(paths_170["geo"]["geo_geometryRadar"], paths_170["geo"]["geo_velocity_msk"], "velocity")
insar_068 = utils.load_h5_data(paths_068["geo"]["geo_geometryRadar"], paths_068["geo"]["geo_velocity_msk"], "velocity")
#insar_068 = utils.load_h5_data(paths_068["geo"]["geo_geometryRadar"], paths_068["geo"]["geo_velocity_msk"], "velocity")

# grid_169 = "/Volumes/WD2TB_Phd/NC_ALOS-2/Data_Zenodo/169/CASR/geo/geo_velocity_msk.grd"
# grid_170 = "/Volumes/WD2TB_Phd/NC_ALOS-2/Data_Zenodo/170/CASR/geo/geo_velocity_msk.grd"
# grid_068 = "/Volumes/WD2TB_Phd/NC_ALOS-2/Data_Zenodo/068/CASR/geo/geo_velocity_msk.grd"

grid_169 = paths_169["grd"]["geo_velocity_msk"]
grid_170 = paths_170["grd"]["geo_velocity_msk"]
grid_068 = paths_068["grd"]["geo_velocity_msk"]


# insar_169 = utils.load_h5_data('/Volumes/WD2TB_Phd/NC_ALOS-2/Data_Zenodo/169/CASR/geo/geo_geometryRadar.h5', 
#                                '/Volumes/WD2TB_Phd/NC_ALOS-2/Data_Zenodo/169/CASR/geo/geo_velocity_msk.h5', "velocity")

# insar_170 = utils.load_h5_data('/Volumes/WD2TB_Phd/NC_ALOS-2/Data_Zenodo/170/CASR/geo/geo_geometryRadar.h5', 
#                                '/Volumes/WD2TB_Phd/NC_ALOS-2/Data_Zenodo/170/CASR/geo/geo_velocity_msk.h5', "velocity")

# insar_068 = utils.load_h5_data('/Volumes/WD2TB_Phd/NC_ALOS-2/Data_Zenodo/068/CASR/geo/geo_geometryRadar.h5', 
#                                '/Volumes/WD2TB_Phd/NC_ALOS-2/Data_Zenodo/068/CASR/geo/geo_velocity_msk.h5', "velocity")


# insar_169 = utils.load_h5_data('/Volumes/WD2TB_Phd/NC_ALOS-2/Data/169/CASR/geo/geo_geometryRadar.h5', 
#                                paths_169["geo"]["geo_velocity_msk"], "velocity")

# insar_170 = utils.load_h5_data('/Volumes/WD2TB_Phd/NC_ALOS-2/Data/170/CASR/geo/geo_geometryRadar.h5', 
#                                paths_170["geo"]["geo_velocity_msk"], "velocity")

# insar_068 = utils.load_h5_data('/Volumes/WD2TB_Phd/NC_ALOS-2/Data/068/CASR/geo/geo_geometryRadar.h5', 
#                                paths_068["geo"]["geo_velocity_msk"], "velocity")



insar_169_std = utils.load_h5_data(paths_169["geo"]["geo_geometryRadar"], paths_169["geo"]["geo_velocity"], "velocityStd")
insar_170_std = utils.load_h5_data(paths_170["geo"]["geo_geometryRadar"], paths_170["geo"]["geo_velocity"], "velocityStd")
insar_068_std = utils.load_h5_data(paths_068["geo"]["geo_geometryRadar"], paths_068["geo"]["geo_velocity"], "velocityStd")


# insar_169_std = utils.load_h5_data('/Volumes/WD2TB_Phd/NC_ALOS-2/Data_Zenodo/169/CASR/geo/geo_geometryRadar.h5', 
#                                    '/Volumes/WD2TB_Phd/NC_ALOS-2/Data_Zenodo/169/CASR/geo/geo_velocity.h5', "velocityStd")
# insar_170_std = utils.load_h5_data('/Volumes/WD2TB_Phd/NC_ALOS-2/Data_Zenodo/170/CASR/geo/geo_geometryRadar.h5', 
#                                    '/Volumes/WD2TB_Phd/NC_ALOS-2/Data_Zenodo/170/CASR/geo/geo_velocity.h5', "velocityStd")
# insar_068_std = utils.load_h5_data('/Volumes/WD2TB_Phd/NC_ALOS-2/Data_Zenodo/068/CASR/geo/geo_geometryRadar.h5', 
#                                    '/Volumes/WD2TB_Phd/NC_ALOS-2/Data_Zenodo/068/CASR/geo/geo_velocity.h5', "velocityStd")

insar_169['Std'] = insar_169_std['Vel']*unit
insar_170['Std'] = insar_170_std['Vel']*unit
insar_068['Std'] = insar_068_std['Vel']*unit

insar_169['Vel'] = insar_169['Vel']*unit
insar_170['Vel'] = insar_170['Vel']*unit
insar_068['Vel'] = insar_068['Vel']*unit

#####################
# Load in GPS and project UNR enu --> los
#####################

# gps_169 = utils.load_UNR_gps('/Volumes/WD2TB_Phd/NC_ALOS-2/Data_Zenodo/068/UNR_IGS14_gps.csv', ref_station)
# gps_170 = utils.load_UNR_gps('/Volumes/WD2TB_Phd/NC_ALOS-2/Data_Zenodo/170/UNR_IGS14_gps.csv', ref_station)
# gps_068 = utils.load_UNR_gps('/Volumes/WD2TB_Phd/NC_ALOS-2/Data_Zenodo/169/UNR_IGS14_gps.csv', ref_station)

gps_169 = utils.load_UNR_gps(paths_gps["169_enu"], ref_station)
gps_170 = utils.load_UNR_gps(paths_gps["170_enu"], ref_station)
gps_068 = utils.load_UNR_gps(paths_gps["068_enu"], ref_station)

gps_169 = utils.calculate_gps_los(gps_169, insar_169)
gps_170 = utils.calculate_gps_los(gps_170, insar_170)
gps_068 = utils.calculate_gps_los(gps_068, insar_068)

gps_169 = utils.calculate_gps_los_error(gps_169, insar_169)
gps_170 = utils.calculate_gps_los_error(gps_170, insar_170)
gps_068 = utils.calculate_gps_los_error(gps_068, insar_068)

#####################
# Find average InSAR velocity for each GPS point 
#####################
gps_169 = utils.calculate_average_insar_velocity(gps_169, insar_169, dist)
gps_170 = utils.calculate_average_insar_velocity(gps_170, insar_170, dist)
gps_068 = utils.calculate_average_insar_velocity(gps_068, insar_068, dist)

#####################
# Calculate stats for no shift 
#####################

# Save pre-shift values for histogram
gps_169_ori = gps_169
gps_170_ori = gps_170
gps_068_ori = gps_068

unr_rmse_169_ori, unr_r2_169_ori, unr_slope_169_ori, unr_intercept_169_ori = utils.calculate_rmse_r2_and_linear_fit(gps_169_ori['LOS_Vel'], gps_169_ori['insar_Vel'])
unr_rmse_170_ori, unr_r2_170_ori, unr_slope_170_ori, unr_intercept_170_ori = utils.calculate_rmse_r2_and_linear_fit(gps_170_ori['LOS_Vel'], gps_170_ori['insar_Vel'])
unr_rmse_068_ori, unr_r2_068_ori, unr_slope_068_ori, unr_intercept_068_ori = utils.calculate_rmse_r2_and_linear_fit(gps_068_ori['LOS_Vel'], gps_068_ori['insar_Vel'])


print("\nTrack 169 (no static shift):")
print(f"RMSE: {unr_rmse_169_ori:.3f}, R²: {unr_r2_169_ori:.3f}, "
      f"Slope: {unr_slope_169_ori:.3f}, Intercept: {unr_intercept_169_ori:.3f}")
res_per_169_ori, res_std_169_ori = utils.calc_residual_percent(gps_169_ori, 2)
print(f"% residuals ≤2 mm/yr: {res_per_169_ori:.1f}% ± {res_std_169_ori:.1f}")

print("\nTrack 170 (cno static shift):")
print(f"RMSE: {unr_rmse_170_ori:.3f}, R²: {unr_r2_170_ori:.3f}, "
      f"Slope: {unr_slope_170_ori:.3f}, Intercept: {unr_intercept_170_ori:.3f}")
res_per_170_ori, res_std_170_ori = utils.calc_residual_percent(gps_170_ori, 2)
print(f"% residuals ≤2 mm/yr: {res_per_170_ori:.1f}% ± {res_std_170_ori:.1f}")

print("\nTrack 068 (no static shift):")
print(f"RMSE: {unr_rmse_068_ori:.3f}, R²: {unr_r2_068_ori:.3f}, "
      f"Slope: {unr_slope_068_ori:.3f}, Intercept: {unr_intercept_068_ori:.3f}")
res_per_068_ori, res_std_068_ori = utils.calc_residual_percent(gps_068_ori, 2)
print(f"% residuals ≤2 mm/yr: {res_per_068_ori:.1f}% ± {res_std_068_ori:.1f}")


#####################
# Calculate static shift 
##################### 

print("Track 169:")
shift_169 = np.round(np.nanmean(gps_169["LOS_Vel"]- gps_169["insar_Vel"]),2)
print("Static shift: %s" %  shift_169)
print(" ")
print("Track 170:")
shift_170 = np.round(np.nanmean(gps_170["LOS_Vel"]- gps_170["insar_Vel"]),2)
print("Static shift: %s" %  shift_170)
print(" ")
print("Track 068:")
shift_068 = np.round(np.nanmean(gps_068["LOS_Vel"]- gps_068["insar_Vel"]),2)
print("Static shift: %s" % shift_068)

gps_169['insar_Vel'] = gps_169['insar_Vel'] + np.nanmean(gps_169["LOS_Vel"]- gps_169["insar_Vel"]) 
gps_170['insar_Vel'] = gps_170['insar_Vel'] + np.nanmean(gps_170["LOS_Vel"]- gps_170["insar_Vel"]) 
gps_068['insar_Vel'] = gps_068['insar_Vel'] + np.nanmean(gps_068["LOS_Vel"]- gps_068["insar_Vel"]) 

#####################
# Calculate distance to reference for plotting
#####################
gps_169 = utils.calculate_distance_to_reference(gps_169, ref_station)
gps_170 = utils.calculate_distance_to_reference(gps_170, ref_station)
gps_068 = utils.calculate_distance_to_reference(gps_068, ref_station)

#####################
# Calculate rmse, r2 and linear fit
#####################
unr_rmse_169, unr_r2_169, unr_slope_169, unr_intercept_169 = utils.calculate_rmse_r2_and_linear_fit(gps_169['LOS_Vel'], gps_169['insar_Vel'])
unr_rmse_170, unr_r2_170, unr_slope_170, unr_intercept_170 = utils.calculate_rmse_r2_and_linear_fit(gps_170['LOS_Vel'], gps_170['insar_Vel'])
unr_rmse_068, unr_r2_068, unr_slope_068, unr_intercept_068 = utils.calculate_rmse_r2_and_linear_fit(gps_068['LOS_Vel'], gps_068['insar_Vel'])


#####################
# Calculate percentage of residuals below 2 mm/yr
#####################    

# print(" ")
# print("Track 169:")
# print("RMSE: %s" % unr_rmse_169)
# unr_rmse_169, unr_r2_169, unr_slope_169, unr_intercept_169
# res_per_169, res_std_169 = utils.calc_residual_percent(gps_169, 2)
# print(" ")
# print("Track 170:")
# print("RMSE: %s" % unr_rmse_170)
# unr_rmse_170, unr_r2_170, unr_slope_170, unr_intercept_170
# res_per_170, res_std_170 = utils.calc_residual_percent(gps_170, 2)
# print(" ")
# print("Track 068:")
# print("RMSE: %s" % unr_rmse_068)
# unr_rmse_068, unr_r2_068, unr_slope_068, unr_intercept_068 
# res_per_068, res_std_068 = utils.calc_residual_percent(gps_068, 2)

print("\nTrack 169 (with static shift):")
print(f"RMSE: {unr_rmse_169:.3f}, R²: {unr_r2_169:.3f}, "
      f"Slope: {unr_slope_169:.3f}, Intercept: {unr_intercept_169:.3f}")
res_per_169, res_std_169 = utils.calc_residual_percent(gps_169, 2)
print(f"% residuals ≤2 mm/yr: {res_per_169:.1f}% ± {res_std_169:.1f}")

print("\nTrack 170 (with static shift):")
print(f"RMSE: {unr_rmse_170:.3f}, R²: {unr_r2_170:.3f}, "
      f"Slope: {unr_slope_170:.3f}, Intercept: {unr_intercept_170:.3f}")
res_per_170, res_std_170 = utils.calc_residual_percent(gps_170, 2)
print(f"% residuals ≤2 mm/yr: {res_per_170:.1f}% ± {res_std_170:.1f}")

print("\nTrack 068 (with static shift):")
print(f"RMSE: {unr_rmse_068:.3f}, R²: {unr_r2_068:.3f}, "
      f"Slope: {unr_slope_068:.3f}, Intercept: {unr_intercept_068:.3f}")
res_per_068, res_std_068 = utils.calc_residual_percent(gps_068, 2)
print(f"% residuals ≤2 mm/yr: {res_per_068:.1f}% ± {res_std_068:.1f}")

#####################
# Calculate RMSE, R² and linear fit on the common‐stations subsets
#####################

# find common station IDs
common_ids = (
    set(gps_170['StaID'])
    & set(gps_169['StaID'])
    & set(gps_068['StaID'])
)

# filter each DataFrame
gps_170_common = gps_170[gps_170['StaID'].isin(common_ids)].reset_index(drop=True)
gps_169_common = gps_169[gps_169['StaID'].isin(common_ids)].reset_index(drop=True)
gps_068_common = gps_068[gps_068['StaID'].isin(common_ids)].reset_index(drop=True)



# Track 169
unr_rmse_169_c, unr_r2_169_c, unr_slope_169_c, unr_intercept_169_c = \
    utils.calculate_rmse_r2_and_linear_fit(
        gps_169_common['LOS_Vel'],
        gps_169_common['insar_Vel']
    )

# Track 170
unr_rmse_170_c, unr_r2_170_c, unr_slope_170_c, unr_intercept_170_c = \
    utils.calculate_rmse_r2_and_linear_fit(
        gps_170_common['LOS_Vel'],
        gps_170_common['insar_Vel']
    )

# Track 068
unr_rmse_068_c, unr_r2_068_c, unr_slope_068_c, unr_intercept_068_c = \
    utils.calculate_rmse_r2_and_linear_fit(
        gps_068_common['LOS_Vel'],
        gps_068_common['insar_Vel']
    )


#####################
# Calculate percentage of residuals below 2 mm/yr on the common‐stations subsets
#####################
print("\nTrack 169 (common stations):")
print(f"RMSE: {unr_rmse_169_c:.3f}, R²: {unr_r2_169_c:.3f}, "
      f"Slope: {unr_slope_169_c:.3f}, Intercept: {unr_intercept_169_c:.3f}")
res_per_169_c, res_std_169_c = utils.calc_residual_percent(gps_169_common, 2)
print(f"% residuals ≤2 mm/yr: {res_per_169_c:.1f}% ± {res_std_169_c:.1f}")

print("\nTrack 170 (common stations):")
print(f"RMSE: {unr_rmse_170_c:.3f}, R²: {unr_r2_170_c:.3f}, "
      f"Slope: {unr_slope_170_c:.3f}, Intercept: {unr_intercept_170_c:.3f}")
res_per_170_c, res_std_170_c = utils.calc_residual_percent(gps_170_common, 2)
print(f"% residuals ≤2 mm/yr: {res_per_170_c:.1f}% ± {res_std_170_c:.1f}")

print("\nTrack 068 (common stations):")
print(f"RMSE: {unr_rmse_068_c:.3f}, R²: {unr_r2_068_c:.3f}, "
      f"Slope: {unr_slope_068_c:.3f}, Intercept: {unr_intercept_068_c:.3f}")
res_per_068_c, res_std_068_c = utils.calc_residual_percent(gps_068_common, 2)
print(f"% residuals ≤2 mm/yr: {res_per_068_c:.1f}% ± {res_std_068_c:.1f}")



# Save selected columns to CSV in one line
# # Specify columns to save
# columns_to_save = ['Lon', 'Lat', 'insar_Vel', 'LOS_Vel', 'StaID']
# #gps_UNR_169[columns_to_save].to_csv(outfile_169, index=False)
# gps_169[columns_to_save].to_csv(paths_gps["169_LOS_comp"], index=False, header=True)
# gps_170[columns_to_save].to_csv(paths_gps["170_LOS_comp"], index=False, header=True)
# gps_068[columns_to_save].to_csv(paths_gps["068_LOS_comp"], index=False, header=True)

#####################
# Scatter Plot
#####################    

# Find the minimum and maximum values in each DataFrame
min_values = [df['dist2ref'].min() for df in [ gps_170, gps_169, gps_068]] # gps_UNR_169,
max_values = [df['dist2ref'].max() for df in [ gps_068]] # gps_UNR_169,
min_dist = 0 #min(min_values)
max_dist = max(max_values)


# Find the overall minimum and maximum values across all DataFrames
min_values = [df['insar_Vel'].min() for df in [gps_170, gps_169, gps_068]] #  gps_UNR_169,
max_values = [df['insar_Vel'].max() for df in [gps_170, gps_169, gps_068]] #  gps_UNR_169,
min_val = min(min_values)-5
max_val = max(max_values)+5

xseq = np.linspace(min_val, max_val, num=100)


#####################
# Plot
#####################   

pygmt.config(FORMAT_GEO_MAP="ddd.xx", MAP_FRAME_TYPE="plain", FONT=10, FONT_TITLE=11,)

fig = pygmt.Figure()

with fig.subplot(nrows=1, ncols=3, figsize=("15c", "5.c"), autolabel="d)", sharex="b", sharey="l",
                 frame=["xaf+lInSAR LOS (mm/yr)", "yaf+lGPS LOS (mm/yr)", "WSrt"], margins=["0.2c", "0.2c"]):
    pygmt.config(FORMAT_GEO_MAP="ddd.xx", MAP_FRAME_TYPE="plain", FONT=10, FONT_TITLE=11,)
    pygmt.makecpt(cmap="viridis", series=[min_dist, max_dist])
    
    fig.basemap(region=[min_val, max_val, min_val, max_val], projection="X5c/5c", panel=True)
    for _, row in gps_169.iterrows():
        fig.plot(
            x=[row['insar_Vel']- row['insar_Vel_std'], row['insar_Vel'] + row['insar_Vel_std']],
            y=[row['LOS_Vel'], row['LOS_Vel']],
            region=[min_val, max_val, min_val, max_val],
            projection="X5c/5c",
            pen="1p,gray"
        )
    fig.plot(y=gps_169_ori['LOS_Vel'], x=gps_169_ori['insar_Vel'], style="c.15c", fill="grey", region=[min_val, max_val, min_val, max_val], projection="X5c/5c")
    fig.plot(y=gps_169['LOS_Vel'], x=gps_169['insar_Vel'], style="c.15c", fill=gps_169['dist2ref'], cmap=True, region=[min_val, max_val, min_val, max_val], projection="X5c/5c")
    #fig.text(text=gps_169["StaID"], y=gps_169['LOS_Vel'], x=gps_169['insar_Vel'], region=[min_val, max_val, min_val, max_val], projection="X5c/5c")
    fig.plot(y=[min_val, max_val], x=[min_val, max_val], region=[min_val, max_val, min_val, max_val], projection="X5c/5c")
    fig.plot(y=unr_intercept_169+unr_slope_169*xseq, x=xseq, region=[min_val, max_val, min_val, max_val], projection="X5c/5c", pen='dash')
    fig.text(text='RMSE = {:.1f} mm/yr'.format(unr_rmse_169), position="TL", offset="0.5c/-0.2c", region=[min_val, max_val, min_val, max_val], projection="X5c/5c")
    fig.text(text='R² = {:.2f}'.format(unr_r2_169), position="TL", offset="0.5c/-0.6c", region=[min_val, max_val, min_val, max_val], projection="X5c/5c")
    fig.text(text='{} Shift'.format(shift_169), position="TL", offset="0.5c/-1.0c", region=[min_val, max_val, min_val, max_val], projection="X5c/5c")
    
    fig.text(text='RMSE = {:.1f} mm/yr'.format(unr_rmse_169_ori), position="BR", offset="-0.5c/1.0c", region=[min_val, max_val, min_val, max_val], projection="X5c/5c")
    fig.text(text='R² = {:.2f}'.format(unr_r2_169_ori), position="BR", offset="-0.5c/0.6c", region=[min_val, max_val, min_val, max_val], projection="X5c/5c")
    fig.text(text='{} GNSS'.format(gps_169_ori.shape[0]), position="BR", offset="-0.5c/0.2c", region=[min_val, max_val, min_val, max_val], projection="X5c/5c")
    
    fig.basemap(region=[min_val, max_val, min_val, max_val], projection="X5c/5c", panel=True)
    for _, row in gps_170.iterrows():
        fig.plot(
            x=[row['insar_Vel']- row['insar_Vel_std'], row['insar_Vel'] + row['insar_Vel_std']],
            y=[row['LOS_Vel'], row['LOS_Vel']],
            region=[min_val, max_val, min_val, max_val],
            projection="X5c/5c",
            pen="1p,gray"
        )
    fig.plot(y=gps_170_ori['LOS_Vel'], x=gps_170_ori['insar_Vel'], style="c.15c", fill="grey", region=[min_val, max_val, min_val, max_val], projection="X5c/5c")
    fig.plot(y=gps_170['LOS_Vel'], x=gps_170['insar_Vel'], style="c.15c", fill=gps_170['dist2ref'], cmap=True, region=[min_val, max_val, min_val, max_val], projection="X5c/5c")
    fig.plot(y=[min_val, max_val], x=[min_val, max_val], region=[min_val, max_val, min_val, max_val], projection="X5c/5c")
    #fig.text(text=gps_170["StaID"], y=gps_170['LOS_Vel'], x=gps_170['insar_Vel'], region=[min_val, max_val, min_val, max_val], projection="X5c/5c")
    fig.plot(y=unr_intercept_170+unr_slope_170*xseq, x=xseq, region=[min_val, max_val, min_val, max_val], projection="X5c/5c", pen='dash')
    fig.text(text='RMSE = {:.1f} mm/yr'.format(unr_rmse_170), position="TL", offset="0.5c/-0.2c", region=[min_val, max_val, min_val, max_val], projection="X5c/5c")
    fig.text(text='R² = {:.2f}'.format(unr_r2_170), position="TL", offset="0.5c/-0.6c", region=[min_val, max_val, min_val, max_val], projection="X5c/5c")
    fig.text(text='{} Shift'.format(shift_170), position="TL", offset="0.5c/-1.0c", region=[min_val, max_val, min_val, max_val], projection="X5c/5c")
    
    fig.text(text='RMSE = {:.1f} mm/yr'.format(unr_rmse_170_ori), position="BR", offset="-0.5c/1.0c", region=[min_val, max_val, min_val, max_val], projection="X5c/5c")
    fig.text(text='R² = {:.2f}'.format(unr_r2_170_ori), position="BR", offset="-0.5c/0.6c", region=[min_val, max_val, min_val, max_val], projection="X5c/5c")
    fig.text(text='{} GNSS'.format(gps_170_ori.shape[0]), position="BR", offset="-0.5c/0.2c", region=[min_val, max_val, min_val, max_val], projection="X5c/5c")
    
    fig.basemap(region=[min_val, max_val, min_val, max_val], projection="X5c/5c", panel=True)
    for _, row in gps_068.iterrows():
        fig.plot(
            x=[row['insar_Vel']- row['insar_Vel_std'], row['insar_Vel'] + row['insar_Vel_std']],
            y=[row['LOS_Vel'], row['LOS_Vel']],
            region=[min_val, max_val, min_val, max_val],
            projection="X5c/5c",
            pen="1p,gray"
        )
    fig.plot(y=gps_068_ori['LOS_Vel'], x=gps_068_ori['insar_Vel'], style="c.15c", fill="grey", region=[min_val, max_val, min_val, max_val], projection="X5c/5c")
    fig.plot(y=gps_068['LOS_Vel'], x=gps_068['insar_Vel'], style="c.15c", fill=gps_068['dist2ref'], cmap=True, region=[min_val, max_val, min_val, max_val], projection="X5c/5c")
    #fig.text(text=gps_068["StaID"], y=gps_068['LOS_Vel'], x=gps_068['insar_Vel'], offset="0/0.2c", region=[min_val, max_val, min_val, max_val], projection="X5c/5c")
    fig.plot(y=[min_val, max_val], x=[min_val, max_val], region=[min_val, max_val, min_val, max_val], projection="X5c/5c")
    fig.plot(y=unr_intercept_068+unr_slope_068*xseq, x=xseq, region=[min_val, max_val, min_val, max_val], projection="X5c/5c", pen='dash')
    fig.text(text='RMSE = {:.1f} mm/yr'.format(unr_rmse_068), position="TL", offset="0.5c/-0.2c", region=[min_val, max_val, min_val, max_val], projection="X5c/5c")
    fig.text(text='R² = {:.2f}'.format(unr_r2_068), position="TL", offset="0.5c/-0.6c", region=[min_val, max_val, min_val, max_val], projection="X5c/5c")
    fig.text(text='{} Shift'.format(shift_068), position="TL", offset="0.5c/-1.0c", region=[min_val, max_val, min_val, max_val], projection="X5c/5c")
    
    fig.text(text='RMSE = {:.1f} mm/yr'.format(unr_rmse_068_ori), position="BR", offset="-0.5c/1.0c", region=[min_val, max_val, min_val, max_val], projection="X5c/5c")
    fig.text(text='R² = {:.2f}'.format(unr_r2_068_ori), position="BR", offset="-0.5c/0.6c", region=[min_val, max_val, min_val, max_val], projection="X5c/5c")
    fig.text(text='{} GNSS'.format(gps_068_ori.shape[0]), position="BR", offset="-0.5c/0.2c", region=[min_val, max_val, min_val, max_val], projection="X5c/5c")
    
    with pygmt.config(
        FONT_ANNOT_PRIMARY="18p,black", 
        FONT_ANNOT_SECONDARY="18p,black",
        FONT_LABEL="18p,black",
        ):
        fig.colorbar(position="JMR+o0.7c/0c+w4.0c/0.4c", frame=["xa+lDistance (km)"])

fig.show()  

fig.shift_origin(yshift="-4.2c")

with fig.subplot(nrows=1, ncols=3, figsize=("15c", "3.c"), autolabel="g)", sharex="b", sharey="l",
                 frame=["xaf+lResidual (mm/yr)", "ya+lCounts", "WSrt"], margins=["0.2c", "0.2c"]):
    
    pygmt.config(FORMAT_GEO_MAP="ddd.xx", MAP_FRAME_TYPE="plain", FONT=10, FONT_TITLE=11 )
    
    fig.histogram(data=gps_169['residual'], frame=["WStr", "x+lResidual (mm/yr)", "ya+lCounts"], series=0.5, fill="dodgerblue", pen="1p", histtype=0, region=[-15, 15, 0, 25], projection="X5c/3c", panel=True)
    fig.histogram(data=gps_169_ori['residual'], frame=["WStr", "x+lResidual (mm/yr)", "ya+lCounts"], series=0.5, fill="grey", transparency=50, pen="1p", histtype=0, region=[-15, 15, 0, 25], projection="X5c/3c")
    #fig.histogram(data=gps_169_common['residual'], frame=["WStr", "x+lResidual (mm/yr)", "ya+lCounts"], series=0.5, fill="red", transparency=50, pen="1p", histtype=0, region=[-15, 15, 0, 25], projection="X5c/3c")
    fig.text(text='{:.1f}% <2 mm/yr'.format(res_per_169), position="TR", offset="-0.2c/-0.2c", region=[-15, 15, 0, 25], projection="X5c/3c")
    #fig.plot(y=[0,20], x=[2,2], region=[-15, 15, 0, 25], projection="X5c/3c", pen='0.8p,dash')
    #fig.plot(y=[0,20], x=[-2,-2], region=[-15, 15, 0, 25], projection="X5c/3c", pen='0.8p,dash')
    
    fig.histogram(data=gps_170['residual'], frame=["wStr", "x+lResidual (mm/yr)", "ya+lCounts"], series=0.5, fill="dodgerblue", 
                  pen="1p", histtype=0, region=[-15, 15, 0, 25], projection="X5c/3c",  panel=True)
    fig.histogram(data=gps_170_ori['residual'], frame=["wStr", "x+lResidual (mm/yr)", "ya+lCounts"], series=0.5, fill="grey", transparency=50, pen="1p", histtype=0, region=[-15, 15, 0, 25], projection="X5c/3c")
    #fig.histogram(data=gps_170_common['residual'], frame=["wStr", "x+lResidual (mm/yr)", "ya+lCounts"], series=0.5, fill="red", transparency=50, pen="1p", histtype=0, region=[-15, 15, 0, 25], projection="X5c/3c")
    fig.text(text='{:.1f}% <2 mm/yr'.format(res_per_170), position="TR", offset="-0.2c/-0.2c", region=[-15, 15, 0, 25], projection="X5c/3c")
    #fig.plot(y=[0,20], x=[2,2], region=[-15, 15, 0, 25], projection="X5c/3c", pen='0.8p,dash')
    #fig.plot(y=[0,20], x=[-2,-2], region=[-15, 15, 0, 25], projection="X5c/3c", pen='0.8p,dash')
    
    fig.histogram(data=gps_068['residual'], frame=["wStr", "x+lResidual (mm/yr)", "ya+lCounts"], series=0.5, fill="dodgerblue", 
                  pen="1p", histtype=0, region=[-15, 15, 0, 25], projection="X5c/3c",  panel=True)
    fig.histogram(data=gps_068_ori['residual'], frame=["wStr", "x+lResidual (mm/yr)", "ya+lCounts"], series=0.5, fill="dodgerblue", transparency=50, pen="1p", histtype=0, region=[-15, 15, 0, 25], projection="X5c/3c")
    #fig.histogram(data=gps_068_common['residual'], frame=["wStr", "x+lResidual (mm/yr)", "ya+lCounts"], series=0.5, fill="red", transparency=50, pen="1p", histtype=0, region=[-15, 15, 0, 25], projection="X5c/3c")
    fig.text(text='{:.1f}% <2 mm/yr'.format(res_per_068), position="TR", offset="-0.2c/-0.2c", region=[-15, 15, 0, 25], projection="X5c/3c")
    #fig.plot(y=[0,20], x=[2,2], region=[-15, 15, 0, 25], projection="X5c/3c", pen='0.8p,dash')
    #fig.plot(y=[0,20], x=[-2,-2], region=[-15, 15, 0, 25], projection="X5c/3c", pen='0.8p,dash')

#####################
# Map View
#####################    
fig.shift_origin(yshift="10.2c")

min_lon=-124.63
max_lon=-119.22
min_lat=36.17
max_lat=42.41

fig_region="%s/%s/%s/%s" % (min_lon, max_lon, min_lat, max_lat)
size = "M8.125c"
size = "M5c"

ref_lat = gps_170[gps_170.StaID.str.contains(ref_station, case=False)]['Lat'].iloc[0]
ref_lon = gps_170[gps_170.StaID.str.contains(ref_station, case=False)]['Lon'].iloc[0]

grid = pygmt.datasets.load_earth_relief(region=fig_region, resolution="15s")
dgrid = pygmt.grdgradient(grid=grid, radiance=[270, 30], region=fig_region)

# # Begin plot
pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain", FONT=10, FONT_TITLE=11,  MAP_TITLE_OFFSET= "-7p"  )

with fig.subplot(nrows=1, ncols=3, figsize=("15c", "7.4c"),autolabel="a)", sharex="b", sharey="l",
                  frame=["WSrt", "xa", "ya"], margins=["0.2c", "0.2c"]):
    
    #####################
    # Track 169
    ##################### 

    fig.basemap(region=[fig_region],projection= size, panel=True)
    fig.grdimage(grid=grid, projection=size, frame=["lbrt", "xa", "ya"], cmap='wiki-france.cpt', shading=dgrid, region=fig_region, transparency=60)
    
    pygmt.makecpt(cmap="vik", series=[-0.025, 0.025])
    fig.grdimage(grid=grid_169, region=[fig_region], projection= size, cmap=True, nan_transparent=True)
    
    # Plot Faults
    for fault_file in common_paths["fault_files"]:
        fig.plot(data=fault_file, pen="0.5p,black", transparency=50, projection= size, region=[fig_region])
    
    fig.coast(shorelines=True, region=[fig_region],projection= size, frame = ["+tDescending 169"])
    pygmt.makecpt(cmap="vik", series=[-25, 25])
    fig.plot(y=gps_169["Lat"], x=gps_169["Lon"], style="c.1c", fill=gps_169['LOS_Vel'], cmap=True, pen="0.5p,black", region=[fig_region],projection= size)
    fig.plot(y=ref_lat, x=ref_lon, style="s.15c", fill="black", pen="0.8p,black", region=[fig_region],projection= size)
    
    df = pd.DataFrame(
        data={
            "x": [-123.9,-123.9 ],
            "y": [37.75,37.75],
            "east_velocity": [-0.173,-0.9848/3],
            "north_velocity": [-0.9848, 0.173/3],
            "east_sigma": [0,0],
            "north_sigma": [0,0],
            "correlation_EN": [0,0],
            "SITE": ["",""],
            })
    fig.velo(data=df, pen="0.8p,black", line=True, spec="e1.5/1/1", vector="0.5c+p2p+e+gblack",region=fig_region, projection=size,)

    #####################
    # Track 170
    ##################### 
    
    fig.basemap(region=[fig_region],projection= size, panel=True)
    fig.grdimage(grid=grid, projection=size, frame=["lbrt", "xa", "ya"], cmap='wiki-france.cpt', shading=dgrid, region=fig_region, transparency=60)
    pygmt.makecpt(cmap="vik", series=[-0.025, 0.025])
    fig.grdimage(grid=grid_170 , region=[fig_region],projection= size, cmap=True, nan_transparent=True)
    
    # Plot Faults
    for fault_file in common_paths["fault_files"]:
        fig.plot(data=fault_file, pen="0.5p,black", transparency=50, projection= size, region=[fig_region])
    
    fig.coast(shorelines=True, region=[fig_region],projection= size, frame = ["+tDescending 170"])
    pygmt.makecpt(cmap="vik", series=[-25, 25])
    fig.plot(y=gps_170["Lat"], x=gps_170["Lon"], style="c.1c", fill=gps_170['LOS_Vel'], cmap=True, pen="0.5p,black", region=[fig_region],projection= size)
    fig.plot(y=ref_lat, x=ref_lon, style="s.15c", fill="black", pen="0.8p,black", region=[fig_region],projection= size)
    
    df = pd.DataFrame(
        data={
            "x": [-123.9,-123.9 ],
            "y": [37.75,37.75],
            "east_velocity": [-0.173,-0.9848/3],
            "north_velocity": [-0.9848, 0.173/3],
            "east_sigma": [0,0],
            "north_sigma": [0,0],
            "correlation_EN": [0,0],
            "SITE": ["",""],
            })
    fig.velo(data=df, pen="0.8p,black", line=True, spec="e1.5/1/1", vector="0.5c+p2p+e+gblack",region=fig_region, projection=size,)

    #####################
    # Track 068
    ##################### 
    
    fig.basemap(region=[fig_region],projection= size, panel=True)
    fig.grdimage(grid=grid, projection=size, frame=["lbrt", "xa", "ya"], cmap='wiki-france.cpt', shading=dgrid, region=fig_region, transparency=60)
    
    pygmt.makecpt(cmap="vik", series=[-0.025, 0.025])
    fig.grdimage(grid=grid_068 , region=[fig_region],projection= size, cmap=True, nan_transparent=True)
    
    # Plot Faults
    for fault_file in common_paths["fault_files"]:
        fig.plot(data=fault_file, pen="0.5p,black", transparency=50, projection= size, region=[fig_region])
    
    fig.coast(shorelines=True, region=[fig_region],projection= size, frame = ["+tAscending 068"])
    pygmt.makecpt(cmap="vik", series=[-25, 25])
    fig.plot(y=gps_068["Lat"], x=gps_068["Lon"], style="c.1c", fill=gps_068['LOS_Vel'], cmap=True, pen="0.5p,black", region=[fig_region],projection= size)
    fig.plot(y=ref_lat, x=ref_lon, style="s.15c", fill="black", pen="0.8p,black", region=[fig_region],projection= size)
    
    df = pd.DataFrame(
        data={
            "x": [-124.0,-124.0 ],
            "y": [36.5, 36.5],
            "east_velocity": [-0.173, 0.9848/3],
            "north_velocity": [0.9848, 0.173/3],
            "east_sigma": [0,0],
            "north_sigma": [0,0],
            "correlation_EN": [0,0],
            "SITE": ["",""],
            })
    fig.velo(data=df, pen="0.8p,black", line=True, spec="e1.5/1/1", vector="0.5c+p2p+e+gblack",region=fig_region, projection=size,)
    
    pygmt.makecpt(cmap="vik", series=[-25, 25])
    with pygmt.config(
        FONT_ANNOT_PRIMARY="18p,black", 
        FONT_ANNOT_SECONDARY="18p,black",
        FONT_LABEL="18p,black",
        ):
        fig.colorbar(position="JMR+o0.35c/0c+w4.0c/0.4c", frame=["xa+lVelocity (mm/yr)"], projection = size)
        
fig.savefig(common_paths["fig_dir"]+f'Fig_4_{ref_station}_InSAR_GNSS_Map_dist{dist}_fullRes_geo_velocity_msk_June30.png', transparent=False, crop=True, anti_alias=True, show=False)
#fig.savefig(fig_dir+f'Fig_4_{ref_station}_InSAR_GNSS_Map_dist{dist}_fullRes_StaticShift_June30.pdf', transparent=False, crop=True, anti_alias=True, show=False)
fig.show()   

# min_lon=-124.63
# max_lon=-119.22
# min_lat=36.17
# max_lat=42.41

# fig_region="%s/%s/%s/%s" % (min_lon, max_lon, min_lat, max_lat)
# size = "M5c"

# #grid = pygmt.datasets.load_earth_relief(region=fig_region, resolution="15s")
# #dgrid = pygmt.grdgradient(grid=grid, radiance=[270, 30], region=fig_region)


# fig = pygmt.Figure()
# # Begin plot
# pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain", FONT=10, FONT_TITLE=11,  MAP_TITLE_OFFSET= "-7p"  )

# with fig.subplot(nrows=1, ncols=3, figsize=("15c", "7.5c"), autolabel=True,sharex="b", sharey="l",
#                   frame=["WSrt", "xa", "ya"], margins=["0.2c", "0.2c"]):

#     fig.basemap(region=[fig_region],projection= size, panel=True)
#     pygmt.makecpt(cmap="vik", series=[-0.025, 0.025])
#     fig.grdimage(grid=paths_169["geo"]["vel_grd"], region=[fig_region],projection= size, cmap=True, nan_transparent=True)
#     # Plot Faults
#     for fault_file in common_paths["fault_files"]:
#         fig.plot(data=fault_file, pen="0.5p,black", transparency=50, projection= size, region=[fig_region])

#     fig.coast(shorelines=True, region=[fig_region],projection= size, frame = ["+tDescending 169"])
#     pygmt.makecpt(cmap="bam", series=[-5, 5, 1])
#     fig.plot(y=gps_169["Lat"], x=gps_169["Lon"], style="c.12c", fill=gps_169['residual'], cmap=True, pen="0.3p,black", region=[fig_region],projection= size)
#     fig.plot(y=ref_lat, x=ref_lon, style="s.15c", fill="black", pen="0.8p,black", region=[fig_region],projection= size)
    
#     fig.basemap(region=[fig_region],projection= size, panel=True)
#     pygmt.makecpt(cmap="vik", series=[-0.025, 0.025])
#     fig.grdimage(grid=paths_170["geo"]["vel_grd"], region=[fig_region],projection= size, cmap=True, nan_transparent=True)
#     # Plot Faults
#     for fault_file in common_paths["fault_files"]:
#         fig.plot(data=fault_file, pen="0.5p,black", transparency=50, projection= size, region=[fig_region])

#     fig.coast(shorelines=True, region=[fig_region],projection= size, frame = ["+tDescending 170"])
#     pygmt.makecpt(cmap="bam", series=[-5, 5, 1])
#     fig.plot(y=gps_170["Lat"], x=gps_170["Lon"], style="c.12c", fill=gps_170['residual'], cmap=True, pen="0.3p,black", region=[fig_region],projection= size)
#     fig.plot(y=ref_lat, x=ref_lon, style="s.15c", fill="black", pen="0.8p,black", region=[fig_region],projection= size)
     
#     fig.basemap(region=[fig_region],projection= size, panel=True)
#     pygmt.makecpt(cmap="vik", series=[-25, 25])
#     pygmt.makecpt(cmap="vik", series=[-0.025, 0.025])
#     fig.grdimage(grid=paths_068["geo"]["vel_grd"], region=[fig_region],projection= size, cmap=True, nan_transparent=True)
#     # Plot Faults
#     for fault_file in common_paths["fault_files"]:
#         fig.plot(data=fault_file, pen="0.5p,black", transparency=50, projection= size, region=[fig_region])

#     fig.coast(shorelines=True, region=[fig_region],projection= size, frame = ["+tAscending 068"])
#     pygmt.makecpt(cmap="bam", series=[-5, 5, 1])
#     fig.plot(y=gps_068["Lat"], x=gps_068["Lon"], style="c.12c", fill=gps_068['residual'], cmap=True, pen="0.3p,black", region=[fig_region],projection= size)
#     fig.plot(y=ref_lat, x=ref_lon, style="s.15c", fill="black", pen="0.8p,black", region=[fig_region],projection= size)
    
#     with pygmt.config(
#         FONT_ANNOT_PRIMARY="18p,black", 
#         FONT_ANNOT_SECONDARY="18p,black",
#         FONT_LABEL="18p,black",
#         ):
#         fig.colorbar(position="JMR+o0.35c/0c+w4.0c/0.4c", frame=["xa+lResidual (mm/yr)"], projection = size)
    
# fig.savefig(fig_dir+f'Fig_4b_{ref_station}_InSAR_GNSS_Map_Residuals_dist{dist}_latstep{lat_step}_lonstep{lon_step}_June20.png', transparent=False, crop=True, anti_alias=True, show=False)
# fig.savefig(fig_dir+f'Fig_4b_{ref_station}_InSAR_GNSS_Map_Residuals_dist{dist}_latstep{lat_step}_lonstep{lon_step}_June20.pdf', transparent=False, crop=True, anti_alias=True, show=False)
# fig.show() 