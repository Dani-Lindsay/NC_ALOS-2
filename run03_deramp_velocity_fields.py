#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 14:42:19 2025
Deramp InSAR velocities using quadratic deramp.

***** Run in pygmt conda environment ***** 

Updated these file in file_paths_P345
# Maks 170 based on phase closure -- proxy for unwrapping errors. 
/Users/daniellelindsay/miniconda3/envs/MintPy_24_2/bin/geocode.py numTriNonzeroIntAmbiguity.h5 -l inputs/geometryRadar.h5 --lalo 0.002 0.002 --bbox 36.17 42.41 -124.63 -119.22
mask.py geo_velocity_msk.h5  -m geo_numTriNonzeroIntAmbiguity.h5 --mask-vmax 150 
mask.py geo_velocity_SET_msk.h5  -m geo_numTriNonzeroIntAmbiguity.h5 --mask-vmax 150 
mask.py geo_velocity_SET_ERA5_msk.h5  -m geo_numTriNonzeroIntAmbiguity.h5 --mask-vmax 150 
mask.py geo_velocity_SET_ERA5_demErr_msk.h5  -m geo_numTriNonzeroIntAmbiguity.h5 --mask-vmax 150 
mask.py geo_velocity_SET_ERA5_demErr_ITRF14_msk.h5  -m geo_numTriNonzeroIntAmbiguity.h5 --mask-vmax 150 
save_gmt.py geo_velocity_msk_msk.h5
save_gmt.py geo_velocity_SET_msk_msk.h5
save_gmt.py geo_velocity_SET_ERA5_msk_msk.h5
save_gmt.py geo_velocity_SET_ERA5_demErr_msk_msk.h5
save_gmt.py geo_velocity_SET_ERA5_demErr_ITRF14_msk_msk.h5

@author: daniellelindsay
"""

import insar_utils as utils
from NC_ALOS2_filepaths import (paths_gps, paths_068, paths_169, paths_170, common_paths)
import numpy as np

unit = 1000 # 1000 = mm 

# ------------------------
# Parameters
# ------------------------
#dist = 0.004    # Averaging distance in degrees (roughly 1km)
dist = common_paths["dist"]
ref_station = common_paths["ref_station"]

# Orbit dictionaries for each track
des169_dic = {"ORBIT_DIRECTION": "descending", "PATH": "169", "Frames": "2800_2850",
              "Heading": -166.92284024662985, "Platform": "ALOS-2", "Sensor": "a2"}
des170_dic = {"ORBIT_DIRECTION": "descending", "PATH": "170", "Frames": "2800_2850",
              "Heading": -166.9210958373203, "Platform": "ALOS-2", "Sensor": "a2"}
asc068_dic  = {"ORBIT_DIRECTION": "ascending",  "PATH": "068", "Frames": "0800_0750",
              "Heading": -13.078884587112812, "Platform": "ALOS-2", "Sensor": "a2"}

# ------------------------
# Set InSAR File Paths from the dictionaries
# ------------------------
# For each track we use the "geo" keys from the corresponding paths dictionary.
geo_file_169 = paths_169["geo"]["geo_geometryRadar"]
geo_file_170 = paths_170["geo"]["geo_geometryRadar"]
geo_file_068 = paths_068["geo"]["geo_geometryRadar"]

vel_file_169 = paths_169["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14_msk"]
vel_file_170 = paths_170["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14_msk"]
vel_file_068 = paths_068["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14_msk"]

itrf_enu_169 = paths_169["geo"]["ITRF_enu"]
itrf_enu_170 = paths_170["geo"]["ITRF_enu"]
itrf_enu_068 = paths_068["geo"]["ITRF_enu"]

# ------------------------
# Load GPS Data
# ------------------------
gps_169 = utils.load_UNR_gps(paths_gps["169_enu"], ref_station)
gps_170 = utils.load_UNR_gps(paths_gps["170_enu"], ref_station)
gps_068 = utils.load_UNR_gps(paths_gps["068_enu"], ref_station)

# Drop GPS with vertical velocity >-5 mm/yr 
gps_169 = gps_169[gps_169['Vu']>=-5]
gps_170 = gps_170[gps_170['Vu']>=-5]
gps_068 = gps_068[gps_068['Vu']>=-5]

# ------------------------
# Correct Plate Motion
# ------------------------
gps_169 = utils.gps_correction_plate_motion(geo_file_169, itrf_enu_169, gps_169, ref_station, unit)
gps_170 = utils.gps_correction_plate_motion(geo_file_170, itrf_enu_170, gps_170, ref_station, unit)
gps_068 = utils.gps_correction_plate_motion(geo_file_068, itrf_enu_068, gps_068, ref_station, unit)

# ------------------------
# Load InSAR Data
# ------------------------
insar_169, shape169 = utils.load_insar_vel_as_df(geo_file_169, vel_file_169, des169_dic)
insar_170, shape170 = utils.load_insar_vel_as_df(geo_file_170, vel_file_170, des170_dic)
insar_068, shape068 = utils.load_insar_vel_as_df(geo_file_068, vel_file_068, asc068_dic)

# Convert to mm for GPS comparison 
insar_169["Vel"] = insar_169["Vel"] * unit 
insar_170["Vel"] = insar_170["Vel"] * unit 
insar_068["Vel"] = insar_068["Vel"] * unit 

# ------------------------
# Process InSAR Data
# ------------------------
# Calculate average InSAR velocity for each GPS point
gps_169 = utils.calculate_average_insar_velocity(gps_169, insar_169, dist)
gps_170 = utils.calculate_average_insar_velocity(gps_170, insar_170, dist)
gps_068 = utils.calculate_average_insar_velocity(gps_068, insar_068, dist)

# Project GPS ENU velocities to InSAR LOS
gps_169 = utils.calculate_gps_los(gps_169, insar_169)
gps_170 = utils.calculate_gps_los(gps_170, insar_170)
gps_068 = utils.calculate_gps_los(gps_068, insar_068)

# Save original InSAR velocities for reference
insar_169["Vel_ori"] = insar_169["Vel"]
insar_170["Vel_ori"] = insar_170["Vel"]
insar_068["Vel_ori"] = insar_068["Vel"]

# ------------------------
# Deramp InSAR (Quadratic Ramp Removal)
# ------------------------
insar_169 = utils.apply_quadratic_deramp_2D(gps_169, insar_169)
insar_170 = utils.apply_quadratic_deramp_2D(gps_170, insar_170)
insar_068 = utils.apply_quadratic_deramp_2D(gps_068, insar_068)

#Update InSAR velocities to the deramped values
insar_169["Vel"] = insar_169["Vel_quadramp"]
insar_170["Vel"] = insar_170["Vel_quadramp"]
insar_068["Vel"] = insar_068["Vel_quadramp"]


# Velocities are scaled by 1/1000 (e.g., converting from mm/yr to m/yr for consistency).
utils.write_new_h5(insar_169["Vel"]/unit, paths_169["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14_msk"], shape169, "deramp")
utils.write_new_h5(insar_170["Vel"]/unit, paths_170["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14_msk"], shape170, "deramp")
utils.write_new_h5(insar_068["Vel"]/unit, paths_068["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14_msk"], shape068, "deramp")

# --- Prep vel files ---    
# Calculate diff.h5
utils.run_command(["diff.py", paths_169["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14_msk"], paths_169["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"], "-o", paths_169["geo"]["diff_deramp"]])
utils.run_command(["diff.py", paths_170["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14_msk"], paths_170["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"], "-o", paths_170["geo"]["diff_deramp"]])
utils.run_command(["diff.py", paths_068["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14_msk"], paths_068["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"], "-o", paths_068["geo"]["diff_deramp"]])

# Calculate diff.h5 for overlapping regions
utils.run_command(["diff.py", paths_169["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"],  paths_170["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"], "-o", paths_169["geo"]["diff_169_170"]])

# Save as gmt grd for plotting 
utils.run_command(["save_gmt.py", paths_169["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"], "-o",  paths_169["grd"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"]])
utils.run_command(["save_gmt.py", paths_170["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"], "-o",  paths_170["grd"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"]])
utils.run_command(["save_gmt.py", paths_068["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"], "-o",  paths_068["grd"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"]])

# Save as gmt grd for plotting 
utils.run_command(["save_gmt.py", paths_169["geo"]["diff_deramp"], "-o",  paths_169["grd"]["diff_deramp"]])
utils.run_command(["save_gmt.py", paths_170["geo"]["diff_deramp"], "-o",  paths_170["grd"]["diff_deramp"]])
utils.run_command(["save_gmt.py", paths_068["geo"]["diff_deramp"], "-o",  paths_068["grd"]["diff_deramp"]])

utils.run_command(["save_gmt.py", paths_169["geo"]["diff_169_170"], "-o",  paths_169["grd"]["diff_169_170"]])

# --- Convert all .grd files to mm (×1000) for each track ---
for track in (paths_169, paths_170, paths_068):
    for name, grd_path in track["grd"].items():
        if grd_path.endswith(".grd"):
            mm_path = grd_path.replace(".grd", "_mm.grd")
            utils.run_command([
                "gmt", "grdmath",
                grd_path, "1000", "MUL", "=", mm_path
            ])
