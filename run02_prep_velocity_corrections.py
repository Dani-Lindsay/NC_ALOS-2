

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 13:40:56 2025

Run in environment with MintPy installed. 

This script takes the MintPy output and does the following:
    1) Calculates velocity field for each time series step
    2) Geocodes all tracks to the same lat-lon increment and bbox
    3) Applies plate motion correction
    
    # --- Plate Motion ---
    run_command(["plate_motion.py", "-g", geo["geo_geometryRadar"],
                 "-v", geo["geo_velocity_SET_ERA5_demErr"], "--plate", "NorthAmerica"])
    
    4) Differences each velocity field for plotting
    5) Masks all velocity fields 
    6) Provides downsampled versions for plotting as well. 

@author: daniellelindsay
"""

#!/usr/bin/env python3
import os
import subprocess
from NC_ALOS2_filepaths import (common_paths, paths_068, paths_169, paths_170, paths_170_5_28, paths_115)
import insar_utils as utils

# ------------------------
# Parameters
# ------------------------

w = common_paths['bbox']['w']
e = common_paths['bbox']['e']
s = common_paths['bbox']['s']
n = common_paths['bbox']['n']

lat_step = common_paths["lat_step"]
lon_step = common_paths["lon_step"]

# Path to geocode.py executable (adjust if needed)
geocode_py = "/Users/daniellelindsay/miniconda3/envs/MintPy_24_2/bin/geocode.py"

def run_command(cmd):
    """Helper function to run a command and print it."""
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def process_track(track):
    """
    Given a track dictionary with keys "CASR" and "geo", run the processing commands:
    timeseries-to-velocity conversion, geocoding, plate motion, diff, and mask.
    """
    casr = track["CASR"]
    geo = track["geo"]
    grd = track["grd"]
    # Determine the geo output directory from one of the geo files (assumed to be in the same folder)
    geo_dir = os.path.dirname(geo["geo_geometryRadar"])

    # --- Timeseries to Velocity Conversion ---
    run_command(["timeseries2velocity.py", casr["timeseries"], "-o", casr["velocity"]])
    run_command(["timeseries2velocity.py", casr["timeseries_SET"], "-o", casr["velocity_SET"]])
    run_command(["timeseries2velocity.py", casr["timeseries_SET_ERA5"], "-o", casr["velocity_SET_ERA5"]])
    run_command(["timeseries2velocity.py", casr["timeseries_SET_ERA5_demErr"], "-o", casr["velocity_SET_ERA5_demErr"]])

    # --- Geocoding Velocity Files ---
    run_command([geocode_py, casr["velocity"], "-l", geo["geometryRadar"],
                 "--lalo", lat_step, lon_step,
                 "--bbox", s, n, w, e,
                 "--outdir", geo_dir])
    run_command([geocode_py, casr["velocity_SET"], "-l", geo["geometryRadar"],
                 "--lalo", lat_step, lon_step,
                 "--bbox", s, n, w, e,
                 "--outdir", geo_dir])
    run_command([geocode_py, casr["velocity_SET_ERA5"], "-l", geo["geometryRadar"],
                 "--lalo", lat_step, lon_step,
                 "--bbox", s, n, w, e,
                 "--outdir", geo_dir])
    run_command([geocode_py, casr["velocity_SET_ERA5_demErr"], "-l", geo["geometryRadar"],
                 "--lalo", lat_step, lon_step,
                 "--bbox", s, n, w, e,
                 "--outdir", geo_dir])

    # --- Additional Geocoding Commands ---
    run_command([geocode_py, geo["geometryRadar"], "incidenceAngle", "-l", geo["geometryRadar"],
                 "--lalo", lat_step, lon_step,
                 "--bbox", s, n, w, e,
                 "--outdir", geo_dir])
    run_command([geocode_py, casr["maskTempCoh"], "incidenceAngle", "-l", geo["geometryRadar"],
                 "--lalo", lat_step, lon_step,
                 "--bbox", s, n, w, e,
                 "--outdir", geo_dir])
    run_command([geocode_py, casr["waterMask"], "incidenceAngle", "-l", geo["geometryRadar"],
                 "--lalo", lat_step, lon_step,
                 "--bbox", s, n, w, e,
                 "--outdir", geo_dir])
    run_command([geocode_py, casr["velocityERA5"], "-l", geo["geometryRadar"],
                 "--lalo", lat_step, lon_step,
                 "--bbox", s, n, w, e,
                 "--outdir", geo_dir])
    run_command([geocode_py, casr["demErr"], "-l", geo["geometryRadar"],
                 "--lalo", lat_step, lon_step,
                 "--bbox", s, n, w, e,
                 "--outdir", geo_dir])
    
    # --- Plate Motion ---
    run_command(["plate_motion.py", "-g", geo["geo_geometryRadar"],
                 "-v", geo["geo_velocity_SET_ERA5_demErr"], "--plate", "NorthAmerica"])

    # --- Diff Commands ---
    run_command(["diff.py", geo["geo_velocity"], geo["geo_velocity_SET"], "-o", geo["diff_SET"]])
    run_command(["diff.py", geo["geo_velocity_SET"], geo["geo_velocity_SET_ERA5"], "-o", geo["diff_ERA5"]])
    run_command(["diff.py", geo["geo_velocity_SET_ERA5"], geo["geo_velocity_SET_ERA5_demErr"], "-o", geo["diff_demErr"]])
    run_command(["diff.py", geo["geo_velocity_SET_ERA5_demErr"], geo["geo_velocity_SET_ERA5_demErr_ITRF14"], "-o", geo["diff_ITRF14"]])

    # --- Mask Commands ---
    run_command(["mask.py", geo["geo_velocity"], "-m", geo["geo_maskTempCoh"], "-o", geo["geo_velocity_msk"]])
    run_command(["mask.py", geo["geo_velocity_SET"], "-m", geo["geo_maskTempCoh"], "-o", geo["geo_velocity_SET_msk"]])
    run_command(["mask.py", geo["geo_velocity_SET_ERA5"], "-m", geo["geo_maskTempCoh"], "-o", geo["geo_velocity_SET_ERA5_msk"]])
    run_command(["mask.py", geo["geo_velocity_SET_ERA5_demErr"], "-m", geo["geo_maskTempCoh"], "-o", geo["geo_velocity_SET_ERA5_demErr_msk"]])
    run_command(["mask.py", geo["geo_velocity_SET_ERA5_demErr_ITRF14"], "-m", geo["geo_maskTempCoh"], "-o", geo["geo_velocity_SET_ERA5_demErr_ITRF14_msk"]])
    run_command(["mask.py", geo["diff_SET"], "-m", geo["geo_waterMask"], "-o", geo["diff_SET"]])
    run_command(["mask.py", geo["diff_ERA5"], "-m", geo["geo_waterMask"], "-o", geo["diff_ERA5"]])
    run_command(["mask.py", geo["diff_demErr"], "-m", geo["geo_waterMask"], "-o", geo["diff_demErr"]])
    run_command(["mask.py", geo["diff_ITRF14"], "-m", geo["geo_waterMask"], "-o", geo["diff_ITRF14"]])    
    
    # --- Save GMT Commands ---
    run_command(["save_gmt.py", geo["diff_SET"], "-o", grd["diff_SET"]])
    run_command(["save_gmt.py", geo["diff_ERA5"], "-o", grd["diff_ERA5"]])
    run_command(["save_gmt.py", geo["diff_demErr"], "-o", grd["diff_demErr"]])
    run_command(["save_gmt.py", geo["diff_ITRF14"], "-o", grd["diff_ITRF14"]])
    run_command(["save_gmt.py", geo["geo_velocity_msk"], "-o", grd["geo_velocity_msk"]])
    run_command(["save_gmt.py", geo["geo_velocity_SET_msk"], "-o", grd["geo_velocity_SET_msk"]])
    run_command(["save_gmt.py", geo["geo_velocity_SET_ERA5_msk"], "-o", grd["geo_velocity_SET_ERA5_msk"]])
    run_command(["save_gmt.py", geo["geo_velocity_SET_ERA5_demErr_msk"], "-o", grd["geo_velocity_SET_ERA5_demErr_msk"]])
    run_command(["save_gmt.py", geo["geo_velocity_SET_ERA5_demErr_ITRF14_msk"], "-o", grd["geo_velocity_SET_ERA5_demErr_ITRF14_msk"]])

#if __name__ == "__main__":
    # Process each track individually. 
    #for track in (paths_068, paths_169, paths_170):
    #    print("\nProcessing track:")
    # process_track(paths_170)
    # process_track(paths_169)
    # process_track(paths_068)
    
# Mask Central Valley Examples
temp_coh = str(0.7)  
run_command(["mask.py",     paths_170_5_28["CentralValley"]["geo_velocity"],    "-m", paths_170_5_28["CentralValley"]["geo_TempCoh"], "--vmin", temp_coh, "-o", paths_170_5_28["CentralValley"]["geo_velocity_msk"]])
run_command(["mask.py",     paths_170_5_28["CentralValley"]["geo_timeseries"],  "-m", paths_170_5_28["CentralValley"]["geo_TempCoh"], "--vmin", temp_coh, "-o", paths_170_5_28["CentralValley"]["geo_timeseries_msk"]])
run_command(["save_gmt.py", paths_170_5_28["CentralValley"]["geo_velocity"],    "-o", paths_170_5_28["CentralValley"]["geo_velocity_grd"]])
run_command(["save_gmt.py", paths_170_5_28["CentralValley"]["geo_velocity_msk"],"-o", paths_170_5_28["CentralValley"]["geo_velocity_msk_grd"]])

# Mask Central Valley Examples
temp_coh = str(0.7)
run_command(["mask.py",     paths_115["CentralValley"]["geo_velocity"],     "-m", paths_115["CentralValley"]["geo_TempCoh"],"--vmin", temp_coh, "-o", paths_115["CentralValley"]["geo_velocity_msk"]])
run_command(["mask.py",     paths_115["CentralValley"]["geo_timeseries"],   "-m", paths_115["CentralValley"]["geo_TempCoh"],"--vmin", temp_coh, "-o", paths_115["CentralValley"]["geo_timeseries_msk"]])
run_command(["save_gmt.py", paths_115["CentralValley"]["geo_velocity"],     "-o", paths_115["CentralValley"]["geo_velocity_grd"]])
run_command(["save_gmt.py", paths_115["CentralValley"]["geo_velocity_msk"], "-o", paths_115["CentralValley"]["geo_velocity_msk_grd"]])

# --- Geocode Central Valley ---
# resample to same lat lon and bb
cv_s, cv_n, cv_w, cv_e = '38.5','39.8','-122.75','-121.7'

lat_step, lon_step = str(lat_step), str(lon_step)
run_command([geocode_py, paths_068["P208"]["velocity"], "-l", paths_068["P208"]["geometryRadar"], 
             "--lalo", lat_step, lon_step, "--bbox", cv_s, cv_n, cv_w, cv_e, "--outdir", paths_068["P208"]["geo"]])
run_command([geocode_py, paths_170["P208"]["velocity"], "-l", paths_170["P208"]["geometryRadar"], 
             "--lalo", lat_step, lon_step, "--bbox", cv_s, cv_n, cv_w, cv_e, "--outdir", paths_170["P208"]["geo"]])

run_command([geocode_py, paths_068["P208"]["maskTempCoh"], "-l", paths_068["P208"]["geometryRadar"], 
             "--lalo", lat_step, lon_step, "--bbox", cv_s, cv_n, cv_w, cv_e, "--outdir", paths_068["P208"]["geo"]])
run_command([geocode_py, paths_170["P208"]["maskTempCoh"], "-l", paths_170["P208"]["geometryRadar"], 
             "--lalo", lat_step, lon_step, "--bbox", cv_s, cv_n, cv_w, cv_e, "--outdir", paths_170["P208"]["geo"]])

run_command([geocode_py, paths_068["P208"]["geometryRadar"], "-l", paths_068["P208"]["geometryRadar"], 
             "--lalo", lat_step, lon_step, "--bbox", cv_s, cv_n, cv_w, cv_e, "--outdir", paths_068["P208"]["geo"]])
run_command([geocode_py, paths_170["P208"]["geometryRadar"], "-l", paths_170["P208"]["geometryRadar"], 
             "--lalo", lat_step, lon_step, "--bbox", cv_s, cv_n, cv_w, cv_e, "--outdir", paths_170["P208"]["geo"]])

run_command(["mask.py", paths_068["P208"]["geo_velocity"], "-m", paths_068["P208"]["geo_maskTempCoh"]])
run_command(["mask.py", paths_170["P208"]["geo_velocity"], "-m", paths_170["P208"]["geo_maskTempCoh"]])

run_command(["save_gmt.py", paths_068["P208"]["geo_velocity_msk"], "-o", paths_068["P208"]["vel_grd"]])
run_command(["save_gmt.py", paths_170["P208"]["geo_velocity_msk"], "-o", paths_170["P208"]["vel_grd"]])


mlv_s, mlv_n, mlv_w, mlv_e = '41.14', '41.90', '-122.5', '-121.2'

#lat_step, lon_step = str(lat_step), str(lon_step)
run_command([geocode_py, paths_068["P784"]["velocity"], "-l", paths_068["P784"]["geometryRadar"], 
             "--lalo", lat_step, lon_step, "--bbox", mlv_s, mlv_n, mlv_w, mlv_e, "--outdir", paths_068["P784"]["geo"]])
run_command([geocode_py, paths_170["P784"]["velocity"], "-l", paths_170["P784"]["geometryRadar"], 
             "--lalo", lat_step, lon_step, "--bbox", mlv_s, mlv_n, mlv_w, mlv_e, "--outdir", paths_170["P784"]["geo"]])

run_command([geocode_py, paths_068["P784"]["maskTempCoh"], "-l", paths_068["P784"]["geometryRadar"], 
             "--lalo", lat_step, lon_step, "--bbox", mlv_s, mlv_n, mlv_w, mlv_e, "--outdir", paths_068["P784"]["geo"]])
run_command([geocode_py, paths_170["P784"]["maskTempCoh"], "-l", paths_170["P784"]["geometryRadar"], 
             "--lalo", lat_step, lon_step, "--bbox", mlv_s, mlv_n, mlv_w, mlv_e, "--outdir", paths_170["P784"]["geo"]])

run_command([geocode_py, paths_068["P784"]["geometryRadar"], "-l", paths_068["P784"]["geometryRadar"], 
             "--lalo", lat_step, lon_step, "--bbox", mlv_s, mlv_n, mlv_w, mlv_e, "--outdir", paths_068["P784"]["geo"]])
run_command([geocode_py, paths_170["P784"]["geometryRadar"], "-l", paths_170["P784"]["geometryRadar"], 
             "--lalo", lat_step, lon_step, "--bbox", mlv_s, mlv_n, mlv_w, mlv_e, "--outdir", paths_170["P784"]["geo"]])

run_command(["mask.py", paths_068["P784"]["geo_velocity"], "-m", paths_068["P784"]["geo_maskTempCoh"], "-o", paths_068["P784"]["geo_velocity_msk"]])
run_command(["mask.py", paths_170["P784"]["geo_velocity"], "-m", paths_170["P784"]["geo_maskTempCoh"], "-o", paths_170["P784"]["geo_velocity_msk"]])

run_command(["save_gmt.py", paths_068["P784"]["geo_velocity_msk"], "-o", paths_068["P784"]["vel_grd"]])
run_command(["save_gmt.py", paths_170["P784"]["geo_velocity_msk"], "-o", paths_170["P784"]["vel_grd"]])

# Mask 170 for geysers
#run_command(["mask.py", paths_170["geo"]["geo_timeseries"], "-m", paths_170["geo"]["geo_maskTempCoh"], "-o", paths_170["geo"]["geo_timeseries_msk"]])

# Mask and save ls as gmt 
temp_coh = str(0.6)  
run_command(["mask.py", paths_170_5_28["EelRiver"]["geo_velocity"],   "-m", paths_170_5_28["EelRiver"]["geo_TempCoh"], "--vmin", temp_coh,"-o", paths_170_5_28["EelRiver"]["geo_velocity_msk"]])
run_command(["mask.py", paths_170_5_28["EelRiver"]["geo_timeseries"], "-m", paths_170_5_28["EelRiver"]["geo_TempCoh"], "--vmin", temp_coh,"-o", paths_170_5_28["EelRiver"]["geo_timeseries_msk"]])

run_command(["mask.py", paths_170_5_28["Graham"]["geo_velocity"],     "-m", paths_170_5_28["Graham"]["geo_TempCoh"], "--vmin", temp_coh, "-o", paths_170_5_28["Graham"]["geo_velocity_msk"]])
run_command(["mask.py", paths_170_5_28["Graham"]["geo_timeseries"],   "-m", paths_170_5_28["Graham"]["geo_TempCoh"], "--vmin", temp_coh, "-o", paths_170_5_28["Graham"]["geo_timeseries_msk"]])

run_command(["save_gmt.py", paths_170_5_28["EelRiver"]["geo_velocity_msk"], "-o", paths_170_5_28["EelRiver"]["geo_velocity_msk_grd"]])
run_command(["save_gmt.py", paths_170_5_28["Graham"]["geo_velocity_msk"],   "-o", paths_170_5_28["Graham"]["geo_velocity_msk_grd"]])
