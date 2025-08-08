#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 10:34:45 2025

@author: daniellelindsay


"""

from datetime import datetime
import numpy as np
import pandas as pd               
from geopy.distance import geodesic
import geopy.distance
import h5py
from scipy import stats
from scipy.optimize import least_squares
import shutil
import pygmt
import netCDF4 as nc
import subprocess
from scipy.stats import linregress
import math

def yymmdd_to_decimal_year(date_str):
    """Convert a date in 'YYMMDD' format to a decimal year, assuming all dates are in the 2000s."""
    # Prepend '20' to the date string to make it 'YYYYMMDD'
    full_date_str = '20' + date_str
    
    date = datetime.strptime(full_date_str, '%Y%m%d')
    decimal_year = date.year + (date.timetuple().tm_yday - 1) / 365.25
    return decimal_year

def date_to_decimal_year(date_input, format_str=None):
    """
    Convert a date to a decimal year.

    :param date_input: Date input which can be a string, byte string, or datetime object.
    :param format_str: Format string for parsing the date. If None, the function will attempt to infer the format.
                       Acceptable formats include: "%Y-%m-%d" (e.g., "2021-12-31"), 
                       "%m/%d/%Y" (e.g., "12/31/2021"), and "%Y%m%d" (e.g., "20211231").
    :return: Decimal year corresponding to the input date.
    """

    # If the input is a byte string, decode it to a regular string
    if isinstance(date_input, bytes):
        date_input = date_input.decode('utf-8')

    # If the input is already a datetime object, use it directly
    if isinstance(date_input, datetime):
        date = date_input
    else:
        # Attempt to infer the format if not provided
        if format_str is None:
            if "-" in date_input:
                format_str = "%Y-%m-%d"
            elif "/" in date_input:
                format_str = "%m/%d/%Y"
            elif len(date_input) == 8:
                format_str = "%Y%m%d"
            else:
                raise ValueError("Unknown date format. Please provide a format string.")

        # Parse the date string using the provided format
        date = datetime.strptime(date_input, format_str)

    start_of_year = datetime(year=date.year, month=1, day=1)
    start_of_next_year = datetime(year=date.year+1, month=1, day=1)
    year_length = (start_of_next_year - start_of_year).total_seconds()
    year_progress = (date - start_of_year).total_seconds()
    decimal_year = date.year + year_progress / year_length
    return decimal_year


def read_baselines(file_path):
    """
    Reads a file containing secondary dates and perpendicular baselines,
    converting dates to decimal years and centering baselines on zero.

    Parameters:
    - file_path (str): Path to the file with each line containing a secondary date in YYMMDD format
                       and a perpendicular baseline in meters.

    Returns:
    - (list of float, list of float): Tuple of two lists:
        1. Secondary dates in decimal year format.
        2. Centered perpendicular baselines in meters.
    """
    # Initialize lists to hold the processed data
    secondary_date = []
    perp_base = []
    
    # Open and read the file
    with open(file_path, 'r') as file:
        next(file)  # Skip the header line
        for line in file:
            parts = line.split()
            if len(parts) >= 4:  # Ensure the line has enough parts
                date_str = parts[1]  # Secondary date is the second column
                baseline = parts[3]  # Perpendicular baseline is the fourth column
                date = datetime.strptime(date_str, "%y%m%d")
                
                # Convert to decimal year
                decimal_year = date.year + (date.timetuple().tm_yday - 1) / 365.25
                
                # Append
                secondary_date.append(decimal_year)
                perp_base.append(float(baseline))

    # Center the baseline values on zero
    mean_baseline = sum(perp_base) / len(perp_base)
    centered_perp_base = [x - mean_baseline for x in perp_base]

    return np.array(secondary_date), np.array(centered_perp_base)

def read_coherence_data(file_path):
    # Initialize the lists to hold the extracted data
    pairs = []
    mean = []
    btemp = []
    bperp = []
    
    # Open the file and process each line
    with open(file_path, 'r') as file:
        for line in file:
            # Skip lines that do not contain the data of interest
            if line.startswith("#") or line.strip() == '':
                continue
            
            # Split the line into components
            components = line.split()
            
            # Extract and append the data to the respective lists
            date_pair = components[0]
            mean_value = float(components[1])
            btemp_value = float(components[2])
            bperp_value = float(components[3])
            
            # Convert to decimal year
            first_date = datetime.strptime(date_pair[:8], '%Y%m%d')
            second_date = datetime.strptime(date_pair[9:], '%Y%m%d')
            first_date_decimal = first_date.year + (first_date.timetuple().tm_yday - 1) / 365.25
            second_date_decimal = second_date.year + (second_date.timetuple().tm_yday - 1) / 365.25
            
            pairs.append([first_date_decimal, second_date_decimal])
            mean.append(mean_value)
            btemp.append(btemp_value)
            bperp.append(bperp_value)
    
    # Convert lists to numpy arrays
    pairs = np.array(pairs)
    mean = np.array(mean)
    btemp = np.array(btemp)
    bperp = np.array(bperp)
    
    # Sort the arrays based on the mean values
    sorted_indices = np.argsort(mean)
    sorted_pairs = pairs[sorted_indices]
    sorted_mean = mean[sorted_indices]
    sorted_btemp = btemp[sorted_indices]
    sorted_bperp = bperp[sorted_indices]
    
    return sorted_pairs, sorted_mean, sorted_btemp, sorted_bperp

def calculate_pairs_bperp(dic):
    """
    Calculate the bperp values for each pair of dates in the 'pairs' key of the dictionary.
    
    Parameters:
    - dic (dict): The dictionary containing 'sec_date', 'centered_perp_base', and 'pairs'.
    
    Returns:
    - np.ndarray: An array of bperp values for each pair.
    """
    # Initialize an empty list to store bperp values for each pair
    pairs_bperp = []
    
    # Iterate over each pair
    for pair in dic["pairs"]:
        # Find the indices of the pair dates in sec_date
        indices = []
        for date in pair:
            index = np.where(dic["sec_date"] == date)[0]
            if index.size > 0:  # Date found in sec_date
                indices.append(index[0])
            else:  # If the date is the reference date (bperp = 0)
                indices.append(-1)
        
        # Use the indices to select the bperp values, default to 0 if index is -1
        pair_bperp = [dic["centered_perp_base"][i] if i != -1 else 0 for i in indices]
        
        # Add the calculated bperp values for the current pair to the list
        pairs_bperp.append(pair_bperp)
    
    # Convert the list to a NumPy array before returning
    return np.array(pairs_bperp)

def meters_to_degrees(meters, lat_deg):
    R = 6378137.0  # Earth radius in meters (WGS84)
    dlat = meters / R * (180.0 / math.pi)
    dlon = meters / (R * math.cos(math.radians(lat_deg))) * (180.0 / math.pi)
    return dlat, dlon

def run_command(cmd):
    """Helper function to run a command and print it."""
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def load_UNR_gps(gps_file, ref_station):
    # Read data from the CSV file and create the DataFrame
    gps_data = pd.read_csv(gps_file, sep=",", header=0, names=['StaID', 'Lon', 'Lat', 'Ve', 'Vn', 'Vu', 'Std_e', 'Std_n', 'Std_u', 'r_e', 'r_n', 'r_u'])
    gps_data['Lon'] = gps_data['Lon'] - 360
    # Drop rows with NaN values
    gps_data = gps_data.dropna()

    # Calculate offset based on ref_station
    offset_e = gps_data[gps_data.StaID.str.contains(ref_station, case=False)]['Ve'].values
    offset_n = gps_data[gps_data.StaID.str.contains(ref_station, case=False)]['Vn'].values
    offset_u = gps_data[gps_data.StaID.str.contains(ref_station, case=False)]['Vu'].values
    
    # Subtract offset from 'GNSS_Vel' column
    gps_data['Ve'] = gps_data['Ve'] - offset_e
    gps_data['Vn'] = gps_data['Vn'] - offset_n
    gps_data['Vu'] = gps_data['Vu'] - offset_u

    return gps_data

def load_insar_vel_as_df(geo_file, vel_file, dic):
    # previously load_insar_data
    with h5py.File(geo_file, 'a') as hf:
        # Read in incidence Angle
        inc = np.array(hf["incidenceAngle"][:])

        if dic["Sensor"] == "s1":
            # Get attributes
            x_start = hf.attrs['X_FIRST']
            y_start = hf.attrs['Y_FIRST']
            x_step = hf.attrs['X_STEP']
            y_step = hf.attrs['Y_STEP']
            length = hf.attrs['LENGTH']
            width = hf.attrs['WIDTH']
            heading = hf.attrs['HEADING']
            azimuth = (float(heading) - 90) * -1 # Convert for right looking

            # Make mesh of Eastings and Northings
            lons = np.arange(float(x_start), float(x_start) + float(x_step) * (float(width)), float(x_step))
            lats = np.arange(float(y_start), float(y_start) + float(y_step) * (float(length)), float(y_step))
            # lons = lons[:-1]

            lon, lat = np.meshgrid(lons, lats)

            # Add dataset of azimuthAngle to geometry
            az = np.full((int(length), int(width)), azimuth)

        if dic["Sensor"] == "a2":
            az = np.array(hf["azimuthAngle"][:])
            lon = np.array(hf["longitude"][:])
            lat = np.array(hf["latitude"][:])

    with h5py.File(vel_file, 'a') as hf:
        vel = np.array(hf["velocity"][:])
        vel[vel == 0] = np.nan

    # Reshape to 1D arrays
    print(f"Vel shape: {vel.shape}")
    shape_h5 = vel.shape
    length = vel.size
    az = az.reshape(length, 1)
    inc = inc.reshape(length, 1)
    lons = lon.reshape(length, 1)
    lats = lat.reshape(length, 1)
    vel = vel.reshape(length, 1)
    print(f"Vel size: {vel.size}")
    

    insar_data = pd.DataFrame(np.concatenate([lons, lats, vel, az, inc], axis=1), columns=['Lon', 'Lat', 'Vel', 'Az', 'Inc'])
    #insar_data = insar_data.dropna()

    # Drop duplicate rows based on all columns
    #insar_data = insar_data.drop_duplicates()

    return insar_data, shape_h5

def calculate_average_insar_velocity(gps_data, insar_data, dist):
    has_std = 'Std' in insar_data.columns  # Check if 'Std' column exists
    for index, row in gps_data.iterrows():
        lat_min = gps_data.at[index, 'Lat'] - dist
        lat_max = gps_data.at[index, 'Lat'] + dist
        lon_min = gps_data.at[index, 'Lon'] - dist
        lon_max = gps_data.at[index, 'Lon'] + dist
        
        insar_roi = insar_data[(insar_data.Lat > lat_min) &
                               (insar_data.Lat < lat_max) &
                               (insar_data.Lon > lon_min) &
                               (insar_data.Lon < lon_max)]
        
        if not insar_roi.empty:
            gps_data.at[index, 'insar_Vel'] = np.nanmedian(insar_roi['Vel'])
            if has_std:
                gps_data.at[index, 'insar_Vel_std'] = np.nanmedian(insar_roi['Std'])
        else:
            gps_data.at[index, 'insar_Vel'] = np.nan
            if has_std:
                gps_data.at[index, 'insar_Vel_std'] = np.nan
    
    gps_data = gps_data.dropna(subset=['insar_Vel'])
    return gps_data

def calculate_gps_los(gps_data, insar_data):

    for index, row in gps_data.iterrows(): 
        i = ((insar_data['Lon'] - row['Lon']) * (insar_data['Lat'] - row['Lat'])).abs().idxmin()                               
        az_angle = insar_data.loc[i]['Az']
        inc_angle = insar_data.loc[i]['Inc']
        
        gps_data.at[index, 'Az'] = az_angle
        gps_data.at[index, 'Inc'] = inc_angle
        
        # project ENU onto LOS
        v_los = (  row['Ve'] * np.sin(np.deg2rad(inc_angle)) * np.sin(np.deg2rad(az_angle)) * -1
                 + row['Vn'] * np.sin(np.deg2rad(inc_angle)) * np.cos(np.deg2rad(az_angle))
                 + row['Vu'] * np.cos(np.deg2rad(inc_angle)))
        gps_data.at[index, 'LOS_Vel'] = v_los

    return gps_data

def apply_quadratic_deramp_2D(gps_data, insar_data):
    """
    Applies 2D quadratic ramp fitting to minimize GPS residuals and removes the ramp from InSAR velocities.

    Parameters:
    - gps_data: DataFrame with GPS data ['Lon', 'Lat', 'insar_Vel', 'UNR_Vel'].
    - insar_data: DataFrame with InSAR data ['Lon', 'Lat', 'Vel'].

    Returns:
    - insar_data: DataFrame with columns ['quadratic_ramp', 'Vel_deramp'] after removing the quadratic ramp.
    """
    # Calculate residuals for GPS data
    gps_data['residual'] = gps_data['LOS_Vel'] - gps_data['insar_Vel']

    # Quadratic ramp model: a * Lon^2 + b * Lat^2 + c * Lon * Lat + d * Lon + e * Lat + f
    def quadratic_ramp_model(params, Lon, Lat):
        return (params[0] * Lon**2 + params[1] * Lat**2 + params[2] * Lon * Lat +
                params[3] * Lon + params[4] * Lat + params[5])

    # Residuals function for least squares fitting
    def residuals(params, Lon, Lat, residual):
        return quadratic_ramp_model(params, Lon, Lat) - residual

    # Fit quadratic ramp model to GPS residuals
    initial_guess = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    result = least_squares(residuals, initial_guess, args=(gps_data['Lon'], gps_data['Lat'], gps_data['residual']))
    a, b, c, d, e, f = result.x

    # Apply quadratic ramp to InSAR data and remove it from velocities
    insar_data['quadratic_ramp'] = (a * insar_data['Lon']**2 + b * insar_data['Lat']**2 +
                                    c * insar_data['Lon'] * insar_data['Lat'] +
                                    d * insar_data['Lon'] + e * insar_data['Lat'] + f)
    insar_data['Vel_quadramp'] = insar_data['Vel'] + insar_data['quadratic_ramp']

    return insar_data

def apply_deramp_2D(gps_data, insar_data):
    """
    Applies 2D ramp fitting to minimize GPS residuals and removes the ramp from InSAR velocities.

    Parameters:
    - gps_data: DataFrame with GPS data ['Lon', 'Lat', 'insar_Vel', 'UNR_Vel'].
    - insar_data: DataFrame with InSAR data ['Lon', 'Lat', 'Vel'].

    Returns:
    - insar_data: DataFrame with columns ['ramp', 'Vel_deramp'] after removing the ramp.
    """
    # Calculate residuals for GPS data
    gps_data['residual'] = gps_data['LOS_Vel'] - gps_data['insar_Vel']

    # 2D ramp model: a * Lon + b * Lat + c
    def ramp_model(params, Lon, Lat):
        return params[0] * Lon + params[1] * Lat + params[2]

    # Residuals function for least squares fitting
    def residuals(params, Lon, Lat, residual):
        return ramp_model(params, Lon, Lat) - residual

    # Fit ramp model to GPS residuals
    result = least_squares(residuals, [0.0, 0.0, 0.0], args=(gps_data['Lon'], gps_data['Lat'], gps_data['residual']))
    a, b, c = result.x

    # Apply ramp to InSAR data and remove it from velocities
    insar_data['ramp'] = a * insar_data['Lon'] + b * insar_data['Lat'] + c
    insar_data['Vel_2Dramp'] = insar_data['Vel'] + insar_data['ramp']

    return insar_data

def write_new_h5(df_col, vel_file_path, shape, suffix):
    """
    Write a new HDF5 file containing a dataset named 'velocity' from the 
    provided DataFrame column (df_col). The new file name is constructed by
    appending '_{suffix}.h5' to the original filename (before the .h5 extension).
    """
    outfile = vel_file_path.replace('.h5', f'_{suffix}.h5')
    
    # Option 1: Create a new file (overwrite if exists)
    vel_np = df_col.to_numpy().reshape(shape)
    with h5py.File(outfile, 'w') as hf:
        hf.create_dataset("velocity", data=vel_np)
    
    # copy metadata from the original file):
    shutil.copyfile(vel_file_path, outfile)
    with h5py.File(outfile, 'a') as hf:
        if "velocity" in hf:
            del hf["velocity"]
        hf.create_dataset("velocity", data=vel_np)
    
    return outfile  # return the new file path for later use

def load_h5_data(geo_file, vel_file, dataset):
    with h5py.File(geo_file, 'a') as hf:
        # Read in incidence Angle
        inc = np.array(hf["incidenceAngle"][:])
        az = np.array(hf["azimuthAngle"][:])
        lon = np.array(hf["longitude"][:])
        lat = np.array(hf["latitude"][:])

    with h5py.File(vel_file, 'a') as hf:
        vel = np.array(hf[dataset][:])
        vel[vel == 0] = np.nan

    # Reshape to 1D arrays
    length = vel.size
    az = az.reshape(length, 1)
    inc = inc.reshape(length, 1)
    lons = lon.reshape(length, 1)
    lats = lat.reshape(length, 1)
    vel = vel.reshape(length, 1)

    insar_data = pd.DataFrame(np.concatenate([lons, lats, vel, az, inc], axis=1), columns=['Lon', 'Lat', 'Vel', 'Az', 'Inc'])
    insar_data = insar_data.dropna()

    # Drop duplicate rows based on all columns
    #insar_data = insar_data.drop_duplicates()

    return insar_data

def calculate_rmse_r2_and_linear_fit(observed, predicted):
    rmse = np.sqrt(np.sum((observed - predicted)**2) / (observed.size - 1))
    slope, intercept, r_value, p_value, std_err = stats.linregress(observed, predicted)
    r2 = r_value ** 2
    #print(f'RMSE: {np.round(rmse,2)}, r2: {np.round(r2,2)}')
    return rmse, r2, slope, intercept

def calculate_gps_los_error(gps_data, insar_data):

    for index, row in gps_data.iterrows(): 
        i = ((insar_data['Lon'] - row['Lon']) * (insar_data['Lat'] - row['Lat'])).abs().idxmin()                               
        az_angle = insar_data.loc[i]['Az']
        inc_angle = insar_data.loc[i]['Inc']
        
        gps_data.at[index, 'Az'] = az_angle
        gps_data.at[index, 'Inc'] = inc_angle
        
        # Convert angles to radians once for efficiency
        inc_rad = np.deg2rad(inc_angle)
        az_rad = np.deg2rad(az_angle)
        
        # Calculate LOS error with proper exponentiation (**2)
        v_los_err = np.sqrt(
            (np.sin(inc_rad) * np.sin(az_rad))**2 * row['Std_e']**2 +
            (np.sin(inc_rad) * np.cos(az_rad))**2 * row['Std_n']**2 +
            (np.cos(inc_rad))**2 * row['Std_u']**2
        )
        gps_data.at[index, 'LOS_Vel_err'] = v_los_err

    return gps_data

def calc_residual_percent(gps_data, threshold):
    # Calculate the mean and standard deviation (1-sigma) of the residuals
    gps_data['residual'] = gps_data['LOS_Vel'] - gps_data['insar_Vel']
    residuals = gps_data['residual']
    mean_residual = np.mean(residuals)
    sigma_residual = np.std(residuals)
    
    # 1-sigma range around the mean
    lower_limit = mean_residual - sigma_residual
    upper_limit = mean_residual + sigma_residual
    
    print(f"Mean of residuals: {mean_residual:.2f}")
    print(f"1-Sigma (Standard Deviation): {sigma_residual:.2f}")
    print(f"1-Sigma Range: [{lower_limit:.2f}, {upper_limit:.2f}]")
    
    res_count = np.sum(np.abs(gps_data['residual']) < sigma_residual)
    total_count = len(gps_data)
    percentage = (res_count / total_count) * 100
    
    print(f"{percentage:.2f}% < 1-sigma {sigma_residual:.2f} mm/yr")
    
    res_count = np.sum(np.abs(gps_data['residual']) < threshold)
    total_count = len(gps_data)
    percentage = (res_count / total_count) * 100
    
    print(f"{percentage:.2f}% < {threshold} mm/yr")
    
    return percentage, sigma_residual

def calculate_distance(lat1, lon1, lat2, lon2):
    # Coordinates of the two points
    coords_1 = (lat1, lon1)
    coords_2 = (lat2, lon2)
    
    # Calculate the distance using geodesic function
    distance = geodesic(coords_1, coords_2).kilometers
    
    return distance

def calculate_distance_to_reference(gps_data, ref_station):
    ref_lat = gps_data[gps_data.StaID.str.contains(ref_station, case=False)]['Lat'].iloc[0]
    ref_lon = gps_data[gps_data.StaID.str.contains(ref_station, case=False)]['Lon'].iloc[0]

    for index, row in gps_data.iterrows(): 
        lat = gps_data.at[index, 'Lat']
        lon = gps_data.at[index, 'Lon']
        gps_data.at[index, 'dist2ref'] = calculate_distance(ref_lat, ref_lon, lat, lon)

    gps_data = gps_data.sort_values(by='dist2ref')
    
    return gps_data

def load_gmt_grid(filename):
    with nc.Dataset(filename) as ds:
        # Assume the grid variables are named 'lon', 'lat', and 'z'
        lon = ds.variables['x'][:]
        lat = ds.variables['y'][:]
        grid_data = ds.variables['z'][:]
    return lon, lat, grid_data

def load_insar_vel_data_as_2Darrays(geo_file, vel_file):
    with h5py.File(geo_file, 'a') as hf:
        # Read in incidence Angle
        inc = np.array(hf["incidenceAngle"][:])
        azi = np.array(hf["azimuthAngle"][:])
        lon = np.array(hf["longitude"][:])
        lat = np.array(hf["latitude"][:])
    with h5py.File(vel_file, 'a') as hf:
        vel = np.array(hf["velocity"][:])
        vel[vel == 0] = np.nan

    # m --> mm 
    vel = vel

    return lon, lat, vel, azi, inc

def load_insar_vel_ts_as_dictionary(dic):
    """
    Read InSAR geometry, velocity, and timeseries data from provided file paths.

    Parameters:
    - geo_file: Path to the file containing InSAR geometry data.
    - vel_file: Path to the file containing InSAR velocity data.
    - ts_file: Path to the file containing InSAR timeseries data.

    Returns:
    A dictionary containing:
    - 'lons': Longitudes from the geometry file.
    - 'lats': Latitudes from the geometry file.
    - 'inc': Incidence angles from the geometry file.
    - 'azi': Azimuth angles from the geometry file.
    - 'vel': Velocities from the velocity file.
    - 'ts': Timeseries data from the timeseries file.
    - 'ts_dates': Decimal years for each date in the timeseries.
    """
    print("Loading %s " % dic["Platform"])

    if dic["Platform"] == "ALOS-2":
        with h5py.File(dic["geo_file"], 'r') as hfgeo:
            print("Loading %s " % dic["geo_file"])
            
            lons = np.array(hfgeo["longitude"][:])
            lats = np.array(hfgeo["latitude"][:])
            inc = np.array(hfgeo["incidenceAngle"][:])
            azi = np.array(hfgeo["azimuthAngle"][:])
            
    if dic["Platform"] == "Sentinel-1":
        with h5py.File(dic["geo_file"], 'r') as hfgeo:
            print("Loading %s " % dic["geo_file"])
            inc = np.array(hfgeo["incidenceAngle"][:])
            # Get attributes
            x_start = hfgeo.attrs['X_FIRST']
            y_start = hfgeo.attrs['Y_FIRST']
            x_step = hfgeo.attrs['X_STEP']
            y_step = hfgeo.attrs['Y_STEP']
            length = hfgeo.attrs['LENGTH']
            width = hfgeo.attrs['WIDTH']
            heading = hfgeo.attrs['HEADING']
            azimuth = (float(heading) - 90) * -1 # Convert for right looking

            print(f'inc has shape:  {inc.shape}')
            print(f'attributes has: {length, width}')
            
            if float(length) != inc.shape[0]:
                print(f'Lats : Length = {length} not equal to inc length {inc.shape[0]}')

            if float(width) != inc.shape[1]:
                print(f'Lons: Width = {width} not equal to inc width {inc.shape[1]}')
            
            # Make mesh of Eastings and Northings using linspace to ensure correct array size
            lon = np.linspace(float(x_start), float(x_start) + float(x_step) * (float(width)-1), int(width))
            lat = np.linspace(float(y_start), float(y_start) + float(y_step) * (float(length)-1), int(length))

            lons, lats = np.meshgrid(lon, lat)
            print(f'linspace lons has shape:  {lons.shape}')
            print(f'linsapre lats has shape:  {lats.shape}')

            # Add dataset of azimuthAngle to geometry
            azi = np.full((int(length), int(width)), float(azimuth))
    
    with h5py.File(dic["vel_file"], 'r') as hfvel:
        print("Loading %s " % dic["vel_file"])
        vel = np.array(hfvel["velocity"][:])
        vel = np.array(hfvel["velocity"][:])
        vel[vel == 0] = np.nan  # Set zero values to nan

    with h5py.File(dic["ts_file"], 'r') as hfvel:
        print("Loading %s " % dic["ts_file"])
        ts = np.array(hfvel["timeseries"][:])
        ts[ts == 0] = np.nan  # Set zero values to nan
        ts_dates_bytes = np.array(hfvel["date"][:])  # Assuming dates are stored as bytes
        print(f'ts has shape:  {ts.shape}')

    # Convert byte strings of dates to decimal years
    ts_dates = [date_to_decimal_year(d.decode('utf-8')) for d in ts_dates_bytes]
    
    # Return a dictionary of the data
    return {
        'lons': lons,
        'lats': lats,
        'inc': inc,
        'azi': azi,
        'vel': vel,
        'ts': ts,
        'ts_dates': ts_dates
    }

def calculate_gps_timeseries_los(gps_ts, insar_df, track):
    """
    Projects a GPS time series onto the InSAR line-of-sight (LOS) direction.
    
    Uses the first GPS coordinate to find the nearest InSAR pixel and extracts its
    constant azimuth and incidence angles, then computes the LOS projection for the
    entire GPS DataFrame.

    Parameters:
      gps_ts : pd.DataFrame
          GPS time series with columns including 'east', 'north', 'up', 'Lat', 'Lon'.
      insar_df : pd.DataFrame
          InSAR data with columns ['Lon', 'Lat', 'Inc', 'Az'] (NaNs dropped).
      track : str
          Identifier for naming the LOS column (e.g., '170' yields 'LOS_170').

    Returns:
      pd.DataFrame: A new DataFrame with added columns 'Azi_<track>', 'Inc_<track>', and 'LOS_<track>'.
    """
    # Create a copy of the input DataFrame to avoid modifying it in-place
    new_df = gps_ts.copy()

    # Find nearest InSAR pixel using the first GPS coordinate.
    ref_lat, ref_lon = new_df.iloc[0]['Lat'], new_df.iloc[0]['Lon']
    distances = np.sqrt((insar_df['Lon'] - ref_lon)**2 + (insar_df['Lat'] - ref_lat)**2)
    nearest = distances.idxmin()
    az_angle, inc_angle = insar_df.loc[nearest, 'Az'], insar_df.loc[nearest, 'Inc']
    
    # Add constant azimuth and incidence to the new DataFrame.
    new_df[f'Azi_{track}'] = az_angle
    new_df[f'Inc_{track}'] = inc_angle
    
    # Compute LOS projection.
    az_rad, inc_rad = np.deg2rad(az_angle), np.deg2rad(inc_angle)
    new_df[f'LOS_{track}'] = (
        - new_df['east'] * np.sin(inc_rad) * np.sin(az_rad) +
          new_df['north'] * np.sin(inc_rad) * np.cos(az_rad) +
          new_df['up']    * np.cos(inc_rad)
    )
    return new_df

def resample_gps_to_insar_dates(gps_df, insar_dates, window_days=6):
    """
    Resamples GPS data to InSAR dates by averaging (or otherwise aggregating)
    measurements within ±window_days (converted to decimal years) around each InSAR date.
    All original columns in gps_df are preserved using an appropriate aggregation:
      - For numeric columns:
          * For displacement columns ('east', 'north', 'up'), the mean is computed.
          * For error columns ('sig_e', 'sig_n', 'sig_u'), errors are propagated
            via sqrt(sum(error^2))/n.
          * Other numeric columns are averaged.
      - For non-numeric columns, the first value in the window is taken.
    
    Parameters:
      gps_df : pd.DataFrame
          GPS data with a decimal year column ('yyyy') and various columns.
      insar_dates : list or array-like
          List of InSAR dates (in decimal years) at which to resample the GPS data.
      window_days : float, optional
          The half-window in days (default is 6) used for matching GPS dates.
    
    Returns:
      pd.DataFrame: A new DataFrame with one row per InSAR date including all original columns,
                    plus a new column 'ts_date' holding the InSAR date.
    """
    import numpy as np
    import pandas as pd

    window = window_days / 365.0
    resampled = []

    # Loop over each InSAR date
    for d in insar_dates:
        mask = np.abs(gps_df['yyyy'] - d) <= window
        subset = gps_df[mask]
        # Initialize a row dictionary with the InSAR date
        row = {'ts_date': d}
        
        if subset.empty:
            # No data within the window: assign NaN for numeric cols and None for non-numeric
            for col in gps_df.columns:
                if np.issubdtype(gps_df[col].dtype, np.number):
                    row[col] = np.nan
                else:
                    row[col] = None
        else:
            n = len(subset)
            for col in gps_df.columns:
                if col in ['east', 'north', 'up']:
                    # Average displacements using nanmean
                    row[col] = np.nanmean(subset[col])
                elif col in ['sig_e', 'sig_n', 'sig_u']:
                    # Propagate error: sqrt(sum(err^2))/n
                    row[col] = np.sqrt(np.nansum(subset[col]**2)) / n
                elif np.issubdtype(gps_df[col].dtype, np.number):
                    # For other numeric columns, use the mean
                    row[col] = np.nanmean(subset[col])
                else:
                    # For non-numeric columns, assume they are constant and take the first value
                    row[col] = subset[col].iloc[0]
                    
        resampled.append(row)
    
    return pd.DataFrame(resampled)

def get_ts_lat_lon_dist(insar_dict, target_lat, target_lon, dist):
    """
    Extracts the mean (median) InSAR timeseries from pixels within a given distance
    of a target latitude and longitude, and then removes the overall mean from the timeseries.

    Parameters:
      insar_dict : dict
          Dictionary containing InSAR data with keys:
            - 'ts': 3D numpy array (time, nrows, ncols)
            - 'lons': 2D numpy array (nrows, ncols)
            - 'lats': 2D numpy array (nrows, ncols)
            - 'ts_dates': list or array of dates (e.g., in decimal years)
      target_lat : float
          The target latitude.
      target_lon : float
          The target longitude.
      dist : float
          The distance (in the same units as the lats/lons) defining the ROI as ±dist around target.

    Returns:
      median_ts : numpy.ndarray
          The adjusted median timeseries for the ROI (each value has the overall mean subtracted).
      ts_dates : list or numpy.ndarray
          The corresponding time stamps from the InSAR data.
    """
    # Create a boolean mask for pixels within the ROI.
    mask = (
        (insar_dict['lons'] > target_lon - dist) & (insar_dict['lons'] < target_lon + dist) &
        (insar_dict['lats'] > target_lat - dist) & (insar_dict['lats'] < target_lat + dist)
    )
    
    # Extract the timeseries for the ROI pixels.
    # insar_dict['ts'] has shape (time, nrows, ncols). Using the mask will flatten the spatial dims.
    roi_ts = insar_dict['ts'][:, mask]  # Shape: (time, n_pixels)
    
    # Compute the median timeseries across the ROI (across pixels) for each time step.
    median_ts = np.nanmedian(roi_ts, axis=1)
    
    # Remove the overall mean from the median timeseries.
    overall_mean = np.nanmean(median_ts)
    median_ts_adjusted = median_ts - overall_mean
    
    return median_ts_adjusted, insar_dict['ts_dates']

def calculate_rmse_nans(observed, predicted):
    """
    Compute RMSE between observed and predicted values while ignoring NaNs.
    RMSE = sqrt(sum((observed - predicted)^2) / (n_valid - 1))
    """
    diff = observed - predicted
    valid = ~np.isnan(diff)
    n_valid = np.sum(valid)
    if n_valid <= 1:
        return np.nan
    return np.sqrt(np.nansum(diff**2) / (n_valid - 1))

def get_start_end_points(lon_ori, lat_ori,  az, dist):
    start_lat, start_lon, start_z = geopy.distance.distance(kilometers=-dist).destination((lat_ori, lon_ori), bearing=az)
    end_lat, end_lon, end_z = geopy.distance.distance(kilometers=dist).destination((lat_ori, lon_ori), bearing=az)
    return start_lon, start_lat, end_lon, end_lat

def extract_profiles(xyz_dataframe, centre_lon, centre_lat, azi, dist, width, dist_bins):
    """
    Extracts profile data from a specified grid file given starting coordinates, azimuth, distance, and width.

    Parameters:
    - xyz df: The df from which to extract data.
    - start_lon, start_lat, end_lat, end_lon: Longitude and latitude of the starting point.
    - width: Width in km of the profile.

    Returns:
    - DataFrame containing the profile data.
    """
    
    # Project data to extract the profile
    points = pygmt.project(data=xyz_dataframe, center=[centre_lon, centre_lat], length=[-dist, dist], width=[-width, width], azimuth=azi, unit=True)
    points.columns = ['x', 'y', 'z', 'p', 'q', 'r', 's']

    # Prepare the DataFrame to return
    mean_points = pd.DataFrame(dist_bins, columns=['distance'])

    for i, d_bin in enumerate(dist_bins):
        points.loc[points['p'].between(d_bin-0.125, d_bin+0.125, 'both'), 'dist'] = d_bin
        subset = points [(points.dist == d_bin)]
        subset = subset.dropna()
        mean_points.loc[i, "longitude"] = np.nanmedian(subset.x)
        mean_points.loc[i, "latitude"] = np.nanmedian(subset.y)
        mean_points.loc[i, "z"] = np.nanmedian(subset.z)
        mean_points.loc[i, "p"] = d_bin
    
    mean = mean_points["z"].mean()
    
    return points, mean_points, mean


def load_h5_generalData_df(geo_file, target_file, dataset):
    with h5py.File(geo_file, 'a') as hf:
        # Read in incidence Angle
        inc = np.array(hf["incidenceAngle"][:])
        az = np.array(hf["azimuthAngle"][:])
        lon = np.array(hf["longitude"][:])
        lat = np.array(hf["latitude"][:])

    with h5py.File(target_file, 'a') as hf:
        data = np.array(hf[dataset][:])
        data[data == 0] = np.nan

    # Reshape to 1D arrays
    length = data.size
    az = az.reshape(length, 1)
    inc = inc.reshape(length, 1)
    lons = lon.reshape(length, 1)
    lats = lat.reshape(length, 1)
    data = data.reshape(length, 1)

    insar_data = pd.DataFrame(np.concatenate([lons, lats, data, az, inc], axis=1), columns=['Lon', 'Lat', dataset, 'Az', 'Inc'])
    insar_data = insar_data.dropna()

    return insar_data

def find_indices_within_radius(lons, lats, point_lon, point_lat, radius):
    """Find indices of grid points within a specified radius of a given lon/lat."""
    # Calculate the squared distance from every point in the grid
    dist_sq = (lons - point_lon)**2 + (lats - point_lat)**2
    # Find indices where distance is within the specified radius squared
    within_radius = dist_sq <= radius**2
    return np.where(within_radius)

def extract_averaged_time_series(data, lons, lats, points, radius):
    """Extract and average time series data for a list of points within a specified radius."""
    ts_list = []
    for lon, lat in points:
        idx_y, idx_x = find_indices_within_radius(lons, lats, lon, lat, radius)
        # Extract the time series data and compute the mean across all points within the radius
        if idx_y.size > 0:  # Check if there are any points within radius
            mean_ts = np.nanmean(data[:, idx_y, idx_x], axis=1)
            mean_ts = mean_ts - mean_ts[0]
        else:
            mean_ts = np.full(data.shape[0], np.nan)  # Return NaN series if no points within radius
        ts_list.append(mean_ts)
        print(ts_list)
    return ts_list

def extract_data_at_points(incidence, lons, lats, points, radius):
    """
    For each (lon, lat) in points, find all grid cells within 'radius'
    and return the mean incidence angle. If no cells fall within radius,
    return NaN for that point.
    """
    inc_list = []
    for lon_pt, lat_pt in points:
        iy, ix = find_indices_within_radius(lons, lats, lon_pt, lat_pt, radius)
        if iy.size > 0:
            mean_inc = np.nanmean(incidence[iy, ix])
        else:
            mean_inc = np.nan
        inc_list.append(mean_inc)
    return inc_list

def compute_velocity(dates, values, start, stop):
    """
    Fit a straight line to ‘values’ vs. ‘dates’ between start and stop,
    automatically ignoring any NaNs, and return (velocity, σ_velocity).

    Parameters
    ----------
    dates : array-like of float
        Decimal-year timestamps.
    values : array-like of float
        Displacements (same length as dates).
    start, stop : float
        Decimal-year window over which to fit.

    Returns
    -------
    slope : float
        Best-fit rate (units of values per year).
    stderr : float
        Standard error of the slope.
    """
    # convert to arrays
    t = np.asarray(dates, dtype=float)
    d = np.asarray(values, dtype=float)

    # mask to window and non-NaN
    m = (t >= start) & (t <= stop) & ~np.isnan(d)
    if m.sum() < 2:
        raise ValueError(f"Not enough valid points in [{start}, {stop}]")

    x = t[m]
    y = d[m]

    res = linregress(x, y)
    return res.slope, res.stderr


def gps_correction_plate_motion(geo_file: str,
                                itrf_enu_file: str,
                                gps_df: pd.DataFrame,
                                ref_station: str,
                                unit_factor: float = 1.0) -> pd.DataFrame:
    """
    Subtract the ITRF‐based plate‐motion ramp (relative to a reference station)
    from GPS ENU velocities, with optional unit conversion.

    Parameters
    ----------
    geo_file : str
        HDF5 path containing datasets 'longitude' and 'latitude'.
    itrf_enu_file : str
        HDF5 path containing datasets 'east','north','up' (in meters per year).
    gps_df : pd.DataFrame
        GPS table with columns ['StaID','Lon','Lat','Ve','Vn','Vu',...]
        (Ve/Vn/Vu should be in the same units as unit_factor conversion).
    ref_station : str
        StaID of the site to use as zero‐point reference.
    unit_factor : float, optional
        Multiply the ITRF east/north/up by this factor before subtraction.
        E.g. 1000 to convert from m yr⁻¹ to mm yr⁻¹.  Defaults to 1.0.

    Returns
    -------
    pd.DataFrame
        Copy of gps_df with added columns 'Ve_corr','Vn_corr','Vu_corr'.
    """

    # 1) load geometry grid
    with h5py.File(geo_file, 'r') as g:
        lon_grid = g['longitude'][:]
        lat_grid = g['latitude'][:]

    lon1d = lon_grid.reshape(-1)
    lat1d = lat_grid.reshape(-1)

    # 2) load and scale ITRF ENU grid
    with h5py.File(itrf_enu_file, 'r') as hf:
        east  = (hf['east'][:]  * unit_factor).reshape(-1)
        north = (hf['north'][:] * unit_factor).reshape(-1)
        up    = (hf['up'][:]    * unit_factor).reshape(-1)

    # mask zeros
    east [east  == 0] = np.nan
    north[north== 0] = np.nan
    up   [up    == 0] = np.nan

    # 3) reference station ENU
    ref = gps_df.loc[gps_df['StaID']==ref_station]
    if ref.empty:
        raise ValueError(f"Reference station '{ref_station}' not in gps_df")
    lon_ref, lat_ref = ref[['Lon','Lat']].iloc[0]

    d2_ref = (lon1d - lon_ref)**2 + (lat1d - lat_ref)**2
    idx_ref = np.nanargmin(d2_ref)
    e_ref, n_ref, u_ref = east[idx_ref], north[idx_ref], up[idx_ref]
    
    # 4) apply correction to each GPS site
    out = gps_df.copy()
    Ve_corr, Vn_corr, Vu_corr = [], [], []

    for _, row in out.iterrows():
        lon0, lat0 = row['Lon'], row['Lat']
        d2 = (lon1d - lon0)**2 + (lat1d - lat0)**2
        idx = np.nanargmin(d2)
        
        rel_e = east[idx] - e_ref
        rel_n = north[idx] - n_ref
        rel_u = up[idx]    - u_ref

        Ve_corr.append(row['Ve'] - rel_e)
        Vn_corr.append(row['Vn'] - rel_n)
        Vu_corr.append(row['Vu'] - rel_u)

    out['Ve'] = Ve_corr
    out['Vn'] = Vn_corr
    out['Vu'] = Vu_corr

    return out

def proj_los_into_vertical_no_horiz(los, inc_degrees):
    """
    Modified from Kathryn Materna tectonic utils. 
    
    Project LOS deformation into a pseudo-vertical deformation,
    assuming horizontal deformation is zero.
    Compute the vertical deformation needed to produce given LOS deformation.

    :param los: float
    :param lkv: list of 3 floats, normalized look vector components E, N, U
    """
    incidence_angle = np.deg2rad(inc_degrees)  # incidence angle from the vertical
    pseudo_vertical_disp = los / np.cos(incidence_angle)  # assuming no horizontal data contributes to LoS
    return pseudo_vertical_disp

def project_los2vector(observations, los_inc_angle_deg, los_az_angle_deg, slope_deg, aspect_deg):
    """
    Calculate the design matrix for converting LOS displacement to slope direction.
    Converts aspect from positive clockwise from north to positive anticlockwise from east.
    """
    # Convert aspect to match LOS azimuth reference
    aspect_deg = -aspect_deg + 90

    # LOS components
    los_E = np.sin(np.deg2rad(los_inc_angle_deg)) * np.sin(np.deg2rad(los_az_angle_deg)) * -1
    los_N = np.sin(np.deg2rad(los_inc_angle_deg)) * np.cos(np.deg2rad(los_az_angle_deg))
    los_U = np.cos(np.deg2rad(los_inc_angle_deg))
    
    # Downslope components
    fault_E = np.sin(np.deg2rad(slope_deg)) * np.sin(np.deg2rad(aspect_deg)) * -1
    fault_N = np.sin(np.deg2rad(slope_deg)) * np.cos(np.deg2rad(aspect_deg))
    fault_U = np.cos(np.deg2rad(slope_deg))
    
    # Normalize vectors
    L = np.array([los_E, los_N, los_U])
    L = L / np.linalg.norm(L)

    F = np.array([fault_E, fault_N, fault_U])
    F = F / np.linalg.norm(F)

    # Compute design matrix (dot product)
    G = np.dot(L, F)
    
    project_vel = observations / G if G != 0 else np.nan
    
    return project_vel

def extract_geometry_at_points(geo_file, points, radius):
    """
    For each (lon, lat) in `points`, sample all fields in `want` 
    within `radius` degrees and return a dict of lists of means.

    Parameters
    ----------
    geo_file : str
        Path to the HDF5 containing your geometry grids.
    points : list of (lon, lat) tuples
        Locations at which to sample.
    radius : float
        Search radius in degrees.

    Returns
    -------
    means : dict
        keys are dataset names (e.g. 'height','slope',…) and values are 
        lists of length len(points) giving the mean at each point.
    """
    want = [
        'height','slope','aspect',
        'incidenceAngle','azimuthAngle',]

    means = {v: [] for v in want}

    with h5py.File(geo_file, 'r') as hf:
        # load lon/lat grids once
        lons = hf['longitude'][()]
        lats = hf['latitude'][()]

        # load all other arrays into memory
        data = {v: hf[v][()] for v in want if v not in ('latitude','longitude')}

        # for each point, find local pixels and compute means
        for lon_pt, lat_pt in points:
            iy, ix = find_indices_within_radius(lons, lats, lon_pt, lat_pt, radius)
            for v in want:
                if v == 'longitude':
                    # just record the point’s longitude
                    means[v].append(lon_pt)
                elif v == 'latitude':
                    means[v].append(lat_pt)
                else:
                    arr = data[v]
                    if iy.size:
                        means[v].append(np.nanmean(arr[iy, ix]))
                    else:
                        means[v].append(np.nan)

    return means
# def calculate_average_insar_velocity_std(gps_data, insar_data, dist):
#     for index, row in gps_data.iterrows():
#         lat_min = gps_data.at[index, 'Lat'] - dist
#         lat_max = gps_data.at[index, 'Lat'] + dist
#         lon_min = gps_data.at[index, 'Lon'] - dist
#         lon_max = gps_data.at[index, 'Lon'] + dist
        
#         insar_roi = insar_data[(insar_data.Lat > lat_min) &
#                                (insar_data.Lat < lat_max) &
#                                (insar_data.Lon > lon_min) &
#                                (insar_data.Lon < lon_max)]
        
#         if not insar_roi.empty:
#             gps_data.at[index, 'insar_Vel'] = np.nanmedian(insar_roi.Vel)
#         else:
#             gps_data.at[index, 'insar_Vel'] = np.nan
    
#     gps_data = gps_data.dropna(subset=['insar_Vel'])
#     return gps_data







# def calculate_rmse_r2_and_linear_fit(observed, predicted):
#     rmse = np.sqrt(np.sum((observed - predicted)**2) / (observed.size - 1))
#     slope, intercept, r_value, p_value, std_err = stats.linregress(observed, predicted)
#     r2 = r_value ** 2
#     #print(f'RMSE: {np.round(rmse,2)}, r2: {np.round(r2,2)}')
#     return rmse, r2, slope, intercept






















    


