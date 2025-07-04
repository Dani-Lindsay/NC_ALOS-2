#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 20:46:37 2025

@author: daniellelindsay
"""

from NC_ALOS2_filepaths import (common_paths, paths_068, paths_169, paths_170, paths_gps, decomp)
import insar_utils as utils
import numpy as np                # Matrix calculations
import pandas as pd               # Pandas for data
import h5py

distance_threshold = common_paths["dist"]
ref_station = common_paths["ref_station"]
ref_lat = common_paths["ref_lat"]
ref_lon = common_paths["ref_lon"]

unit = 1000

##########################3
# Method 1: Project GNSS to LOS and subtract. 
##########################3

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

def project_gps2los(azi, inc, gps_e, gps_n):
    """
    Project GPS ENU data onto LOS for each data point.
    
    Parameters:
    des_azi - pandas Series representing the descending azimuth angles.
    des_inc - pandas Series representing the descending incidence angles.
    gps_e - pandas Series representing the GPS east velocities.
    gps_n - pandas Series representing the GPS north velocities.

    Returns:
    v_los - pandas Series with the projected velocities onto LOS.
    """

    # Calculate the LOS velocities using vectorized operations
    v_los = (
        gps_e * np.sin(np.radians(inc)) * np.sin(np.radians(azi)) * -1
        + gps_n * np.sin(np.radians(inc)) * np.cos(np.radians(azi))
    )
    
    return v_los

# Function to perform least-squares inversion for a single row (i.e., one data point)
def invert_single_point_ad2eu(row):
    """
    Perform least-squares inversion using only ascending and descending LOS vectors
    to solve for East and Up components (North is excluded).
    """
    # Check if 'asc_vel' or 'des_vel' is NaN
    if pd.isnull(row['asc_vel']) or pd.isnull(row['des_vel']):
        # If any of the specified columns are NaN, return NaN for east and up
        return pd.Series({'east': np.nan, 'up': np.nan})
    
    # Design matrix (only solving for East and Up, excluding North)
    G = np.array([
        [np.sin(np.radians(row['asc_azi'])) * np.sin(np.radians(row['asc_inc'])) * -1,  # East component (ascending)
         np.cos(np.radians(row['asc_inc']))],  # Up component (ascending)
        [np.sin(np.radians(row['des_azi'])) * np.sin(np.radians(row['des_inc'])) * -1,  # East component (descending)
         np.cos(np.radians(row['des_inc']))]   # Up component (descending)
    ])
    
    # Observation vector (LOS displacements for ascending and descending)
    dlos = np.array([row['asc_vel'], row['des_vel']])
    
    # Perform the least-squares inversion for this point
    EN_result, _, _, _ = np.linalg.lstsq(G, dlos, rcond=None)
    
    # Return the East and Up components
    return pd.Series({'east': EN_result[0], 'up': EN_result[1]})

# Function to perform least-squares inversion for a single row (i.e., one data point)
def invert_single_point_aden2enu(row):
    """
    Check if any of 'asc_vel', 'des_vel', or 'gnss_east' is NaN
    if pd.isnull(row['asc_vel']) or pd.isnull(row['des_vel']) or pd.isnull(row['gnss_east']):
        # If any of the specified columns are NaN, return NaN for east, north, and up
        return pd.Series({'east': np.nan, 'north': np.nan, 'up': np.nan})
    
    Design matrix for ascending and descending LOS vectors at a single point
    project ENU onto LOS
    v_los = (  row['Ve'] * np.sin(np.deg2rad(inc_angle)) * np.sin(np.deg2rad(az_angle)) * -1
              + row['Vn'] * np.sin(np.deg2rad(inc_angle)) * np.cos(np.deg2rad(az_angle))
              + row['Vu'] * np.cos(np.deg2rad(inc_angle)))
    gps_data.at[index, 'UNR_Vel'] = v_los
    """
    G = np.array([
        [np.sin(np.radians(row['asc_azi'])) * np.sin(np.radians(row['asc_inc'])) * -1,  # East component (ascending)
         np.cos(np.radians(row['asc_azi'])) * np.sin(np.radians(row['asc_inc'])),  # North component (ascending)
         np.cos(np.radians(row['asc_inc']))],  # Up component (ascending)
        [np.sin(np.radians(row['des_azi'])) * np.sin(np.radians(row['des_inc'])) * -1,  # East component (descending)
         np.cos(np.radians(row['des_azi'])) * np.sin(np.radians(row['des_inc'])),  # North component (descending)
         np.cos(np.radians(row['des_inc']))],  # Up component (descending)
        [1, 0, 0],  # GNSS East
        [0, 1, 0]   # GNSS North
    ])
    
    # Observation vector (LOS displacements and GNSS displacements for a single point)
    dlos = np.array([row['asc_vel'], row['des_vel'], row['gnss_east'], row['gnss_north']])
    
    # Perform the least-squares inversion for this point
    ENU_result, _, _, _ = np.linalg.lstsq(G, dlos, rcond=None)
    
    # Return the East, North, and Up components
    return pd.Series({'east': ENU_result[0], 'north': ENU_result[1], 'up': ENU_result[2]})


# Function to calculate the mean of each column in the ROI and assign it to gps_df
def calculate_column_averages(gps_data, insar_data, dist, columns_to_average):
    """
    For each row in gps_data, find the ROI in insar_data based on a distance threshold
    and calculate the mean of each column in columns_to_average within the ROI.
    """
    for index, row in gps_data.iterrows():
        # Define latitude and longitude bounds for the ROI
        lat_min = gps_data.at[index, 'Lat'] - dist
        lat_max = gps_data.at[index, 'Lat'] + dist
        lon_min = gps_data.at[index, 'Lon'] - dist
        lon_max = gps_data.at[index, 'Lon'] + dist
        
        # Find the region of interest (ROI) in insar_data based on asc_lon, asc_lat
        insar_roi = insar_data[(insar_data['asc_lat'] > lat_min) &
                               (insar_data['asc_lat'] < lat_max) &
                               (insar_data['asc_lon'] > lon_min) &
                               (insar_data['asc_lon'] < lon_max)]
        
        if not insar_roi.empty:
            # For each column to average, calculate the mean value in the ROI
            for col in columns_to_average:
                gps_data.at[index, col] = np.nanmean(insar_roi[col])
        else:
            # If ROI is empty, assign NaN to the respective columns
            for col in columns_to_average:
                gps_data.at[index, col] = np.nan

    return gps_data

def write_new_h5_with_indices(df, col_name, outfile, shape, suffix):
    """
    Write a new HDF5 file from scratch containing a dataset named 'velocity'
    by reassembling the original 2D array using preserved 'row' and 'col' columns
    from the DataFrame. Cells corresponding to dropped rows will remain as NaN.
    
    Parameters:
      df: DataFrame that contains the column col_name along with 'row' and 'col' fields.
      col_name: The name of the DataFrame column to be saved.
      outfile: The output file path for the new HDF5 file.
      shape: The original 2D shape (tuple) of the data.
      suffix: A suffix string (for record-keeping; not used in file naming here).
    
    Returns:
      outfile: The new HDF5 file path.
    """
    # Create a full array filled with NaNs
    full_array = np.full(shape, np.nan)
    
    # Fill in the values using the preserved 'row' and 'col' indices
    for _, row in df.iterrows():
        r = int(row['row'])
        c = int(row['col'])
        full_array[r, c] = row[col_name]
    
    # Write the full array to a new HDF5 file from scratch
    with h5py.File(outfile, 'w') as hf:
        hf.create_dataset("velocity", data=full_array)
    
    return outfile

###########################
# Load GNSS data
###########################

# Define column names and read the GPS ENU file using pandas
columns = ['Lon', 'Lat', 'Ve', 'Vn', 'Vu', 'Std_e', 'Std_n', 'Std_u', 'StaID']
#gps_df = pd.read_csv(paths_gps["visr"]["gps_enu"], delim_whitespace=True, comment='#', names=columns)
gps_df = utils.load_UNR_gps(paths_gps["170_enu"], ref_station)

# Read GPS grid files for north and east displacements along with the coordinate arrays
gnss_lon_1d, gnss_lat_1d, gnss_north = utils.load_gmt_grid(paths_gps["visr"]["north"])
_, _, gnss_east = utils.load_gmt_grid(paths_gps["visr"]["east"])

# Compute step sizes from the 1D coordinate arrays
lon_step = gnss_lon_1d[1] - gnss_lon_1d[0]
lat_step = gnss_lat_1d[1] - gnss_lat_1d[0]

# Extend the 1D coordinate arrays by appending one extra value (last value + step size)
gnss_lon_1d_ext = np.append(gnss_lon_1d, gnss_lon_1d[-1] + lon_step)
gnss_lat_1d_ext = np.append(gnss_lat_1d, gnss_lat_1d[-1] + lat_step)

# Create a 2D meshgrid from the extended coordinate arrays
gnss_lon, gnss_lat_ext = np.meshgrid(gnss_lon_1d_ext, gnss_lat_1d_ext)

# Flip the latitude axis of the coordinate meshgrid for proper orientation
gnss_lat = np.flipud(gnss_lat_ext)

# Flip the displacement arrays (north and east) as well
gnss_north_flipped = np.flipud(gnss_north)
gnss_east_flipped = np.flipud(gnss_east)

# Extend the displacement arrays by repeating the final column and row
# Step 1: Add an extra column by horizontally stacking the last column
gnss_east_ext = np.hstack([gnss_east_flipped, gnss_east_flipped[:, -1][:, np.newaxis]])
gnss_north_ext = np.hstack([gnss_north_flipped, gnss_north_flipped[:, -1][:, np.newaxis]])

# Step 2: Add an extra row by vertically stacking the last row
gnss_east = np.vstack([gnss_east_ext, gnss_east_ext[-1, :]])
gnss_north = np.vstack([gnss_north_ext, gnss_north_ext[-1, :]])

###########################
# Load InSAR Data 
###########################

des_lon, des_lat, des_vel, des_azi, des_inc = utils.load_insar_vel_data_as_2Darrays(
    paths_170["geo"]["geo_geometryRadar"], 
    paths_170["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"]
)

asc_lon, asc_lat, asc_vel, asc_azi, asc_inc = utils.load_insar_vel_data_as_2Darrays(
    paths_068["geo"]["geo_geometryRadar"], 
    paths_068["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"]
)

des_vel = des_vel * unit
asc_vel = asc_vel * unit

# Assume asc_lon is a 2D array with shape (ny, nx)
ny, nx = asc_lon.shape
rows, cols = np.indices((ny, nx))

# Convert all arrays into a DataFrame, including location data (longitude and latitude)
data = pd.DataFrame({
    'asc_lon': asc_lon.ravel(),
    'asc_lat': asc_lat.ravel(),
    'des_lon': des_lon.ravel(),
    'des_lat': des_lat.ravel(),
    'asc_inc': asc_inc.ravel(),
    'des_inc': des_inc.ravel(),
    'asc_azi': asc_azi.ravel(),
    'des_azi': des_azi.ravel(),
    'asc_vel': asc_vel.ravel(),
    'des_vel': des_vel.ravel(),
    'gnss_east': gnss_east.ravel(),
    'gnss_north': gnss_north.ravel(),
    'gnss_lon': gnss_lon.ravel(),  # Add GNSS longitude
    'gnss_lat': gnss_lat.ravel()   # Add GNSS latitude
})

orig_shape = asc_lon.shape  # e.g., (ny, nx)

ny, nx = asc_lon.shape
rows, cols = np.indices((ny, nx))
data['row'] = rows.ravel()
data['col'] = cols.ravel()

# Drop rows where 'asc_vel', 'des_vel', or 'gnss_east' contain NaN values
data = data.dropna(subset=['asc_vel', 'des_vel', 'gnss_east'])

# Optionally reset the index
data.reset_index(drop=True, inplace=True)

##########################3
# Project GNSS to LOS 
##########################3

# Project east and north velocities to LOS
data["des_gpsLOS"] = project_gps2los(data["des_azi"], data["des_inc"], data["gnss_east"], data["gnss_north"])
data["asc_gpsLOS"] = project_gps2los(data["asc_azi"], data["asc_inc"], data["gnss_east"], data["gnss_north"])

# Subtract east and north from LOS to get to sudo vertical. 
data["des_sudo_Up"] = data["des_vel"] - data["des_gpsLOS"]
data["asc_sudo_Up"] = data["asc_vel"] - data["asc_gpsLOS"]

##########################3
# Method 2 & 3: Invert asc. des, east, up. 
##########################3

# Chunk size (number of rows to process at once)
chunk_size = 5000

# Split the DataFrame into chunks and process each chunk separately
num_chunks = len(data) // chunk_size + 1  # Calculate the number of chunks
for chunk_idx in range(num_chunks):
    # Determine the start and end indices for the current chunk
    start_idx = chunk_idx * chunk_size
    end_idx = min((chunk_idx + 1) * chunk_size, len(data))

    # Process the chunk of the DataFrame
    print(f"Processing chunk {chunk_idx + 1}/{num_chunks}, rows {start_idx} to {end_idx}")
    
    data_chunk = data.iloc[start_idx:end_idx].copy()  # Get the current chunk of data

    # Apply the first inversion (asc and des solving for East and Up)
    data_chunk[['ad_east', 'ad_up']] = data_chunk.apply(invert_single_point_ad2eu, axis=1)

    # Apply the second inversion (asc, des, and GNSS solving for East, North, and Up)
    data_chunk[['aden_east', 'aden_north', 'aden_up']] = data_chunk.apply(invert_single_point_aden2enu, axis=1)
    
    # Store the results back in the original DataFrame
    data.loc[start_idx:end_idx-1, ['ad_east', 'ad_up']] = data_chunk[['ad_east', 'ad_up']]
    data.loc[start_idx:end_idx-1, ['aden_east', 'aden_north', 'aden_up']] = data_chunk[['aden_east', 'aden_north', 'aden_up']]


##########################3
# RMSE for each "Up" 
##########################3

# Columns to calculate RMSE and R²
columns_for_rmse = ['des_sudo_Up', 'asc_sudo_Up', 'ad_up', "aden_up"]

# Apply the function to gps_df and data (this will populate gps_df with averaged values)
gps_df = calculate_column_averages(gps_df, data, distance_threshold, columns_for_rmse)
gps_df = gps_df.dropna()
gps_df_sorted = gps_df.sort_values(by='Lat')

# Dictionary to store the results for each column
results_dict = {}

# Loop through each column in gps_df and compare with `gps_df['Vu']`
for col in columns_for_rmse:
    observed = gps_df['Vu']
    predicted = gps_df[col]
    
    # Ensure no NaN values before calculations
    valid_indices = (~np.isnan(observed)) & (~np.isnan(predicted))
    
    if valid_indices.any():
        # Calculate RMSE and R²
        rmse, r2, slope, intercept = utils.calculate_rmse_r2_and_linear_fit(observed[valid_indices], predicted[valid_indices])
        results_dict[col] = {'rmse': rmse, 'r2': r2, 'slope': slope, 'intercept': intercept}
        
        # Calculate residuals (observed - predicted)
        residuals = observed[valid_indices] - predicted[valid_indices]
        #Positive residuals mean that the GPS velocity (observed) is larger than the InSAR velocity (predicted). 
        #Negative residuals mean that the InSAR velocity (predicted) is larger than the GPS velocity (observed). 
        
        # Store residuals in the gps_df as a new column named 'residual_<col>'
        gps_df.loc[valid_indices, f'residual_{col}'] = residuals
    else:
        print(f"No valid data for comparison between 'Vu' and '{col}'.")

# Display the results
print("RMSE and R² results for each column:")
for col, metrics in results_dict.items():
    print(f"{col}: RMSE = {metrics['rmse']:.2f}, R² = {metrics['r2']:.2f}, Slope = {metrics['slope']:.2f}, Intercept = {metrics['intercept']:.2f}")



# Save the desired columns as new HDF5 files
#df, col_name, outfile, shape, suffix
outfile_asc_semi = write_new_h5_with_indices(data, 'asc_sudo_Up',
                                             decomp["CASR"]["asc_semi"],
                                             orig_shape, "asc_semi")
outfile_des_semi = write_new_h5_with_indices(data, 'des_sudo_Up',
                                             decomp["CASR"]["des_semi"],
                                             orig_shape, "des_semi")

outfile_insar_only = write_new_h5_with_indices(data, 'ad_up',
                                               decomp["CASR"]["insar_only_up"],
                                               orig_shape, "insar_only")
outfile_insar_only = write_new_h5_with_indices(data, 'ad_east',
                                               decomp["CASR"]["insar_only_east"],
                                               orig_shape, "insar_only_east")

outfile_gps_insar = write_new_h5_with_indices(data, 'aden_up',
                                              decomp["CASR"]["gps_insar_up"],
                                              orig_shape, "gps_insar")
outfile_gps_insar = write_new_h5_with_indices(data, 'aden_east',
                                              decomp["CASR"]["gps_insar_east"],
                                              orig_shape, "gps_insar_east")
outfile_gps_insar = write_new_h5_with_indices(data, 'aden_north',
                                              decomp["CASR"]["gps_insar_north"],
                                              orig_shape, "gps_insar_north")




src_file = paths_170["geo"]["geo_velocity_SET_ERA5_demErr_ITRF14_deramp_msk"]

with h5py.File(src_file, "r") as src, h5py.File(decomp["CASR"]["asc_semi"], "a") as dest:
    for key, value in src.attrs.items():
        dest.attrs[key] = value

with h5py.File(src_file, "r") as src, h5py.File(decomp["CASR"]["des_semi"], "a") as dest:
    for key, value in src.attrs.items():
        dest.attrs[key] = value

with h5py.File(src_file, "r") as src, h5py.File(decomp["CASR"]["insar_only_up"], "a") as dest:
    for key, value in src.attrs.items():
        dest.attrs[key] = value
        
with h5py.File(src_file, "r") as src, h5py.File(decomp["CASR"]["insar_only_east"], "a") as dest:
    for key, value in src.attrs.items():
        dest.attrs[key] = value

with h5py.File(src_file, "r") as src, h5py.File(decomp["CASR"]["gps_insar_up"], "a") as dest:
    for key, value in src.attrs.items():
        dest.attrs[key] = value
        
with h5py.File(src_file, "r") as src, h5py.File(decomp["CASR"]["gps_insar_east"], "a") as dest:
    for key, value in src.attrs.items():
        dest.attrs[key] = value
        
with h5py.File(src_file, "r") as src, h5py.File(decomp["CASR"]["gps_insar_north"], "a") as dest:
    for key, value in src.attrs.items():
        dest.attrs[key] = value


# --- Convert all .grd files to mm (×1000) for each track ---
for name, grd_path in decomp["CASR"].items():
    if grd_path.endswith(".h5"):
        utils.run_command(["save_gmt.py", decomp["CASR"][name], "-o",  decomp["grd"][name]])

        #utils.run_command(["gmt", "grdmath", grd_path, "1000", "MUL", "=", mm_path])

##########################3
# Repeat for MLV Examples 
##########################3

ref_station = "P784"

###########################
# Load InSAR Data 
###########################

des_lon, des_lat, des_vel, des_azi, des_inc = utils.load_insar_vel_data_as_2Darrays(
    paths_170["P784"]["geo_geometryRadar"], 
    paths_170["P784"]["geo_velocity_msk"]
)

asc_lon, asc_lat, asc_vel, asc_azi, asc_inc = utils.load_insar_vel_data_as_2Darrays(
    paths_068["P784"]["geo_geometryRadar"], 
    paths_068["P784"]["geo_velocity_msk"]
)

des_vel = des_vel * unit
asc_vel = asc_vel * unit

# Assume asc_lon is a 2D array with shape (ny, nx)
ny, nx = asc_lon.shape
rows, cols = np.indices((ny, nx))

# Convert all arrays into a DataFrame, including location data (longitude and latitude)
data = pd.DataFrame({
    'asc_lon': asc_lon.ravel(),
    'asc_lat': asc_lat.ravel(),
    'des_lon': des_lon.ravel(),
    'des_lat': des_lat.ravel(),
    'asc_inc': asc_inc.ravel(),
    'des_inc': des_inc.ravel(),
    'asc_azi': asc_azi.ravel(),
    'des_azi': des_azi.ravel(),
    'asc_vel': asc_vel.ravel(),
    'des_vel': des_vel.ravel(),
})

orig_shape = asc_lon.shape  # e.g., (ny, nx)

ny, nx = asc_lon.shape
rows, cols = np.indices((ny, nx))
data['row'] = rows.ravel()
data['col'] = cols.ravel()

# Drop rows where 'asc_vel', 'des_vel', or 'gnss_east' contain NaN values
data = data.dropna(subset=['asc_vel', 'des_vel', ])

# Optionally reset the index
data.reset_index(drop=True, inplace=True)


##########################3
# Method 2 & 3: Invert asc. des, east, up. 
##########################3

# Chunk size (number of rows to process at once)
chunk_size = 5000

# Split the DataFrame into chunks and process each chunk separately
num_chunks = len(data) // chunk_size + 1  # Calculate the number of chunks
for chunk_idx in range(num_chunks):
    # Determine the start and end indices for the current chunk
    start_idx = chunk_idx * chunk_size
    end_idx = min((chunk_idx + 1) * chunk_size, len(data))

    # Process the chunk of the DataFrame
    print(f"Processing chunk {chunk_idx + 1}/{num_chunks}, rows {start_idx} to {end_idx}")
    
    data_chunk = data.iloc[start_idx:end_idx].copy()  # Get the current chunk of data

    # Apply the first inversion (asc and des solving for East and Up)
    data_chunk[['ad_east', 'ad_up']] = data_chunk.apply(invert_single_point_ad2eu, axis=1)

    # Store the results back in the original DataFrame
    data.loc[start_idx:end_idx-1, ['ad_east', 'ad_up']] = data_chunk[['ad_east', 'ad_up']]


# Save as new hdf5
outfile_insar_only = write_new_h5_with_indices(data, 'ad_up',
                                               decomp["P784"]["insar_only_up"],
                                               orig_shape, "insar_only")
outfile_insar_only = write_new_h5_with_indices(data, 'ad_east',
                                               decomp["P784"]["insar_only_east"],
                                               orig_shape, "insar_only_east")

# Add attributes 
src_file = paths_170["P784"]["geo_velocity_msk"]

with h5py.File(src_file, "r") as src, h5py.File(decomp["P784"]["insar_only_up"], "a") as dest:
    for key, value in src.attrs.items():
        dest.attrs[key] = value
        
with h5py.File(src_file, "r") as src, h5py.File(decomp["P784"]["insar_only_east"], "a") as dest:
    for key, value in src.attrs.items():
        dest.attrs[key] = value

# Save as gmt
utils.run_command(["save_gmt.py", decomp["P784"]["insar_only_up"],   "-o",  decomp["P784"]["insar_only_up_grd"]])
utils.run_command(["save_gmt.py", decomp["P784"]["insar_only_east"], "-o",  decomp["P784"]["insar_only_east_grd"]])
