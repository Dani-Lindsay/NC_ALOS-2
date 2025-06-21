# NC_ALOS-2
# 3D Deformation in Northern California with ALOS-2

This repository contains all of the scripts, notebooks, and post-processing tools used for my PhD thesis **“3D Deformation in Northern California with ALOS-2”**. It is organized to generate figures, perform geospatial and time-series analyses, and reproduce all results in the dissertation.

---

## Table of Contents

- [Overview](#overview)  
- [Thesis Context](#thesis-context)  
- [Repository Structure](#repository-structure)  
- [Dependencies](#dependencies)  
- [Getting Started](#getting-started)  
  - [Data Preparation](#data-preparation)  
  - [Generating Figures](#generating-figures)  
  - [Running Post-Processing](#running-post-processing)  
- [Notebooks](#notebooks)  
- [Scripts](#scripts)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Overview

This codebase supports the processing and analysis of ALOS-2 ScanSAR data to produce 3D surface deformation maps, time-series plots, and statistical summaries for Northern California. It includes:

- **Preprocessing** routines for Interferometric SAR stacks  
- **Time-series extraction** and denoising  
- **Integration** with GNSS ground-truth data  
- **Visualization** with PyGMT, Matplotlib, and custom plotting scripts  
- **Post-processing**: subsidence mapping, creep-rate calculation, and anomaly detection  

---

## Thesis Context

The work demonstrates how L-band ScanSAR can reveal multi-scale tectonic and non-tectonic deformation across Northern California. Key applications include:

1. Mapping long‐wavelength regional uplift/subsidence  
2. Tracking localized creep on major faults (Rodgers Creek, Maacama, Hayward)  
3. Validating InSAR velocities against continuous GNSS stations  
4. Identifying transient deformation events (landslides, aquifer changes)  

---

## Repository Structure



