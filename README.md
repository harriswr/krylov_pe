# Krylov Subspace Method for Matrix Exponentials in the Acoustic Parabolic Wave Equation

### Admin Information
18.335 Final Project
Author: William R. Harris
Date: 8 May 2025

### Summary of package
The enclosed package contains the code used to generate figures and perform analysis
for the 18.335 final project.

The goals of this project are:
- Develop code to perform matrix exponetiation using Krylov subspace methods,
- Implement this for use in a simple 2D acoustic parabolic equation for underwater uses,
- Compare to analytic solutions and established PE methods for validation

Additional validation of the code will be performed using:
- Analytic solutions to 1D heat diffusion equations
- Comparison of my implementation to Scipy built in utilities.

The organization of this project is as follows:
- conda_env:
    - A copy of the Python3 packages used for this project. If desired to rerun these codes,
    this should have all the packages needed to run.
- krylov_pe:
    - Functions and other subroutines used. This is where the meat of the project is stored.
- scripts:
    - Functions used to generate figures and perform analysis for the project. Relies on imports
    from the krylov_pe directory.
