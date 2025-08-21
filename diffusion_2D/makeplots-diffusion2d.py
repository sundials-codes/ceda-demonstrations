#!/usr/bin/env python3
#------------------------------------------------------------
# Programmer(s):  Daniel R. Reynolds @ SMU
#------------------------------------------------------------
# Copyright (c) 2025, Southern Methodist University.
# All rights reserved.
# For details, see the LICENSE file.
#------------------------------------------------------------

# imports
import socket
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

# Set plot defaults: increase default font size, increase plot width, enable LaTeX rendering
plt.rc('font', size=15)
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.constrained_layout.use'] = True

# method name-mangling function, to convert between "internal" names and official method names
def mname(method):
    if (method == 'dirk2-Jacobi'):
        return 'DIRK2 + Jacobi'
    if (method == 'dirk2-hypre'):
        return 'DIRK2 + hypre'
    if (method == 'dirk3-Jacobi'):
        return 'DIRK3 + Jacobi'
    if (method == 'dirk3-hypre'):
        return 'DIRK3 + hypre'
    if (method == 'erk2'):
        return 'SSP-RK-2'
    if (method == 'erk3'):
        return 'SSP-RK-3'
    if (method == 'rkc'):
        return 'RKC-2'
    if (method == 'rkl'):
        return 'RKL-2'

def comparison_plot(fname, kx, tol, picname):
    """
    Generates a plot to compare all possible methods (as different curves) versus grid size, for a given diffusion coefficient and a given relative tolerance.  Saves the graphic to the requested filename.
    """

    # open simulation results as a Pandas dataframe
    data = pd.read_excel(fname)

    fig = plt.figure(figsize=[9,6])
    gs = GridSpec(2, 3, figure=fig)
    ax0 = fig.add_subplot(gs[0,0])  # top-left:     number of time steps
    ax1 = fig.add_subplot(gs[0,1])  # top-right:    number of RHS evals
    ax2 = fig.add_subplot(gs[1,0])  # bottom-left:  relative solution accuracy
    ax3 = fig.add_subplot(gs[1,1])  # bottom-right: runtime

    # iterate over all methods, adding data to plot
    for method in data['method'].sort_values().unique():

        # extract relevant data arrays
        grid = (data.groupby(['method','kx','rtol']).get_group((method,kx,tol)))['grid']
        steps = (data.groupby(['method','kx','rtol']).get_group((method,kx,tol)))['Steps']
        fevals = (data.groupby(['method','kx','rtol']).get_group((method,kx,tol)))['FEvals']
        relerr = (data.groupby(['method','kx','rtol']).get_group((method,kx,tol)))['Accuracy']
        runtime = (data.groupby(['method','kx','rtol']).get_group((method,kx,tol)))['Runtime']

        # add data to plots
        ax0.loglog(grid, steps, marker='o', ls='-', label=mname(method))
        ax1.loglog(grid, fevals, marker='o', ls='-', label=mname(method))
        ax2.loglog(grid, relerr, marker='o', ls='-', label=mname(method))
        ax3.loglog(grid, runtime, marker='o', ls='-', label=mname(method))

    fig.suptitle("Method comparisons, kx = " + repr(kx) + ', tol = ' + repr(tol))
    handles, labels = ax0.get_legend_handles_labels()
    ax0.set_title('Time steps')
    ax1.set_title('RHS evals')
    ax2.set_title('Relative error')
    ax3.set_title('Runtime')
    ax2.set_xlabel('grid size')
    ax3.set_xlabel('grid size')
    ax0.grid(linestyle='--', linewidth=0.5)
    ax1.grid(linestyle='--', linewidth=0.5)
    ax2.grid(linestyle='--', linewidth=0.5)
    ax3.grid(linestyle='--', linewidth=0.5)
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=[0.725, 0.92])
    plt.savefig(picname)

def efficiency_plot(fname, kx, grid, picname):
    """
    Generates an "efficiency plot" that shows the relative solution error versus runtime for all methods at a given spatial grid size, for a given diffusion coefficient and a given relative tolerance.  Saves the graphic to the requested filename.
    """

    # open simulation results as a Pandas dataframe
    data = pd.read_excel(fname)

    # iterate over all methods, adding data to plot
    fig = plt.figure()
    for method in data['method'].sort_values().unique():

        # extract relevant data arrays
        relerr = (data.groupby(['method','kx','grid']).get_group((method,kx,grid)))['Accuracy']
        runtime = (data.groupby(['method','kx','grid']).get_group((method,kx,grid)))['Runtime']

        # add data to plot
        plt.loglog(runtime, relerr, marker='o', ls='-', label=mname(method))

    plt.title('Computational efficiency (kx = ' + repr(kx) + ', grid = ' + repr(grid) + ')')
    plt.xlabel('runtime (sec)')
    plt.ylabel('temporal relative error')
    plt.grid(linestyle='--', linewidth=0.5)
    plt.legend()
    plt.savefig(picname)

def print_failed_tests(fname):
    data = pd.read_excel(fname)
    failed = data[data.ReturnCode != 0]
    print('Failed ' + str(len(failed)) + ' tests')
    if (len(failed) > 0):
        print(failed)

fname = "results_diffusion_2D.xlsx"
kxvals = [0.1, 1.0, 10.0]
tolvals = [1.e-3, 1.e-5]
grids = [32, 64, 128, 256]

# generate comparison plots for each (kx,tol) pair
for kx in kxvals:
    for tol in tolvals:
        comparison_plot(fname, kx, tol, 'comparison-kx'+repr(kx)+'_tol'+repr(tol)+'.png')

# generate efficiency plots for each (kx,grid) pair
for kx in kxvals:
    for grid in grids:
        efficiency_plot(fname, kx, grid, 'efficiency-kx'+repr(kx)+'_nx'+repr(grid)+'.png')

# print a list of all failed tests to stdout
print_failed_tests(fname)

# display plots
plt.show()
