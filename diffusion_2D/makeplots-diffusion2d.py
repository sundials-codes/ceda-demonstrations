#!/usr/bin/env python3
#------------------------------------------------------------
# Programmer(s):  Daniel R. Reynolds @ UMBC
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
        return 'DIRK2'
    if (method == 'dirk2-hypre'):
        return 'DIRK2 + hypre'
    if (method == 'dirk3-Jacobi'):
        return 'DIRK3'
    if (method == 'dirk3-hypre'):
        return 'DIRK3 + hypre'
    if (method == 'erk2'):
        return 'SSP2'
    if (method == 'erk3'):
        return 'SSP3'
    if (method == 'erk4'):
        return 'SSP4'
    if (method == 'rkc'):
        return 'RKC'
    if (method == 'rkl'):
        return 'RKL'

# line styles for different methods
def linestyle(method):
    if (method == 'dirk2-Jacobi'):
        c = 'gray'
        s = '>'
        l = '-'
    if (method == 'dirk2-hypre'):
        c = 'gray'
        s = '>'
        l = '--'
    if (method == 'dirk3-Jacobi'):
        c = 'brown'
        s = '<'
        l = '-'
    if (method == 'dirk3-hypre'):
        c = 'brown'
        s = '<'
        l = '--'
    if (method == 'erk2'):
        c = 'red'
        s = 'v'
        l = '-'
    if (method == 'erk3'):
        c = 'purple'
        s = 'D'
        l = '-'
    if (method == 'erk4'):
        c = 'green'
        s = '^'
        l = '-'
    if (method == 'rkc'):
        c = 'blue'
        s = 'o'
        l = '-'
    if (method == 'rkl'):
        c = 'orange'
        s = 's'
        l = '-'
    return c, s, l

def comparison_plot(fname, kx, tol, titleprefix, picname, retainedmethods=None):
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

        # skip methods not in retained list (if given)
        if (retainedmethods is not None):
            if (method not in retainedmethods):
                continue

        # extract relevant data arrays
        grid = (data.groupby(['method','kx','rtol']).get_group((method,kx,tol)))['grid']
        steps = (data.groupby(['method','kx','rtol']).get_group((method,kx,tol)))['Steps']
        fevals = (data.groupby(['method','kx','rtol']).get_group((method,kx,tol)))['FEvals']
        relerr = (data.groupby(['method','kx','rtol']).get_group((method,kx,tol)))['Accuracy']
        runtime = (data.groupby(['method','kx','rtol']).get_group((method,kx,tol)))['Runtime']

        # add data to plots
        c, s, l = linestyle(method)
        ax0.loglog(grid, steps, marker=s, ls=l, color=c, label=mname(method))
        ax0.set_xticks([])
        ax1.loglog(grid, fevals, marker=s, ls=l, color=c, label=mname(method))
        ax1.set_xticks([])
        ax2.loglog(grid, relerr, marker=s, ls=l, color=c, label=mname(method))
        ax2.set_xticks([])
        ax3.loglog(grid, runtime, marker=s, ls=l, color=c, label=mname(method))
        ax3.set_xticks([])

    fig.suptitle(titleprefix + r' Solver statistics ($\nu_x$ = ' + repr(kx) + ', tol = ' + repr(tol) + ')')
    ax0.set_title('Time steps')
    ax1.set_title('RHS evals')
    ax2.set_title('Relative error')
    ax3.set_title('Runtime')
    ax0.grid(linestyle='--', linewidth=0.5)
    ax1.grid(linestyle='--', linewidth=0.5)
    ax2.grid(linestyle='--', linewidth=0.5)
    ax3.grid(linestyle='--', linewidth=0.5)
    ax0.set_xticks([])
    ax0.tick_params(axis='x', labelbottom=False)
    ax1.set_xticks([])
    ax1.tick_params(axis='x', labelbottom=False)
    ax2.set_xticks([])
    ax2.tick_params(axis='x', labelbottom=False)
    ax3.set_xticks([])
    ax3.tick_params(axis='x', labelbottom=False)
    ax2.set_xlabel('Grid size')
    ax3.set_xlabel('Grid size')
    plt.xticks([])
    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=[0.725, 0.92])
    plt.savefig(picname)

adaptivity_figsize = (8,6)
adaptivity_bbox = (0.57, 0.96)
adaptivity_xticks = [1e-6, 1e-4, 1e-2]
adaptivity_xlim = [10**(-6.5), 10**(-1.5)]
def adaptivity_plot(fname, kx, grid, titleprefix, picname, retainedmethods=None):
    """
    Generates a plot to compare the temporal adaptivity performance for all possible methods (as different curves) versus tolerance, for a given diffusion coefficient and a given grid size.  Saves the graphic to the requested filename.
    """

    # open simulation results as a Pandas dataframe
    data = pd.read_excel(fname)

    # iterate over all methods, adding data to plot
    fig = plt.figure(figsize=adaptivity_figsize)
    gs = GridSpec(2, 2, figure=fig)
    ax0 = fig.add_subplot(gs[0,0])  # top-left
    ax1 = fig.add_subplot(gs[1,0])  # bottom-left
    for method in data['method'].sort_values().unique():

        # skip methods not in retained list (if given)
        if (retainedmethods is not None):
            if (method not in retainedmethods):
                continue

        # extract relevant data arrays
        rtol = (data.groupby(['method','kx','grid']).get_group((method,kx,grid)))['rtol']
        runtime = (data.groupby(['method','kx','grid']).get_group((method,kx,grid)))['Runtime']
        relerr = (data.groupby(['method','kx','grid']).get_group((method,kx,grid)))['Accuracy']

        # add data to plot
        c, s, l = linestyle(method)
        ax0.loglog(rtol, relerr, marker=s, ls=l, color=c, label=mname(method))
        ax1.loglog(rtol, runtime, marker=s, ls=l, color=c, label=mname(method))

    handles, labels = ax0.get_legend_handles_labels()
    ax0.set_ylabel('temporal relative error')
    #ax0.set_ylim(adaptivity_xlim)
    ax0.set_yticks(adaptivity_xticks)
    ax0.set_xlim(adaptivity_xlim)
    ax0.set_xticks(adaptivity_xticks)
    ax0.tick_params(axis='x', labelbottom=False)
    ax1.set_xlabel('relative tolerance')
    ax1.set_ylabel('runtime (sec)')
    ax1.set_xticks(adaptivity_xticks)
    ax1.set_xlim(adaptivity_xlim)
    titletxt = titleprefix + r' Adaptivity ($\nu_x$ = ' + repr(kx) + ', ' + repr(grid) + r'$^2$ grid)'
    ax0.set_title(titletxt)
    ax0.grid(linestyle='--', linewidth=0.5)
    ax1.grid(linestyle='--', linewidth=0.5)
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=adaptivity_bbox)
    plt.savefig(picname)


efficiency_figsize = (4,3)
efficiency_bbox = (0.55, 0.95)
def efficiency_plot(fname, kxvals, grids, picname, retainedmethods=None):
    """
    Generates an "efficiency plot" that shows the relative solution error versus runtime for all methods at a given set of spatial grid sizes and for a given set of diffusion coefficients.  Saves the graphic to the requested filename.
    """

    # open simulation results as a Pandas dataframe
    data = pd.read_excel(fname)

    # determine the number of subplots needed
    ngrids = len(grids)
    nkx = len(kxvals)
    fig = plt.figure(figsize=[efficiency_figsize[0]*(nkx+1), efficiency_figsize[1]*ngrids])
    gs = GridSpec(ngrids, nkx+1, figure=fig)

    # iterate over all (kx,grid) pairs, creating subplots
    ax = []
    plotnum = 0
    for i in range(ngrids):
        grid = grids[i]
        for j in range(nkx):
            kx = kxvals[j]
            ax.append(fig.add_subplot(gs[i,j]))
            plotnum = plotnum + 1
            for method in data['method'].sort_values().unique():

                # skip methods not in retained list (if given)
                if (retainedmethods is not None):
                    if (method not in retainedmethods):
                        continue

                # extract relevant data arrays
                relerr = (data.groupby(['method','kx','grid']).get_group((method,kx,grid)))['Accuracy']
                runtime = (data.groupby(['method','kx','grid']).get_group((method,kx,grid)))['Runtime']

                # add data to plot
                c, s, l = linestyle(method)
                ax[plotnum-1].loglog(runtime, relerr, marker=s, ls=l, color=c, label=mname(method))

            # add titles/labels if this plot is in the right position
            if (i == 0):
                ax[plotnum-1].set_title(r'$\nu_x$ = ' + repr(kx))
            if ((i == ngrids-1) and (j==1)):
                ax[plotnum-1].set_xlabel('runtime (sec)')
            if (j == 0):
                ax[plotnum-1].set_ylabel(repr(grid) + r'$^2$ grid')
            ax[plotnum-1].grid(linestyle='--', linewidth=0.5)

    handles, labels = ax[0].get_legend_handles_labels()
    fig.supylabel('relative error')
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=[1.06*(nkx)/(nkx+1), 0.98])
    plt.savefig(picname)

def print_failed_tests(fname):
    data = pd.read_excel(fname)
    failed = data[data.ReturnCode != 0]
    print('Failed ' + str(len(failed)) + ' tests')
    if (len(failed) > 0):
        print(failed)

kxvals = [0.1, 1.0, 10.0]
grids = [64, 128, 256]
#grids = [32, 64]

## Adaptive test results
fname = "results_diffusion_2D.xlsx"
tolvals = [1.e-2, 1.e-3, 1.e-4, 1.e-5, 1.e-6]

# generate comparison plots for each (kx,tol) pair
for kx in kxvals:
    for tol in tolvals:
        comparison_plot(fname, kx, tol, '', 'adaptive_comparison-kx'+repr(kx)+'_tol'+repr(tol)+'.pdf',
                        ['dirk2-Jacobi','dirk3-Jacobi','erk2','erk3','erk4','rkc','rkl'])

# generate adaptivity plots for each (kx,grid) pair
for kx in kxvals:
    for grid in grids:
        adaptivity_plot(fname, kx, grid, '', 'adaptivity-kx'+repr(kx)+'_nx'+repr(grid)+'.pdf',
                        ['dirk2-Jacobi','dirk3-Jacobi','erk2','erk3','erk4','rkc','rkl'])

# generate efficiency plot for each (kx,grid) pair
efficiency_plot(fname, kxvals, grids, 'adaptive_efficiency.pdf',
                ['dirk2-Jacobi','dirk3-Jacobi','erk2','erk3','erk4','rkc','rkl'])

# print a list of all failed tests to stdout
print("Adaptive step failed tests:")
print_failed_tests(fname)

# ## Fixed test results
# fname = "results_diffusion_2D_fixedstep.xlsx"

# # generate efficiency plots for each (kx,grid) pair
# for kx in kxvals:
#     for grid in grids:
#         efficiency_plot(fname, kx, grid, 'Fixed step', 'fixed_efficiency-kx'+repr(kx)+'_nx'+repr(grid)+'.pdf',
#                         ['dirk2-hypre','dirk3-hypre','erk2','erk3','erk4','rkc','rkl'])

# # print a list of all failed tests to stdout
# print("Fixed step failed tests:")
# print_failed_tests(fname)

# display plots
#plt.show()
