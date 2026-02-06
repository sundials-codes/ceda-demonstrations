#!/usr/bin/env python3
#------------------------------------------------------------
# Programmer(s):  Daniel R. Reynolds @ UMBC
#------------------------------------------------------------
# Copyright (c) 2025, Southern Methodist University.
# All rights reserved.
# For details, see the LICENSE file.
#------------------------------------------------------------

# imports
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Set plot defaults: increase default font size, increase plot width, enable LaTeX rendering
plt.rc('font', size=15)
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.constrained_layout.use'] = True

# method name-mangling function, to convert between "internal" names and official method names
def mname(method):
    if (method == 'dirk2-Jacobi'):
        return 'DIRK2'
    if (method == 'dirk3-Jacobi'):
        return 'DIRK3'
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
    if (method == 'dirk3-Jacobi'):
        c = 'brown'
        s = '<'
        l = '-'
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

kxvals = [0.1, 1.0, 10.0]
grids = [64, 128, 256]

# Adaptive test results
fname = "results_diffusion_2D.xlsx"
tolvals = [1.e-2, 1.e-3, 1.e-4, 1.e-5, 1.e-6]

# generate efficiency plot for each (kx,grid) pair
efficiency_plot(fname, kxvals, grids, 'fd-adaptive_efficiency.pdf',
                ['dirk2-Jacobi','dirk3-Jacobi','erk2','erk3','erk4','rkc','rkl'])

