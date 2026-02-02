#!/usr/bin/env python3
#------------------------------------------------------------
# Programmer(s):  Daniel R. Reynolds @ UMBC
#-----------------------------------------------------------
# Copyright (c) 2025, University of Maryland Baltimore County
# All rights reserved.
# For details, see the LICENSE file.
#------------------------------------------------------------

# imports
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

# Set plot defaults: increase default font size, increase plot width, enable LaTeX rendering
plt.rc('font', size=15)
plt.rcParams['figure.figsize'] = [7.2, 4.8]
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.constrained_layout.use'] = True

# flags to turn on/off certain plots
Generate_PDF = True
Generate_PNG = False
Plot_Fixed = True
Plot_Adaptive = True

def method_line_style(method, eigsafety, user_dom_eig, normtype):
    """Return the line style, marker, and color for a given method, eigsafety, user_dom_eig, and normtype."""
    lsty = '-'
    mark = 'o'
    colr = 'k'
    if (method == 'RKC' or method == 'RKL'):
        if (eigsafety == 1.01):
            if (user_dom_eig):
                mark = 'o'
            else:
                mark = 'v'
        elif (eigsafety == 1.05):
            if (user_dom_eig):
                mark = '8'
            else:
                mark = 's'
        elif (eigsafety == 1.1):
            if (user_dom_eig):
                mark = 'p'
            else:
                mark = '*'
        else:
            if (user_dom_eig):
                mark = '+'
            else:
                mark = 'D'
    if (normtype == 1):
        lsty = '-'
    else:
        lsty = '--'
    if (method == 'RKC'):
        colr = 'C0'
    elif (method == 'RKL'):
        colr = 'C1'
    elif (method == 'SSP2'):
        colr = 'C2'
    elif (method == 'SSP3'):
        colr = 'C3'
    elif (method == 'SSP4'):
        colr = 'C4'
    else:
        raise ValueError('Unknown method: %d' % method)
    return lsty, mark, colr


################################
def make_generic_plot(data, kval, SSPNormTypes, STSEigSafeties, STSUserDomEigs, STSNormTypes, STSDEEMaxIters,
                      PlotType, figsize, bbox, ylim, xlabel, ylabel, titletxt, picname):
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0,0])
    kdata = data.groupby(['k',]).get_group((kval,))

    for method in kdata['method'].unique():

        mdata = kdata.groupby(['method',]).get_group((method,))

        # plot RK method results
        if (method == 'SSP2' or method == 'SSP3' or method == 'SSP4'):

            for normtype in mdata['normtype'].unique():
                if normtype not in SSPNormTypes:
                    continue
                ndata = mdata.groupby(['normtype',]).get_group((normtype,))
                stepsize = ndata['h'].to_numpy()
                rtol = ndata['rtol'].to_numpy()
                rhsevals = ndata['RHSEvals'].to_numpy() + ndata['RHSEvalsDEE'].to_numpy()
                accuracy = ndata['Accuracy'].to_numpy()
                ltext = method
                if len(SSPNormTypes) > 1:
                    ltext = '%s, normtype %i' % (ltext, normtype)
                l,m,c = method_line_style(method, None, None, normtype)
                if PlotType == 'Convergence':
                    ax1.loglog(stepsize, accuracy, linestyle=l, marker=m, color=c, label=ltext)
                elif PlotType == 'Accuracy':
                    ax1.loglog(rtol, accuracy, linestyle=l, marker=m, color=c, label=ltext)
                elif PlotType == 'Efficiency':
                    ax1.loglog(rhsevals, accuracy, linestyle=l, marker=m, color=c, label=ltext)

        # plot STS method results
        else:

            for eigsafety in mdata['eigsafety'].unique():
                if eigsafety not in STSEigSafeties:
                    continue
                edata = mdata.groupby(['eigsafety',]).get_group((eigsafety,))
                for user_dom_eig in edata['user_dom_eig'].unique():
                    if user_dom_eig not in STSUserDomEigs:
                        continue
                    udata = edata.groupby(['user_dom_eig',]).get_group((user_dom_eig,))
                    for normtype in udata['normtype'].unique():
                        if normtype not in STSNormTypes:
                            continue
                        ndata = udata.groupby(['normtype',]).get_group((normtype,))
                        for dee_maxiters in ndata['dee_maxiters'].unique():
                            if dee_maxiters not in STSDEEMaxIters:
                                continue
                            ddata = ndata.groupby(['dee_maxiters',]).get_group((dee_maxiters,))
                            stepsize = ddata['h'].to_numpy()
                            rtol = ddata['rtol'].to_numpy()
                            rhsevals = ddata['RHSEvals'].to_numpy() + ddata['RHSEvalsDEE'].to_numpy()
                            accuracy = ddata['Accuracy'].to_numpy()
                            ltext = method
                            if len(STSEigSafeties) > 1:
                                ltext = '%s, safety %.2f' % (ltext, eigsafety)
                            if len(STSUserDomEigs) > 1:
                                ltext = '%s, user-dom-eig %i' % (ltext, int(user_dom_eig))
                            if len(STSNormTypes) > 1:
                                ltext = '%s, normtype %i' % (ltext, normtype)
                            if len(STSDEEMaxIters) > 1:
                                ltext = '%s, dee-maxiter %i' % (ltext, dee_maxiters)
                            l,m,c = method_line_style(method, eigsafety, user_dom_eig, normtype)
                            if PlotType == 'Convergence':
                                ax1.loglog(stepsize, accuracy, linestyle=l, marker=m, color=c, label=ltext)
                            elif PlotType == 'Accuracy':
                                ax1.loglog(rtol, accuracy, linestyle=l, marker=m, color=c, label=ltext)
                            elif PlotType == 'Efficiency':
                                ax1.loglog(rhsevals, accuracy, linestyle=l, marker=m, color=c, label=ltext)

    handles, labels = ax1.get_legend_handles_labels()
    ax1.set_title(titletxt)
    ax1.set_ylabel(ylabel)
    ax1.set_xlabel(xlabel)
    ax1.grid(linestyle='--', linewidth=0.5)
    if (ylim != None):
        ax1.set_ylim(ylim)
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=bbox, fontsize=12)
    if (Generate_PNG):
        plt.savefig(picname + '.png')
    if (Generate_PDF):
        plt.savefig(picname + '.pdf')


# generate plots, loading data from stored output
convergence_figsize = (10,4)
convergence_bbox = (0.55, 0.95)
convergence_ylim = (1e-10, 1e1)
#convergence_ylim = None

efficiency_figsize = (10,4)
efficiency_bbox = (0.55, 0.95)
efficiency_ylim = (1e-10, 1e1)

accuracy_figsize = (10,4)
accuracy_bbox = (0.55, 0.95)
accuracy_ylim = None

if (Plot_Fixed):
    data=pd.read_excel('results_gk_diffusion_1x1v_p1_fixed.xlsx')
    for k in [0.1, 1.0, 10.0]:
        make_generic_plot(data, k, [1], [1.05, 1.1], [False, True], [1], [1000],
                          'Convergence', convergence_figsize, convergence_bbox, convergence_ylim,
                          'h', 'accuracy', 'Convergence, k ='+repr(k), 'fixed_convergence-k'+repr(k))
        make_generic_plot(data, k, [1], [1.05, 1.1], [False, True], [1], [1000],
                          'Efficiency', efficiency_figsize, efficiency_bbox, efficiency_ylim, 'accuracy',
                          r'$f$ evals', 'Fixed step efficiency, k ='+repr(k), 'fixed_efficiency-k'+repr(k))

if (Plot_Adaptive):
    data=pd.read_excel('results_gk_diffusion_1x1v_p1_adaptive.xlsx')
    for k in [0.1, 1.0, 10.0]:
        make_generic_plot(data, k, [1, 2], [1.05, 1.1], [False, True], [1, 2], [1000],
                          'Accuracy', accuracy_figsize, accuracy_bbox, accuracy_ylim, 'rtol', 'accuracy',
                          'Adaptive accuracy, k ='+repr(k), 'adaptive_accuracy-k'+repr(k))
        make_generic_plot(data, k, [1, 2], [1.05, 1.1], [False, True], [1, 2], [1000], 'Efficiency',
                          efficiency_figsize, efficiency_bbox, efficiency_ylim, 'accuracy', r'$f$ evals',
                          'Adaptive step efficiency, k ='+repr(k), 'adaptive_efficiency-k'+repr(k))

# display plots
plt.show()
