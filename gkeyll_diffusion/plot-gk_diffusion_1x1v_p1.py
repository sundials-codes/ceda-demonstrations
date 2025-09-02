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
# fixed-step convergence plot
convergence_figsize = (10,4)
convergence_bbox = (0.55, 0.95)
convergence_ylim = (1e-10, 1e1)
#convergence_ylim = None
def make_convergence_comparison_plot(data, kval, titletxt, picname):
    fig = plt.figure(figsize=convergence_figsize)
    gs = GridSpec(1, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0,0])
    for k in data['k'].unique():

        # only plot for the selected k value
        if (k != kval):
            continue

        kdata = data.groupby(['k',]).get_group((k,))

        for method in kdata['method'].unique():

            mdata = kdata.groupby(['method',]).get_group((method,))

            # plot RK method results
            if (method == 'SSP2' or method == 'SSP3' or method == 'SSP4'):

                stepsize = mdata['h'].to_numpy()
                accuracy = mdata['Accuracy'].to_numpy()
                rates = np.log(accuracy[1:] / accuracy[:-1]) / np.log(stepsize[1:] / stepsize[:-1])
                medrate = np.median(rates)
                ltext = '%s (rate = %.2f)' % (method,medrate)
                l,m,c = method_line_style(method, None, None, 1)
                ax1.loglog(stepsize, accuracy, linestyle=l, marker=m, color=c, label=ltext)

            # plot STS method results
            else:

                for eigsafety in mdata['eigsafety'].unique():
                    edata = mdata.groupby(['eigsafety',]).get_group((eigsafety,))
                    for user_dom_eig in edata['user_dom_eig'].unique():
                        udata = edata.groupby(['user_dom_eig',]).get_group((user_dom_eig,))

                        # only plot for 1000 dee_maxiters
                        for dee_maxiters in udata['dee_maxiters'].unique():
                            if (dee_maxiters != 1000):
                                continue
                            ddata = udata.groupby(['dee_maxiters',]).get_group((dee_maxiters,))
                            stepsize = ddata['h'].to_numpy()
                            accuracy = ddata['Accuracy'].to_numpy()
                            rates = np.log(accuracy[1:] / accuracy[:-1]) / np.log(stepsize[1:] / stepsize[:-1])
                            medrate = np.median(rates)
                            ltext = '%s+%.2f+%i (rate = %.2f)' % (method,eigsafety,int(user_dom_eig),medrate)
                            l,m,c = method_line_style(method, eigsafety, user_dom_eig, 1)
                            ax1.loglog(stepsize, accuracy, linestyle=l, marker=m, color=c, label=ltext)

    handles, labels = ax1.get_legend_handles_labels()
    ax1.set_title(titletxt)
    ax1.set_xlabel(r'h')
    ax1.set_ylabel(r'accuracy')
    if (convergence_ylim != None):
        ax1.set_ylim(convergence_ylim)
    ax1.grid(linestyle='--', linewidth=0.5)
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=convergence_bbox)
    if (Generate_PNG):
        plt.savefig(picname + '.png')
    if (Generate_PDF):
        plt.savefig(picname + '.pdf')


################################
# efficiency comparison plot
efficiency_figsize = (10,4)
efficiency_bbox = (0.55, 0.95)
efficiency_ylim = (1e-10, 1e1)
def make_efficiency_comparison_plot(data, kval, titletxt, picname):
    fig = plt.figure(figsize=efficiency_figsize)
    gs = GridSpec(1, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0,0])
    for k in data['k'].unique():

        # only plot for the selected k value
        if (k != kval):
            continue

        kdata = data.groupby(['k',]).get_group((k,))

        for method in kdata['method'].unique():

            mdata = kdata.groupby(['method',]).get_group((method,))

            # plot RK method results
            if (method == 'SSP2' or method == 'SSP3' or method == 'SSP4'):

                for normtype in mdata['normtype'].unique():
                    ndata = mdata.groupby(['normtype',]).get_group((normtype,))
                    rhsevals = ndata['RHSEvals'].to_numpy() + ndata['RHSEvalsDEE'].to_numpy()
                    accuracy = ndata['Accuracy'].to_numpy()
                    ltext = '%s-%i' % (method,normtype)
                    l,m,c = method_line_style(method, None, None, normtype)
                    ax1.loglog(rhsevals, accuracy, linestyle=l, marker=m, color=c, label=ltext)

            # plot STS method results
            else:

                for eigsafety in mdata['eigsafety'].unique():
                    edata = mdata.groupby(['eigsafety',]).get_group((eigsafety,))
                    for user_dom_eig in edata['user_dom_eig'].unique():
                        udata = edata.groupby(['user_dom_eig',]).get_group((user_dom_eig,))
                        for normtype in udata['normtype'].unique():
                            ndata = udata.groupby(['normtype',]).get_group((normtype,))

                            # only plot for 1000 dee_maxiters
                            for dee_maxiters in ndata['dee_maxiters'].unique():
                                if (dee_maxiters != 1000):
                                    continue
                                ddata = ndata.groupby(['dee_maxiters',]).get_group((dee_maxiters,))
                                rhsevals = ddata['RHSEvals'].to_numpy() + ddata['RHSEvalsDEE'].to_numpy()
                                accuracy = ddata['Accuracy'].to_numpy()
                                ltext = '%s+%.2f+%i+%i' % (method,eigsafety,int(user_dom_eig),normtype)
                                l,m,c = method_line_style(method, eigsafety, user_dom_eig, normtype)
                                ax1.loglog(rhsevals, accuracy, linestyle=l, marker=m, color=c, label=ltext)

    handles, labels = ax1.get_legend_handles_labels()
    ax1.set_title(titletxt)
    ax1.set_ylabel(r'accuracy')
    ax1.set_xlabel(r'$f$ evals')
    ax1.grid(linestyle='--', linewidth=0.5)
    if (efficiency_ylim != None):
        ax1.set_ylim(efficiency_ylim)
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=efficiency_bbox)
    if (Generate_PNG):
        plt.savefig(picname + '.png')
    if (Generate_PDF):
        plt.savefig(picname + '.pdf')

################################
# accuracy comparison plot
accuracy_figsize = (10,4)
accuracy_bbox = (0.55, 0.95)
accuracy_ylim = None
def make_accuracy_comparison_plot(data, kval, titletxt, picname):
    fig = plt.figure(figsize=accuracy_figsize)
    gs = GridSpec(1, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0,0])
    for k in data['k'].unique():

        # only plot for the selected k value
        if (k != kval):
            continue

        kdata = data.groupby(['k',]).get_group((k,))

        for method in kdata['method'].unique():

            mdata = kdata.groupby(['method',]).get_group((method,))

            # plot RK method results
            if (method == 'SSP2' or method == 'SSP3' or method == 'SSP4'):

                for normtype in mdata['normtype'].unique():
                    ndata = mdata.groupby(['normtype',]).get_group((normtype,))
                    rtol = ndata['rtol'].to_numpy()
                    accuracy = ndata['Accuracy'].to_numpy()
                    ltext = '%s-%i' % (method,normtype)
                    l,m,c = method_line_style(method, None, None, normtype)
                    ax1.loglog(rtol, accuracy, linestyle=l, marker=m, color=c, label=ltext)

            # plot STS method results
            else:

                for eigsafety in mdata['eigsafety'].unique():
                    edata = mdata.groupby(['eigsafety',]).get_group((eigsafety,))
                    for user_dom_eig in edata['user_dom_eig'].unique():
                        udata = edata.groupby(['user_dom_eig',]).get_group((user_dom_eig,))
                        for normtype in udata['normtype'].unique():
                            ndata = udata.groupby(['normtype',]).get_group((normtype,))

                            # only plot for 1000 dee_maxiters
                            for dee_maxiters in ndata['dee_maxiters'].unique():
                                if (dee_maxiters != 1000):
                                    continue
                                ddata = ndata.groupby(['dee_maxiters',]).get_group((dee_maxiters,))
                                rtol = ddata['rtol'].to_numpy()
                                accuracy = ddata['Accuracy'].to_numpy()
                                ltext = '%s+%.2f+%i+%i' % (method,eigsafety,int(user_dom_eig),normtype)
                                l,m,c = method_line_style(method, eigsafety, user_dom_eig, normtype)
                                ax1.loglog(rtol, accuracy, linestyle=l, marker=m, color=c, label=ltext)

    handles, labels = ax1.get_legend_handles_labels()
    ax1.set_title(titletxt)
    ax1.set_xlabel(r'rtol')
    ax1.set_ylabel(r'accuracy')
    if (accuracy_ylim != None):
        ax1.set_ylim(accuracy_ylim)
    ax1.grid(linestyle='--', linewidth=0.5)
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=accuracy_bbox)
    if (Generate_PNG):
        plt.savefig(picname + '.png')
    if (Generate_PDF):
        plt.savefig(picname + '.pdf')


# generate plots, loading data from stored output
if (Plot_Fixed):
    data=pd.read_excel('results_gk_diffusion_1x1v_p1_fixed.xlsx')
    for k in [0.1, 1.0, 10.0]:
        make_convergence_comparison_plot(data, k, 'Convergence, k ='+repr(k), 'fixed_convergence-k'+repr(k))
        make_efficiency_comparison_plot(data, k, 'Fixed step efficiency, k ='+repr(k), 'fixed_efficiency-k'+repr(k))
if (Plot_Adaptive):
    data=pd.read_excel('results_gk_diffusion_1x1v_p1_adaptive.xlsx')
    for k in [0.1, 1.0, 10.0]:
        make_accuracy_comparison_plot(data, k, 'Adaptive accuracy, k ='+repr(k), 'adaptive_accuracy-k'+repr(k))
        make_efficiency_comparison_plot(data, k, 'Adaptive step efficiency, k ='+repr(k), 'adaptive_efficiency-k'+repr(k))

# display plots
plt.show()
