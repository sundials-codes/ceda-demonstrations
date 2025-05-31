#!/usr/bin/env python3
#------------------------------------------------------------
# Programmer(s):  Daniel R. Reynolds @ SMU
#------------------------------------------------------------
# Copyright (c) 2024, Southern Methodist University.
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
Plot_ADR = True
Plot_AD = True
Plot_RD = True
Plot_Fixed = True
Plot_Adaptive = False

# legend locations
convergence_figsize = (10,4)
convergence_bbox = (0.55, 0.95)
convergence_ylim = (1e-9, 1e-3)


def make_convergence_comparison_plot(data, titletxt, picname):
    fig = plt.figure(figsize=convergence_figsize)
    gs = GridSpec(1, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0,0])
    for integrator in data['inttype'].unique():
        intdata = data.groupby(['inttype',]).get_group((integrator,))

        if (integrator != 'ExtSTS'):
            stepsize = intdata['fixedh'].to_numpy()
            accuracy = intdata['Accuracy'].to_numpy()
            rates = np.log(accuracy[1:] / accuracy[:-1]) / np.log(stepsize[1:] / stepsize[:-1])
            medrate = np.median(rates)
            ltext = '%s (rate = %.2f)' % (integrator,medrate)
            ax1.loglog(stepsize, accuracy, label=ltext, markersize=10)

        else:
            for extsts in intdata['extststype'].unique():
                extstsdata = intdata.groupby(['extststype',]).get_group((extsts,))
                for sts in extstsdata['ststype'].unique():
                    stsdata = extstsdata.groupby(['ststype',]).get_group((sts,))
                    stepsize = stsdata['fixedh'].to_numpy()
                    accuracy = stsdata['Accuracy'].to_numpy()
                    rates = np.log(accuracy[1:] / accuracy[:-1]) / np.log(stepsize[1:] / stepsize[:-1])
                    medrate = np.median(rates)
                    ltext = '%s+%s+%s (rate = %.2f)' % (integrator,extsts,sts,medrate)
                    ax1.loglog(stepsize, accuracy, label=ltext, markersize=10)

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



# generate plots, loading data from stored output
if (Plot_ADR):
    if (Plot_Fixed):
        data=pd.read_excel('AdvDiffRx-fixed.xlsx')
        make_convergence_comparison_plot(data, 'AdvDiffRx Convergence', 'adr_fixed_convergence')
    if (Plot_Adaptive):
        data=pd.read_excel('AdvDiffRx-adaptive.xlsx')
        #make_accuracy_comparison_plot(data, 'AdvDiffRx Accuracy', 'adr_adaptive_accuracy')
        #make_work_comparison_plot(data, 'AdvDiffRx Work', 'adr_adaptive_work')
        #make_efficiency_comparison_plot(data, 'AdvDiffRx Efficiency', 'adr_adaptive_efficiency')

if (Plot_AD):
    if (Plot_Fixed):
        data=pd.read_excel('AdvDiff-fixed.xlsx')
        make_convergence_comparison_plot(data, 'AdvDiff Convergence', 'ad_fixed_convergence')
    if (Plot_Adaptive):
        data=pd.read_excel('AdvDiff-adaptive.xlsx')
        #make_accuracy_comparison_plot(data, 'AdvDiff Accuracy', 'ad_adaptive_accuracy')
        #make_work_comparison_plot(data, 'AdvDiff Work', 'ad_adaptive_work')
        #make_efficiency_comparison_plot(data, 'AdvDiff Efficiency', 'ad_adaptive_efficiency')

if (Plot_RD):
    if (Plot_Fixed):
        data=pd.read_excel('RxDiff-fixed.xlsx')
        make_convergence_comparison_plot(data, 'RxDiff Convergence', 'rd_fixed_convergence')
    if (Plot_Adaptive):
        data=pd.read_excel('RxDiff-adaptive.xlsx')
        #make_accuracy_comparison_plot(data, 'RxDiff Accuracy', 'rd_adaptive_accuracy')
        #make_work_comparison_plot(data, 'RxDiff Work', 'rd_adaptive_work')
        #make_efficiency_comparison_plot(data, 'RxDiff Efficiency', 'rd_adaptive_efficiency')

# display plots
plt.show()
