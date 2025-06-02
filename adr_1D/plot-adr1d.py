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
Plot_Adaptive = True

# utility functions to generate plots
def ark_table_name(table_id):
    """Return the name of the ARK table with the given ID."""
    if (table_id == 0):
        return 'Default RK'
    elif (table_id == 1):
        return 'ARS-ARK21'
    elif (table_id == 2):
        return 'Giraldo-ARK21'
    elif (table_id == 3):
        return 'Ralston-ERK21'
    elif (table_id == 4):
        return 'HeunEuler-ERK21'
    elif (table_id == 5):
        return 'SSP-SDIRK21'
    elif (table_id == 6):
        return 'Giraldo-DIRK21'
    else:
        raise ValueError('Unknown table ID: %d' % table_id)

convergence_figsize = (10,4)
convergence_bbox = (0.55, 0.95)
#convergence_ylim = (1e-10, 1e-3)
convergence_ylim = None
def make_convergence_comparison_plot(data, titletxt, picname):
    fig = plt.figure(figsize=convergence_figsize)
    gs = GridSpec(1, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0,0])
    for integrator in data['inttype'].unique():
        intdata = data.groupby(['inttype',]).get_group((integrator,))

        if (integrator != 'ExtSTS'):
            for table_id in intdata['table_id'].unique():
                tabledata = intdata.groupby(['table_id',]).get_group((table_id,))
                stepsize = tabledata['fixedh'].to_numpy()
                accuracy = tabledata['Accuracy'].to_numpy()
                rates = np.log(accuracy[1:] / accuracy[:-1]) / np.log(stepsize[1:] / stepsize[:-1])
                medrate = np.median(rates)
                ltext = '%s (rate = %.2f)' % (ark_table_name(table_id),medrate)
                ax1.loglog(stepsize, accuracy, label=ltext)

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
                    ax1.loglog(stepsize, accuracy, label=ltext)

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

efficiency_figsize = (10,10)
efficiency_bbox = (0.55, 0.95)
def make_efficiency_comparison_plot(data, titletxt, picname, plot_adv=True, plot_rx=True):
    fig = plt.figure(figsize=efficiency_figsize)
    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[1,0])
    if (plot_adv):
        ax_adv = ax2
    else:
        ax_rx = ax2
    if (plot_adv and plot_rx):
        ax3 = fig.add_subplot(gs[1,1])
        ax_rx = ax3
    for integrator in data['inttype'].unique():
        intdata = data.groupby(['inttype',]).get_group((integrator,))
        ax_diff = ax1

        if (integrator != 'ExtSTS'):
            for table_id in intdata['table_id'].unique():
                tabledata = intdata.groupby(['table_id',]).get_group((table_id,))
                accuracy = tabledata['Accuracy'].to_numpy()
                diffevals = tabledata['DiffEvals'].to_numpy()
                if (plot_adv):
                    advevals = tabledata['AdvEvals'].to_numpy()
                if (plot_rx):
                    rxevals = tabledata['RxEvals'].to_numpy()
                ltext = ark_table_name(table_id)
                ax_diff.loglog(diffevals, accuracy, label=ltext)
                if (plot_adv):
                    ax_adv.loglog(advevals, accuracy, label=ltext)
                if (plot_rx):
                    ax_rx.loglog(rxevals, accuracy, label=ltext)

        else:
            for extsts in intdata['extststype'].unique():
                extstsdata = intdata.groupby(['extststype',]).get_group((extsts,))
                for sts in extstsdata['ststype'].unique():
                    stsdata = extstsdata.groupby(['ststype',]).get_group((sts,))
                    accuracy = stsdata['Accuracy'].to_numpy()
                    diffevals = stsdata['DiffEvals'].to_numpy()
                    if (plot_adv):
                        advevals = stsdata['AdvEvals'].to_numpy()
                    if (plot_rx):
                        rxevals = stsdata['RxEvals'].to_numpy()
                    ltext = '%s+%s+%s' % (integrator,extsts,sts)
                    ax_diff.loglog(diffevals, accuracy, label=ltext)
                    if (plot_adv):
                        ax_adv.loglog(advevals, accuracy, label=ltext)
                    if (plot_rx):
                        ax_rx.loglog(rxevals, accuracy, label=ltext)

    handles, labels = ax1.get_legend_handles_labels()
    ax1.set_title(titletxt)
    ax_diff.set_ylabel(r'accuracy')
    ax_diff.set_xlabel(r'$f^D$ evals')
    ax_diff.grid(linestyle='--', linewidth=0.5)
    if (plot_adv):
        ax_adv.set_ylabel(r'accuracy')
        ax_adv.set_xlabel(r'$f^A$ evals')
        ax_adv.grid(linestyle='--', linewidth=0.5)
    if (plot_rx):
        ax_rx.set_ylabel(r'accuracy')
        ax_rx.set_xlabel(r'$f^R$ evals')
        ax_rx.grid(linestyle='--', linewidth=0.5)
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=efficiency_bbox)
    if (Generate_PNG):
        plt.savefig(picname + '.png')
    if (Generate_PDF):
        plt.savefig(picname + '.pdf')

accuracy_figsize = (10,4)
accuracy_bbox = (0.55, 0.95)
accuracy_ylim = None
def make_accuracy_comparison_plot(data, titletxt, picname):
    fig = plt.figure(figsize=accuracy_figsize)
    gs = GridSpec(1, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0,0])
    for integrator in data['inttype'].unique():
        intdata = data.groupby(['inttype',]).get_group((integrator,))

        if (integrator != 'ExtSTS'):
            for table_id in intdata['table_id'].unique():
                tabledata = intdata.groupby(['table_id',]).get_group((table_id,))
                rtol = tabledata['rtol'].to_numpy()
                accuracy = tabledata['Accuracy'].to_numpy()
                ltext = ark_table_name(table_id)
                ax1.loglog(rtol, accuracy, label=ltext)

        else:
            for extsts in intdata['extststype'].unique():
                extstsdata = intdata.groupby(['extststype',]).get_group((extsts,))
                for sts in extstsdata['ststype'].unique():
                    stsdata = extstsdata.groupby(['ststype',]).get_group((sts,))
                    rtol = stsdata['rtol'].to_numpy()
                    accuracy = stsdata['Accuracy'].to_numpy()
                    ltext = '%s+%s+%s' % (integrator,extsts,sts)
                    ax1.loglog(rtol, accuracy, label=ltext)

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
if (Plot_ADR):
    if (Plot_Fixed):
        data=pd.read_excel('AdvDiffRx-fixed.xlsx')
        make_convergence_comparison_plot(data, 'AdvDiffRx Convergence', 'adr_fixed_convergence')
        make_efficiency_comparison_plot(data, 'AdvDiffRx Efficiency (Fixed)', 'adr_fixed_efficiency')
    if (Plot_Adaptive):
        data=pd.read_excel('AdvDiffRx-adapt.xlsx')
        make_accuracy_comparison_plot(data, 'AdvDiffRx Accuracy', 'adr_adaptive_accuracy')
        make_efficiency_comparison_plot(data, 'AdvDiffRx Efficiency', 'adr_adaptive_efficiency')

if (Plot_AD):
    if (Plot_Fixed):
        data=pd.read_excel('AdvDiff-fixed.xlsx')
        make_convergence_comparison_plot(data, 'AdvDiff Convergence', 'ad_fixed_convergence')
        make_efficiency_comparison_plot(data, 'AdvDiff Efficiency (Fixed)', 'ad_fixed_efficiency', plot_rx=False)
    if (Plot_Adaptive):
        data=pd.read_excel('AdvDiff-adapt.xlsx')
        make_accuracy_comparison_plot(data, 'AdvDiff Accuracy', 'ad_adaptive_accuracy')
        make_efficiency_comparison_plot(data, 'AdvDiff Efficiency', 'ad_adaptive_efficiency', plot_rx=False)

if (Plot_RD):
    if (Plot_Fixed):
        data=pd.read_excel('RxDiff-fixed.xlsx')
        make_convergence_comparison_plot(data, 'RxDiff Convergence', 'rd_fixed_convergence')
        make_efficiency_comparison_plot(data, 'RxDiff Efficiency (Fixed)', 'rd_fixed_efficiency', plot_adv=False)
if (Plot_Adaptive):
        data=pd.read_excel('RxDiff-adapt.xlsx')
        make_accuracy_comparison_plot(data, 'RxDiff Accuracy', 'rd_adaptive_accuracy')
        make_efficiency_comparison_plot(data, 'RxDiff Efficiency', 'rd_adaptive_efficiency', plot_adv=False)

# display plots
plt.show()
