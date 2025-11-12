#!/usr/bin/env python3
#------------------------------------------------------------
# Programmer(s):  Daniel R. Reynolds @ UMBC
#------------------------------------------------------------
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
Plot_ADR = True
Plot_AD = True
Plot_RD = True
Plot_Fixed = True
Plot_Adaptive = True

# utility functions to generate plots
def ark_table_name(table_id):
    """Return the name of the ARK table with the given ID."""
    if (table_id == 1):
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

def rk_line_style(table_id):
    """Return the marker and color for plotting the ARK table with the given ID."""
    if (table_id == 1):
        return 'x', 'C0'
    elif (table_id == 2):
        return '+', 'C1'
    elif (table_id == 5):
        return 'x', 'C7'
    elif (table_id == 6):
        return '+', 'C8'
    else:
        raise ValueError('Unknown table ID: %d' % table_id)

def strang_line_style(sts):
    """Return the marker and color for plotting the Strang + STS
       method."""
    if (sts == 'RKL'):
        return 'x', 'C6'
    else:
        return '+', 'C6'

def extsts_line_style(extsts,sts):
    """Return the marker and color for plotting the extended STS method type and
       STS method with the given IDs."""
    if (extsts == 'ARS'):
        if (sts == 'RKL'):
            return 'x', 'C2'
        else:
            return '+', 'C2'
    elif (extsts == 'Giraldo'):
        if (sts == 'RKL'):
            return 'x', 'C3'
        else:
            return '+', 'C3'
    elif (extsts == 'Ralston'):
        if (sts == 'RKL'):
            return 'x', 'C4'
        else:
            return '+', 'C4'
    elif (extsts == 'SSPSDIRK2'):
        if (sts == 'RKL'):
            return 'x', 'C5'
        else:
            return '+', 'C5'
    else:
        raise ValueError('Unknown extsts type: %d' % extsts)

convergence_figsize = (10,4)
convergence_bbox = (0.55, 0.95)
#convergence_ylim = (1e-10, 1e-3)
convergence_ylim = None
def make_convergence_comparison_plot(data, titletxt, picname, integrators=None):
    fig = plt.figure(figsize=convergence_figsize)
    gs = GridSpec(1, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0,0])
    for integrator in data['inttype'].unique():

        intdata = data.groupby(['inttype',]).get_group((integrator,))

        if (integrator == 'ExtSTS'):
            for extsts in intdata['extststype'].unique():
                extstsdata = intdata.groupby(['extststype',]).get_group((extsts,))
                for sts in extstsdata['ststype'].unique():
                    stsdata = extstsdata.groupby(['ststype',]).get_group((sts,))
                    stepsize = stsdata['fixedh'].to_numpy()
                    accuracy = stsdata['Accuracy'].to_numpy()
                    rates = np.log(accuracy[1:] / accuracy[:-1]) / np.log(stepsize[1:] / stepsize[:-1])
                    medrate = np.nanmedian(rates)
                    ltext = '%s+%s+%s (rate = %.2f)' % (integrator,extsts,sts,medrate)
                    m,c = extsts_line_style(extsts,sts)
                    DoPlot = True
                    if (integrators is not None):
                        if ltext not in integrators:
                            DoPlot = False
                    if DoPlot:
                        ax1.loglog(stepsize, accuracy, marker=m, color=c, label=ltext)

        elif (integrator == 'PIROCK'):
            stepsize = intdata['fixedh'].to_numpy()
            accuracy = intdata['Accuracy'].to_numpy()
            rates = np.log(accuracy[1:] / accuracy[:-1]) / np.log(stepsize[1:] / stepsize[:-1])
            medrate = np.nanmedian(rates)
            ltext = '%s (rate = %.2f)' % (integrator,medrate)
            DoPlot = True
            if (integrators is not None):
                if ltext not in integrators:
                    DoPlot = False
            if DoPlot:
                ax1.loglog(stepsize, accuracy, marker='.', color='k', label=ltext)

        elif (integrator == 'Strang'):
            for sts in intdata['ststype'].unique():
                stsdata = intdata.groupby(['ststype',]).get_group((sts,))
                stepsize = stsdata['fixedh'].to_numpy()
                accuracy = stsdata['Accuracy'].to_numpy()
                rates = np.log(accuracy[1:] / accuracy[:-1]) / np.log(stepsize[1:] / stepsize[:-1])
                medrate = np.nanmedian(rates)
                ltext = '%s+%s (rate = %.2f)' % (integrator,sts,medrate)
                m,c = strang_line_style(sts)
                DoPlot = True
                if (integrators is not None):
                    if ltext not in integrators:
                        DoPlot = False
                if DoPlot:
                    ax1.loglog(stepsize, accuracy, marker=m, color=c, label=ltext)

        else:
            for table_id in intdata['table_id'].unique():
                tabledata = intdata.groupby(['table_id',]).get_group((table_id,))
                stepsize = tabledata['fixedh'].to_numpy()
                accuracy = tabledata['Accuracy'].to_numpy()
                rates = np.log(accuracy[1:] / accuracy[:-1]) / np.log(stepsize[1:] / stepsize[:-1])
                medrate = np.nanmedian(rates)
                ltext = '%s (rate = %.2f)' % (ark_table_name(table_id),medrate)
                m,c = rk_line_style(table_id)
                DoPlot = True
                if (integrators is not None):
                    if ltext not in integrators:
                        DoPlot = False
                if DoPlot:
                    ax1.loglog(stepsize, accuracy, marker=m, color=c, label=ltext)

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

efficiency_figsize = (10,8)
efficiency_bbox = (0.55, 0.95)
def make_efficiency_comparison_plot(data, titletxt, picname, plot_adv=True, plot_rx=True, integrators=None):
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

        if (integrator == 'ExtSTS'):
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
                    m,c = extsts_line_style(extsts,sts)
                    DoPlot = True
                    if (integrators is not None):
                        if ltext not in integrators:
                            DoPlot = False
                    if DoPlot:
                        ax_diff.loglog(diffevals, accuracy, marker=m, color=c, label=ltext)
                        if (plot_adv):
                            ax_adv.loglog(advevals, accuracy, marker=m, color=c, label=ltext)
                        if (plot_rx):
                            ax_rx.loglog(rxevals, accuracy, marker=m, color=c, label=ltext)

        elif (integrator == 'PIROCK'):
            accuracy = intdata['Accuracy'].to_numpy()
            diffevals = intdata['DiffEvals'].to_numpy()
            if (plot_adv):
                advevals = intdata['AdvEvals'].to_numpy()
            if (plot_rx):
                rxevals = intdata['RxEvals'].to_numpy()
            ltext = '%s' % (integrator)
            DoPlot = True
            if (integrators is not None):
                if ltext not in integrators:
                    DoPlot = False
            if DoPlot:
                ax_diff.loglog(diffevals, accuracy, marker='.', color='k', label=ltext)
                if (plot_adv):
                    ax_adv.loglog(advevals, accuracy, marker='.', color='k', label=ltext)
                if (plot_rx):
                    ax_rx.loglog(rxevals, accuracy, marker='.', color='k', label=ltext)

        elif (integrator == 'Strang'):
            for sts in intdata['ststype'].unique():
                stsdata = intdata.groupby(['ststype',]).get_group((sts,))
                accuracy = stsdata['Accuracy'].to_numpy()
                diffevals = stsdata['DiffEvals'].to_numpy()
                if (plot_adv):
                    advevals = stsdata['AdvEvals'].to_numpy()
                if (plot_rx):
                    rxevals = stsdata['RxEvals'].to_numpy()
                ltext = '%s+%s' % (integrator,sts)
                m,c = strang_line_style(sts)
                DoPlot = True
                if (integrators is not None):
                    if ltext not in integrators:
                        DoPlot = False
                if DoPlot:
                    ax_diff.loglog(diffevals, accuracy, marker=m, color=c, label=ltext)
                    if (plot_adv):
                        ax_adv.loglog(advevals, accuracy, marker=m, color=c, label=ltext)
                    if (plot_rx):
                        ax_rx.loglog(rxevals, accuracy, marker=m, color=c, label=ltext)

        else:
            for table_id in intdata['table_id'].unique():
                tabledata = intdata.groupby(['table_id',]).get_group((table_id,))
                accuracy = tabledata['Accuracy'].to_numpy()
                diffevals = tabledata['DiffEvals'].to_numpy()
                if (plot_adv):
                    advevals = tabledata['AdvEvals'].to_numpy()
                if (plot_rx):
                    rxevals = tabledata['RxEvals'].to_numpy()
                ltext = ark_table_name(table_id)
                m,c = rk_line_style(table_id)
                DoPlot = True
                if (integrators is not None):
                    if ltext not in integrators:
                        DoPlot = False
                if DoPlot:
                    ax_diff.loglog(diffevals, accuracy, marker=m, color=c, label=ltext)
                    if (plot_adv):
                        ax_adv.loglog(advevals, accuracy, marker=m, color=c, label=ltext)
                    if (plot_rx):
                        ax_rx.loglog(rxevals, accuracy, marker=m, color=c, label=ltext)

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

runtime_efficiency_figsize = (10,4)
runtime_efficiency_bbox = (0.55, 0.95)
def make_runtime_efficiency_comparison_plot(data, titletxt, picname, integrators=None):
    fig = plt.figure(figsize=runtime_efficiency_figsize)
    gs = GridSpec(1, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0,0])
    for integrator in data['inttype'].unique():
        intdata = data.groupby(['inttype',]).get_group((integrator,))

        if (integrator == 'ExtSTS'):
            for extsts in intdata['extststype'].unique():
                extstsdata = intdata.groupby(['extststype',]).get_group((extsts,))
                for sts in extstsdata['ststype'].unique():
                    stsdata = extstsdata.groupby(['ststype',]).get_group((sts,))
                    accuracy = stsdata['Accuracy'].to_numpy()
                    runtime = stsdata['RunTime'].to_numpy()
                    ltext = '%s+%s+%s' % (integrator,extsts,sts)
                    m,c = extsts_line_style(extsts,sts)
                    DoPlot = True
                    if (integrators is not None):
                        if ltext not in integrators:
                            DoPlot = False
                    if DoPlot:
                        ax1.loglog(runtime, accuracy, marker=m, color=c, label=ltext)

        elif (integrator == 'PIROCK'):
            accuracy = intdata['Accuracy'].to_numpy()
            runtime = intdata['RunTime'].to_numpy()
            ltext = '%s' % (integrator)
            DoPlot = True
            if (integrators is not None):
                if ltext not in integrators:
                    DoPlot = False
            if DoPlot:
                ax1.loglog(runtime, accuracy, marker='.', color='k', label=ltext)

        elif (integrator == 'Strang'):
            for sts in intdata['ststype'].unique():
                stsdata = intdata.groupby(['ststype',]).get_group((sts,))
                accuracy = stsdata['Accuracy'].to_numpy()
                runtime = stsdata['RunTime'].to_numpy()
                ltext = '%s+%s' % (integrator,sts)
                m,c = strang_line_style(sts)
                DoPlot = True
                if (integrators is not None):
                    if ltext not in integrators:
                        DoPlot = False
                if DoPlot:
                    ax1.loglog(runtime, accuracy, marker=m, color=c, label=ltext)

        else:
            for table_id in intdata['table_id'].unique():
                tabledata = intdata.groupby(['table_id',]).get_group((table_id,))
                accuracy = tabledata['Accuracy'].to_numpy()
                runtime = tabledata['RunTime'].to_numpy()
                ltext = ark_table_name(table_id)
                m,c = rk_line_style(table_id)
                DoPlot = True
                if (integrators is not None):
                    if ltext not in integrators:
                        DoPlot = False
                if DoPlot:
                    ax1.loglog(runtime, accuracy, marker=m, color=c, label=ltext)

    handles, labels = ax1.get_legend_handles_labels()
    ax1.set_title(titletxt)
    ax1.set_ylabel(r'accuracy')
    ax1.set_xlabel(r'runtime')
    ax1.grid(linestyle='--', linewidth=0.5)
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=runtime_efficiency_bbox)
    if (Generate_PNG):
        plt.savefig(picname + '.png')
    if (Generate_PDF):
        plt.savefig(picname + '.pdf')

accuracy_figsize = (10,4)
accuracy_bbox = (0.55, 0.95)
accuracy_ylim = None
def make_accuracy_comparison_plot(data, titletxt, picname, integrators=None):
    fig = plt.figure(figsize=accuracy_figsize)
    gs = GridSpec(1, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0,0])
    for integrator in data['inttype'].unique():
        intdata = data.groupby(['inttype',]).get_group((integrator,))

        if (integrator == 'ExtSTS'):
            for extsts in intdata['extststype'].unique():
                extstsdata = intdata.groupby(['extststype',]).get_group((extsts,))
                for sts in extstsdata['ststype'].unique():
                    stsdata = extstsdata.groupby(['ststype',]).get_group((sts,))
                    rtol = stsdata['rtol'].to_numpy()
                    accuracy = stsdata['Accuracy'].to_numpy()
                    ltext = '%s+%s+%s' % (integrator,extsts,sts)
                    m,c = extsts_line_style(extsts,sts)
                    DoPlot = True
                    if (integrators is not None):
                        if ltext not in integrators:
                            DoPlot = False
                    if DoPlot:
                        ax1.loglog(rtol, accuracy, marker=m, color=c, label=ltext)

        elif (integrator == 'PIROCK'):
            rtol = intdata['rtol'].to_numpy()
            accuracy = intdata['Accuracy'].to_numpy()
            ltext = '%s' % (integrator)
            DoPlot = True
            if (integrators is not None):
                if ltext not in integrators:
                    DoPlot = False
            if DoPlot:
                ax1.loglog(rtol, accuracy, marker='.', color='k', label=ltext)

        elif (integrator == 'Strang'):
            for sts in intdata['ststype'].unique():
                stsdata = intdata.groupby(['ststype',]).get_group((sts,))
                rtol = stsdata['rtol'].to_numpy()
                accuracy = stsdata['Accuracy'].to_numpy()
                ltext = '%s+%s' % (integrator,sts)
                m,c = strang_line_style(sts)
                DoPlot = True
                if (integrators is not None):
                    if ltext not in integrators:
                        DoPlot = False
                if DoPlot:
                    ax1.loglog(rtol, accuracy, marker=m, color=c, label=ltext)

        else:
            for table_id in intdata['table_id'].unique():
                tabledata = intdata.groupby(['table_id',]).get_group((table_id,))
                rtol = tabledata['rtol'].to_numpy()
                accuracy = tabledata['Accuracy'].to_numpy()
                ltext = ark_table_name(table_id)
                m,c = rk_line_style(table_id)
                DoPlot = True
                if (integrators is not None):
                    if ltext not in integrators:
                        DoPlot = False
                if DoPlot:
                    ax1.loglog(rtol, accuracy, marker=m, color=c, label=ltext)

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


# utility routine to extract a specific reaction network from a Pandas dataframe.
# returns both the network name, and the dataframe subset
def extract_RxNet(data, RxNetwork):
    RxNetName = RxNetwork[0]
    RxNetData = data.groupby(['A','B','eps']).get_group((RxNetwork[1],RxNetwork[2],RxNetwork[3]))
    return RxNetName, RxNetData

# generate plots, loading data from stored output
if (Plot_ADR):
    if (Plot_Fixed):
        data=pd.read_excel('AdvDiffRx-fixed.xlsx')
        make_convergence_comparison_plot(data, 'AdvDiffRx Convergence', 'adr_fixed_convergence')
        make_efficiency_comparison_plot(data, 'AdvDiffRx Efficiency (Fixed)', 'adr_fixed_efficiency')
        make_runtime_efficiency_comparison_plot(data, 'AdvDiffRx Runtime Efficiency (Fixed)', 'adr_fixed_runtime_efficiency')
    if (Plot_Adaptive):
        data=pd.read_excel('AdvDiffRx-adapt.xlsx')
        make_accuracy_comparison_plot(data, 'AdvDiffRx Accuracy', 'adr_adaptive_accuracy')
        make_efficiency_comparison_plot(data, 'AdvDiffRx Efficiency', 'adr_adaptive_efficiency')
        make_runtime_efficiency_comparison_plot(data, 'AdvDiffRx Runtime Efficiency', 'adr_adaptive_runtime_efficiency')

if (Plot_AD):
    if (Plot_Fixed):
        data=pd.read_excel('AdvDiff-fixed.xlsx')
        make_convergence_comparison_plot(data, 'AdvDiff Convergence', 'ad_fixed_convergence')
        make_efficiency_comparison_plot(data, 'AdvDiff Efficiency (Fixed)', 'ad_fixed_efficiency', plot_rx=False)
        make_runtime_efficiency_comparison_plot(data, 'AdvDiff Runtime Efficiency (Fixed)', 'ad_fixed_runtime_efficiency')
    if (Plot_Adaptive):
        data=pd.read_excel('AdvDiff-adapt.xlsx')
        make_accuracy_comparison_plot(data, 'AdvDiff Accuracy', 'ad_adaptive_accuracy')
        make_efficiency_comparison_plot(data, 'AdvDiff Efficiency', 'ad_adaptive_efficiency', plot_rx=False)
        make_runtime_efficiency_comparison_plot(data, 'AdvDiff Runtime Efficiency', 'ad_adaptive_runtime_efficiency')

if (Plot_RD):
    if (Plot_Fixed):
        data=pd.read_excel('RxDiff-fixed.xlsx')
        make_convergence_comparison_plot(data, 'RxDiff Convergence', 'rd_fixed_convergence')
        make_efficiency_comparison_plot(data, 'RxDiff Efficiency (Fixed)', 'rd_fixed_efficiency', plot_adv=False)
        make_runtime_efficiency_comparison_plot(data, 'RxDiff Runtime Efficiency (Fixed)', 'rd_fixed_runtime_efficiency')
if (Plot_Adaptive):
        data=pd.read_excel('RxDiff-adapt.xlsx')
        make_accuracy_comparison_plot(data, 'RxDiff Accuracy', 'rd_adaptive_accuracy')
        make_efficiency_comparison_plot(data, 'RxDiff Efficiency', 'rd_adaptive_efficiency', plot_adv=False)
        make_runtime_efficiency_comparison_plot(data, 'RxDiff Runtime Efficiency', 'rd_adaptive_runtime_efficiency')

# display plots
plt.show()
