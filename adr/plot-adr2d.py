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
#Plot_RD = True
Plot_RD = False
Plot_Fixed = True
Plot_Adaptive = True

# utility functions to generate plots
def ark_table_name(table_id):
    """Return the name of the ARK table with the given ID."""
    if (table_id == 1):
        return 'ARS'
    elif (table_id == 2):
        return 'Giraldo'
    elif (table_id == 3):
        return 'Ralston'
    elif (table_id == 4):
        return 'Heun-Euler'
    elif (table_id == 5):
        return 'SSP-SDIRK21'
    else:
        raise ValueError('Unknown table ID: %d' % table_id)

def rk_line_style(table_id,implicitrx):
    """Return the marker, color, and line style for plotting the ARK table with the given ID."""
    if (implicitrx):
        ls = '--'
    else:
        ls = '-'
    if (table_id == 1):
        return 'x', 'C0', ls
    elif (table_id == 2):
        return '+', 'C1', ls
    elif (table_id == 3):
        return '+', 'C2', ls
    elif (table_id == 4):
        return '+', 'C3', ls
    elif (table_id == 5):
        return 'x', 'C4', ls
    else:
        raise ValueError('Unknown table ID: %d' % table_id)

def strang_line_style(sts):
    """Return the marker, color, and line style for plotting the Strang + STS
       method."""
    if (sts == 'RKL'):
        return 'x', 'C8', '-'
    else:
        return '+', 'C8', '-'

def extsts_line_style(extsts,sts):
    """Return the marker, color, and line style for plotting the extended STS method type and
       STS method with the given IDs."""
    if (extsts == 'ARS'):
        if (sts == 'RKL'):
            return 'x', 'C5', '-'
        else:
            return '+', 'C5', '-'
    elif (extsts == 'Giraldo'):
        if (sts == 'RKL'):
            return 'x', 'C6', '-'
        else:
            return '+', 'C6', '-'
    elif (extsts == 'Ralston'):
        if (sts == 'RKL'):
            return 'x', 'C7', '-'
        else:
            return '+', 'C7', '-'
    elif (extsts == 'Heun-Euler'):
        if (sts == 'RKL'):
            return 'x', 'C8', '-'
        else:
            return '+', 'C8', '-'
    elif (extsts == 'SSPSDIRK2'):
        if (sts == 'RKL'):
            return 'x', 'C9', '-'
        else:
            return '+', 'C9', '-'
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
                    ltext = '%s+%s+%s' % (integrator,extsts,sts)
                    rate = ' (rate = %.2f)' % (medrate)
                    m,c,l = extsts_line_style(extsts,sts)
                    DoPlot = True
                    if (integrators is not None):
                        if ltext not in integrators:
                            DoPlot = False
                    if DoPlot:
                        ax1.loglog(stepsize, accuracy, marker=m, color=c, linestyle=l, label=ltext+rate)

        elif (integrator == 'PIROCK'):
            stepsize = intdata['fixedh'].to_numpy()
            accuracy = intdata['Accuracy'].to_numpy()
            rates = np.log(accuracy[1:] / accuracy[:-1]) / np.log(stepsize[1:] / stepsize[:-1])
            medrate = np.nanmedian(rates)
            ltext = integrator
            rate = ' (rate = %.2f)' % (medrate)
            DoPlot = True
            if (integrators is not None):
                if ltext not in integrators:
                    DoPlot = False
            if DoPlot:
                ax1.loglog(stepsize, accuracy, marker='.', color='k', linestyle='-', label=ltext+rate)

        elif (integrator == 'Strang'):
            for sts in intdata['ststype'].unique():
                stsdata = intdata.groupby(['ststype',]).get_group((sts,))
                stepsize = stsdata['fixedh'].to_numpy()
                accuracy = stsdata['Accuracy'].to_numpy()
                rates = np.log(accuracy[1:] / accuracy[:-1]) / np.log(stepsize[1:] / stepsize[:-1])
                medrate = np.nanmedian(rates)
                ltext = '%s+%s' % (integrator,sts)
                rate = ' (rate = %.2f)' % (medrate)
                m,c,l = strang_line_style(sts)
                DoPlot = True
                if (integrators is not None):
                    if ltext not in integrators:
                        DoPlot = False
                if DoPlot:
                    ax1.loglog(stepsize, accuracy, marker=m, color=c, linestyle=l, label=ltext+rate)

        else:
            for table_id in intdata['table_id'].unique():
                for rxtype in intdata['implicitrx'].unique():
                    tabledata = intdata.groupby(['table_id','implicitrx']).get_group((table_id,rxtype))
                    stepsize = tabledata['fixedh'].to_numpy()
                    accuracy = tabledata['Accuracy'].to_numpy()
                    rates = np.log(accuracy[1:] / accuracy[:-1]) / np.log(stepsize[1:] / stepsize[:-1])
                    medrate = np.nanmedian(rates)
                    if (rxtype):
                        rxtxt = 'impl-R'
                    else:
                        rxtxt = 'expl-R'
                    ltext = '%s, %s' % (ark_table_name(table_id),rxtxt)
                    rate = ' (rate = %.2f)' % (medrate)
                    m,c,l = rk_line_style(table_id,rxtype)
                    DoPlot = True
                    if (integrators is not None):
                        if ltext not in integrators:
                            DoPlot = False
                    if DoPlot:
                        ax1.loglog(stepsize, accuracy, marker=m, color=c, linestyle=l, label=ltext+rate)

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
                        if (np.sum(rxevals) == 0):
                            rxevals = stsdata['AdvEvals'].to_numpy()
                    ltext = '%s+%s+%s' % (integrator,extsts,sts)
                    m,c,l = extsts_line_style(extsts,sts)
                    DoPlot = True
                    if (integrators is not None):
                        if ltext not in integrators:
                            DoPlot = False
                    if DoPlot:
                        ax_diff.loglog(diffevals, accuracy, marker=m, color=c, linestyle=l, label=ltext)
                        if (plot_adv):
                            ax_adv.loglog(advevals, accuracy, marker=m, color=c, linestyle=l, label=ltext)
                        if (plot_rx):
                            ax_rx.loglog(rxevals, accuracy, marker=m, color=c, linestyle=l, label=ltext)

        elif (integrator == 'PIROCK'):
            accuracy = intdata['Accuracy'].to_numpy()
            diffevals = intdata['DiffEvals'].to_numpy()
            if (plot_adv):
                advevals = intdata['AdvEvals'].to_numpy()
            if (plot_rx):
                rxevals = intdata['RxEvals'].to_numpy()
                if (np.sum(rxevals) == 0):
                    rxevals = intdata['AdvEvals'].to_numpy()
            ltext = '%s' % (integrator)
            DoPlot = True
            if (integrators is not None):
                if ltext not in integrators:
                    DoPlot = False
            if DoPlot:
                ax_diff.loglog(diffevals, accuracy, marker='.', color='k', linestyle='-', label=ltext)
                if (plot_adv):
                    ax_adv.loglog(advevals, accuracy, marker='.', color='k', linestyle='-', label=ltext)
                if (plot_rx):
                    ax_rx.loglog(rxevals, accuracy, marker='.', color='k', linestyle='-', label=ltext)

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
                m,c,l = strang_line_style(sts)
                DoPlot = True
                if (integrators is not None):
                    if ltext not in integrators:
                        DoPlot = False
                if DoPlot:
                    ax_diff.loglog(diffevals, accuracy, marker=m, color=c, linestyle=l, label=ltext)
                    if (plot_adv):
                        ax_adv.loglog(advevals, accuracy, marker=m, color=c, linestyle=l, label=ltext)
                    if (plot_rx):
                        ax_rx.loglog(rxevals, accuracy, marker=m, color=c, linestyle=l, label=ltext)

        else:
            for table_id in intdata['table_id'].unique():
                for rxtype in intdata['implicitrx'].unique():
                    tabledata = intdata.groupby(['table_id','implicitrx']).get_group((table_id,rxtype))
                    accuracy = tabledata['Accuracy'].to_numpy()
                    diffevals = tabledata['DiffEvals'].to_numpy()
                    if (plot_adv):
                        advevals = tabledata['AdvEvals'].to_numpy()
                    if (plot_rx):
                        rxevals = tabledata['RxEvals'].to_numpy()
                    if (rxtype):
                        rxtxt = 'impl-R'
                    else:
                        rxtxt = 'expl-R'
                    ltext = '%s, %s' % (ark_table_name(table_id),rxtxt)
                    m,c,l = rk_line_style(table_id,rxtype)
                    DoPlot = True
                    if (integrators is not None):
                        if ltext not in integrators:
                            DoPlot = False
                    if DoPlot:
                        ax_diff.loglog(diffevals, accuracy, marker=m, color=c, linestyle=l, label=ltext)
                        if (plot_adv):
                            ax_adv.loglog(advevals, accuracy, marker=m, color=c, linestyle=l, label=ltext)
                        if (plot_rx):
                            ax_rx.loglog(rxevals, accuracy, marker=m, color=c, linestyle=l, label=ltext)

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
                    m,c,l = extsts_line_style(extsts,sts)
                    DoPlot = True
                    if (integrators is not None):
                        if ltext not in integrators:
                            DoPlot = False
                    if DoPlot:
                        ax1.loglog(runtime, accuracy, marker=m, color=c, linestyle=l, label=ltext)

        elif (integrator == 'PIROCK'):
            accuracy = intdata['Accuracy'].to_numpy()
            runtime = intdata['RunTime'].to_numpy()
            ltext = '%s' % (integrator)
            DoPlot = True
            if (integrators is not None):
                if ltext not in integrators:
                    DoPlot = False
            if DoPlot:
                ax1.loglog(runtime, accuracy, marker='.', color='k', linestyle='-', label=ltext)

        elif (integrator == 'Strang'):
            for sts in intdata['ststype'].unique():
                stsdata = intdata.groupby(['ststype',]).get_group((sts,))
                accuracy = stsdata['Accuracy'].to_numpy()
                runtime = stsdata['RunTime'].to_numpy()
                ltext = '%s+%s' % (integrator,sts)
                m,c,l = strang_line_style(sts)
                DoPlot = True
                if (integrators is not None):
                    if ltext not in integrators:
                        DoPlot = False
                if DoPlot:
                    ax1.loglog(runtime, accuracy, marker=m, color=c, linestyle=l, label=ltext)

        else:
            for table_id in intdata['table_id'].unique():
                for rxtype in intdata['implicitrx'].unique():
                    tabledata = intdata.groupby(['table_id','implicitrx']).get_group((table_id,rxtype))
                    accuracy = tabledata['Accuracy'].to_numpy()
                    runtime = tabledata['RunTime'].to_numpy()
                    if (rxtype):
                        rxtxt = 'impl-R'
                    else:
                        rxtxt = 'expl-R'
                    ltext = '%s, %s' % (ark_table_name(table_id),rxtxt)
                    m,c,l = rk_line_style(table_id,rxtype)
                    DoPlot = True
                    if (integrators is not None):
                        if ltext not in integrators:
                            DoPlot = False
                    if DoPlot:
                        ax1.loglog(runtime, accuracy, marker=m, color=c, linestyle=l, label=ltext)

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
                    m,c,l = extsts_line_style(extsts,sts)
                    DoPlot = True
                    if (integrators is not None):
                        if ltext not in integrators:
                            DoPlot = False
                    if DoPlot:
                        ax1.loglog(rtol, accuracy, marker=m, color=c, linestyle=l, label=ltext)

        elif (integrator == 'PIROCK'):
            rtol = intdata['rtol'].to_numpy()
            accuracy = intdata['Accuracy'].to_numpy()
            ltext = '%s' % (integrator)
            DoPlot = True
            if (integrators is not None):
                if ltext not in integrators:
                    DoPlot = False
            if DoPlot:
                ax1.loglog(rtol, accuracy, marker='.', color='k', linestyle='-', label=ltext)

        elif (integrator == 'Strang'):
            for sts in intdata['ststype'].unique():
                stsdata = intdata.groupby(['ststype',]).get_group((sts,))
                rtol = stsdata['rtol'].to_numpy()
                accuracy = stsdata['Accuracy'].to_numpy()
                ltext = '%s+%s' % (integrator,sts)
                m,c,l = strang_line_style(sts)
                DoPlot = True
                if (integrators is not None):
                    if ltext not in integrators:
                        DoPlot = False
                if DoPlot:
                    ax1.loglog(rtol, accuracy, marker=m, color=c, linestyle=l, label=ltext)

        else:
            for table_id in intdata['table_id'].unique():
                for rxtype in intdata['implicitrx'].unique():
                    tabledata = intdata.groupby(['table_id','implicitrx']).get_group((table_id,rxtype))
                    rtol = tabledata['rtol'].to_numpy()
                    accuracy = tabledata['Accuracy'].to_numpy()
                    if (rxtype):
                        rxtxt = 'impl-R'
                    else:
                        rxtxt = 'expl-R'
                    ltext = '%s, %s' % (ark_table_name(table_id),rxtxt)
                    m,c,l = rk_line_style(table_id,rxtype)
                    DoPlot = True
                    if (integrators is not None):
                        if ltext not in integrators:
                            DoPlot = False
                    if DoPlot:
                        ax1.loglog(rtol, accuracy, marker=m, color=c, linestyle=l, label=ltext)

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
        data=pd.read_excel('AdvDiffRx2D-fixed.xlsx')
        make_convergence_comparison_plot(data, 'AdvDiffRx Convergence', 'adr2d_fixed_convergence',
                                         integrators=['ARS, impl-R', 'Giraldo, impl-R', 'ExtSTS+ARS+RKC', 'ExtSTS+Heun-Euler+RKL', 'ExtSTS+Giraldo+RKL', 'PIROCK'])
        make_efficiency_comparison_plot(data, 'AdvDiffRx Efficiency (Fixed)', 'adr2d_fixed_efficiency',
                                        integrators=['ARS, impl-R', 'Giraldo, impl-R', 'ExtSTS+ARS+RKC', 'ExtSTS+Heun-Euler+RKL', 'ExtSTS+Giraldo+RKL', 'PIROCK'])
        make_runtime_efficiency_comparison_plot(data, 'AdvDiffRx Runtime Efficiency (Fixed)', 'adr2d_fixed_runtime_efficiency',
                                                integrators=['ARS, impl-R', 'Giraldo, impl-R', 'ExtSTS+ARS+RKC', 'ExtSTS+Heun-Euler+RKL', 'ExtSTS+Giraldo+RKL', 'PIROCK'])
    if (Plot_Adaptive):
        data=pd.read_excel('AdvDiffRx2D-adapt.xlsx')
        make_accuracy_comparison_plot(data, 'AdvDiffRx Accuracy', 'adr2d_adaptive_accuracy',
                                      integrators=['ARS, impl-R', 'Giraldo, impl-R', 'ExtSTS+ARS+RKC', 'ExtSTS+Heun-Euler+RKL', 'ExtSTS+Giraldo+RKL', 'PIROCK'])
        make_efficiency_comparison_plot(data, 'AdvDiffRx Efficiency', 'adr2d_adaptive_efficiency',
                                        integrators=['ARS, impl-R', 'Giraldo, impl-R', 'ExtSTS+ARS+RKC', 'ExtSTS+Heun-Euler+RKL', 'ExtSTS+Giraldo+RKL', 'PIROCK'])
        make_runtime_efficiency_comparison_plot(data, 'AdvDiffRx Runtime Efficiency', 'adr2d_adaptive_runtime_efficiency',
                                                integrators=['ARS, impl-R', 'Giraldo, impl-R', 'ExtSTS+ARS+RKC', 'ExtSTS+Heun-Euler+RKL', 'ExtSTS+Giraldo+RKL', 'PIROCK'])

if (Plot_RD):
    if (Plot_Fixed):
        data=pd.read_excel('RxDiff2D-fixed.xlsx')
        make_convergence_comparison_plot(data, 'RxDiff Convergence', 'rd2d_fixed_convergence')
        make_efficiency_comparison_plot(data, 'RxDiff Efficiency (Fixed)', 'rd2d_fixed_efficiency', plot_adv=False)
        make_runtime_efficiency_comparison_plot(data, 'RxDiff Runtime Efficiency (Fixed)', 'rd2d_fixed_runtime_efficiency')
    if (Plot_Adaptive):
        data=pd.read_excel('RxDiff2D-adapt.xlsx')
        make_accuracy_comparison_plot(data, 'RxDiff Accuracy', 'rd2d_adaptive_accuracy')
        make_efficiency_comparison_plot(data, 'RxDiff Efficiency', 'rd2d_adaptive_efficiency', plot_adv=False)
        make_runtime_efficiency_comparison_plot(data, 'RxDiff Runtime Efficiency', 'rd2d_adaptive_runtime_efficiency')

# display plots
#plt.show()
