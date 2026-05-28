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
        return 'Heun-Euler-ERK21'
    elif (table_id == 5):
        return 'SSP-SDIRK21'
    elif (table_id == 6):
        return 'Giraldo-DIRK21'
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
        return 'x', 'C7', ls
    elif (table_id == 6):
        return '+', 'C8', ls
    else:
        raise ValueError('Unknown table ID: %d' % table_id)

def strang_line_style(sts,implicitrx):
    """Return the marker, color, and line style for plotting the Strang + STS
       method."""
    if (implicitrx):
        ls = '--'
    else:
        ls = '-'
    if (sts == 'RKL'):
        return 'x', 'C6', ls
    else:
        return '+', 'C6', ls

def extsts_line_style(extsts,sts,implicitrx):
    """Return the marker, color, and line style for plotting the extended STS method type and
       STS method with the given IDs."""
    if (implicitrx):
        ls = '--'
    else:
        ls = '-'
    if (extsts == 'ARS'):
        if (sts == 'RKL'):
            return 'x', 'C2', ls
        else:
            return '+', 'C2', ls
    elif (extsts == 'Giraldo'):
        if (sts == 'RKL'):
            return 'x', 'C3', ls
        else:
            return '+', 'C3', ls
    elif (extsts == 'Ralston'):
        if (sts == 'RKL'):
            return 'x', 'C4', ls
        else:
            return '+', 'C4', ls
    elif (extsts == 'SSPSDIRK2'):
        if (sts == 'RKL'):
            return 'x', 'C5', ls
        else:
            return '+', 'C5', ls
    elif (extsts == 'IRK21a'):
        if (sts == 'RKL'):
            return 'o', 'C6', ls
        else:
            return 's', 'C6', ls
    elif (extsts == 'ESDIRK34a'):
        if (sts == 'RKL'):
            return 'o', 'C7', ls
        else:
            return 's', 'C7', ls
    elif (extsts == 'ERK22a'):
        if (sts == 'RKL'):
            return 'o', 'C8', ls
        else:
            return 's', 'C8', ls
    elif (extsts == 'ERK22b'):
        if (sts == 'RKL'):
            return 'o', 'C9', ls
        else:
            return 's', 'C9', ls
    elif (extsts == 'MERK21'):
        if (sts == 'RKL'):
            return 'o', 'C10', ls
        else:
            return 's', 'C10', ls
    elif (extsts == 'MERK32'):
        if (sts == 'RKL'):
            return 'o', 'C11', ls
        else:
            return 's', 'C11', ls
    elif (extsts == 'MRISR21'):
        if (sts == 'RKL'):
            return 'o', 'C12', ls
        else:
            return 's', 'C12', ls
    elif (extsts == 'Heun-Euler'):
        if (sts == 'RKL'):
            return 'x', 'C4', ls
        else:
            return '+', 'C4', ls
    elif (extsts == 'SSP22'):
        if (sts == 'RKL'):
            return 'o', 'C13', ls
        else:
            return 's', 'C13', ls
    elif (extsts == 'SSP32'):
        if (sts == 'RKL'):
            return 'o', 'C14', ls
        else:
            return 's', 'C14', ls
    elif (extsts == 'SSP42'):
        if (sts == 'RKL'):
            return 'o', 'C15', ls
        else:
            return 's', 'C15', ls
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
                    for rxtype in stsdata['implicitrx'].unique():
                        pdata = stsdata[stsdata['implicitrx'] == rxtype]
                        stepsize = pdata['fixedh'].to_numpy()
                        accuracy = pdata['Accuracy'].to_numpy()
                        rates = np.log(accuracy[1:] / accuracy[:-1]) / np.log(stepsize[1:] / stepsize[:-1])
                        medrate = np.nanmedian(rates)
                        if (len(extstsdata['implicitrx'].unique()) > 1):
                            if (rxtype):
                                rxtxt = ', impl-R'
                            else:
                                rxtxt = ', expl-R'
                        else:
                            rxtxt = ''
                        ltext = '%s+%s+%s%s' % (integrator,extsts,sts,rxtxt)
                        rate = ' (rate = %.2f)' % (medrate)
                        m,c,l = extsts_line_style(extsts,sts,rxtype)
                        DoPlot = True
                        if (integrators is not None):
                            if ltext not in integrators:
                                DoPlot = False
                        if DoPlot:
                            ax1.loglog(stepsize, accuracy, marker=m, color=c, linestyle=l, label=ltext+rate)

        elif (integrator == 'PIROCK'):
            for rxtype in intdata['implicitrx'].unique():
                pdata = intdata[intdata['implicitrx'] == rxtype]
                stepsize = pdata['fixedh'].to_numpy()
                accuracy = pdata['Accuracy'].to_numpy()
                rates = np.log(accuracy[1:] / accuracy[:-1]) / np.log(stepsize[1:] / stepsize[:-1])
                medrate = np.nanmedian(rates)
                if (len(intdata['implicitrx'].unique()) > 1):
                    if (rxtype):
                        rxtxt = ', impl-R'
                        ls = '--'
                    else:
                        rxtxt = ', expl-R'
                        ls = '-'
                else:
                    rxtxt = ''
                    ls = '-'
                ltext = '%s%s' % (integrator,rxtxt)
                rate = ' (rate = %.2f)' % (medrate)
                DoPlot = True
                if (integrators is not None):
                    if ltext not in integrators:
                        DoPlot = False
                if DoPlot:
                    ax1.loglog(stepsize, accuracy, marker='.', color='k', linestyle=ls, label=ltext+rate)

        elif (integrator == 'Strang'):
            for sts in intdata['ststype'].unique():
                stsdata = extstsdata.groupby(['ststype',]).get_group((sts,))
                for rxtype in stsdata['implicitrx'].unique():
                    pdata = stsdata[stsdata['implicitrx'] == rxtype]
                    stepsize = pdata['fixedh'].to_numpy()
                    accuracy = pdata['Accuracy'].to_numpy()
                    rates = np.log(accuracy[1:] / accuracy[:-1]) / np.log(stepsize[1:] / stepsize[:-1])
                    medrate = np.nanmedian(rates)
                    if (len(intdata['implicitrx'].unique()) > 1):
                        if (rxtype):
                            rxtxt = ', impl-R'
                        else:
                            rxtxt = ', expl-R'
                    else:
                        rxtxt = ''
                    ltext = '%s+%s%s' % (integrator,sts,rxtxt)
                    rate = ' (rate = %.2f)' % (medrate)
                    m,c,l = strang_line_style(sts,rxtype)
                    DoPlot = True
                    if (integrators is not None):
                        if ltext not in integrators:
                            DoPlot = False
                    if DoPlot:
                        ax1.loglog(stepsize, accuracy, marker=m, color=c, linestyle=l, label=ltext+rate)

        else:
            for table_id in intdata['table_id'].unique():
                tabledata = intdata.groupby(['table_id',]).get_group((table_id,))
                for rxtype in tabledata['implicitrx'].unique():
                    pdata = tabledata[tabledata['implicitrx'] == rxtype]
                    stepsize = pdata['fixedh'].to_numpy()
                    accuracy = pdata['Accuracy'].to_numpy()
                    rates = np.log(accuracy[1:] / accuracy[:-1]) / np.log(stepsize[1:] / stepsize[:-1])
                    medrate = np.nanmedian(rates)
                    if (len(tabledata['implicitrx'].unique()) > 1):
                        if (rxtype):
                            rxtxt = ', impl-R'
                        else:
                            rxtxt = ', expl-R'
                    else:
                        rxtxt = ''
                    ltext = '%s%s' % (ark_table_name(table_id),rxtxt)
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
                    for rxtype in stsdata['implicitrx'].unique():
                        pdata = stsdata[stsdata['implicitrx'] == rxtype]
                        accuracy = pdata['Accuracy'].to_numpy()
                        diffevals = pdata['DiffEvals'].to_numpy()
                        if (plot_adv):
                            advevals = pdata['AdvEvals'].to_numpy()
                        if (plot_rx):
                            rxevals = pdata['RxEvals'].to_numpy()
                            if (np.sum(rxevals) == 0):
                                rxevals = pdata['AdvEvals'].to_numpy()
                        if (len(extstsdata['implicitrx'].unique()) > 1):
                            if (rxtype):
                                rxtxt = ', impl-R'
                            else:
                                rxtxt = ', expl-R'
                        else:
                            rxtxt = ''
                        ltext = '%s+%s+%s%s' % (integrator,extsts,sts,rxtxt)
                        m,c,l = extsts_line_style(extsts,sts,rxtype)
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
            for rxtype in intdata['implicitrx'].unique():
                pdata = intdata[intdata['implicitrx'] == rxtype]
                accuracy = pdata['Accuracy'].to_numpy()
                diffevals = pdata['DiffEvals'].to_numpy()
                if (plot_adv):
                    advevals = pdata['AdvEvals'].to_numpy()
                if (plot_rx):
                    rxevals = pdata['RxEvals'].to_numpy()
                    if (np.sum(rxevals) == 0):
                        rxevals = pdata['AdvEvals'].to_numpy()
                if (len(intdata['implicitrx'].unique()) > 1):
                    if (rxtype):
                        rxtxt = ', impl-R'
                        ls = '--'
                    else:
                        rxtxt = ', expl-R'
                        ls = '-'
                else:
                    rxtxt = ''
                    ls = '-'
                ltext = '%s%s' % (integrator,rxtxt)
                DoPlot = True
                if (integrators is not None):
                    if ltext not in integrators:
                        DoPlot = False
                if DoPlot:
                    ax_diff.loglog(diffevals, accuracy, marker='.', color='k', linestyle=ls, label=ltext)
                    if (plot_adv):
                        ax_adv.loglog(advevals, accuracy, marker='.', color='k', linestyle=ls, label=ltext)
                    if (plot_rx):
                        ax_rx.loglog(rxevals, accuracy, marker='.', color='k', linestyle=ls, label=ltext)

        elif (integrator == 'Strang'):
            for sts in intdata['ststype'].unique():
                stsdata = extstsdata.groupby(['ststype',]).get_group((sts,))
                for rxtype in stsdata['implicitrx'].unique():
                    pdata = stsdata[stsdata['implicitrx'] == rxtype]
                    accuracy = pdata['Accuracy'].to_numpy()
                    diffevals = pdata['DiffEvals'].to_numpy()
                    if (plot_adv):
                        advevals = pdata['AdvEvals'].to_numpy()
                    if (plot_rx):
                        rxevals = pdata['RxEvals'].to_numpy()
                    if (len(intdata['implicitrx'].unique()) > 1):
                        if (rxtype):
                            rxtxt = ', impl-R'
                        else:
                            rxtxt = ', expl-R'
                    else:
                        rxtxt = ''
                    ltext = '%s+%s%s' % (integrator,sts,rxtxt)
                    m,c,l = strang_line_style(sts,rxtype)
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
                tabledata = intdata.groupby(['table_id',]).get_group((table_id,))
                for rxtype in tabledata['implicitrx'].unique():
                    pdata = tabledata[tabledata['implicitrx'] == rxtype]
                    accuracy = pdata['Accuracy'].to_numpy()
                    diffevals = pdata['DiffEvals'].to_numpy()
                    if (plot_adv):
                        advevals = pdata['AdvEvals'].to_numpy()
                    if (plot_rx):
                        rxevals = pdata['RxEvals'].to_numpy()
                    if (len(tabledata['implicitrx'].unique()) > 1):
                        if (rxtype):
                            rxtxt = ', impl-R'
                        else:
                            rxtxt = ', expl-R'
                    else:
                        rxtxt = ''
                    ltext = '%s%s' % (ark_table_name(table_id),rxtxt)
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

    # determine average runtime per RHS call over full set of non-PIROCK experiments
    AdvRhsTime = 0.0
    DiffRhsTime = 0
    RxRhsTime = 0.0
    AdvRhsNum = 0
    DiffRhsNum = 0.0
    RxRhsNum = 0
    for integrator in data['inttype'].unique():
        intdata = data.groupby(['inttype',]).get_group((integrator,))
        if (integrator != 'PIROCK'):
            AdvRhsTime += np.nansum(intdata['AdvTime'].to_numpy())
            DiffRhsTime += np.nansum(intdata['DiffTime'].to_numpy())
            RxRhsTime += np.nansum(intdata['RxTime'].to_numpy())
            AdvRhsNum += np.nansum(intdata['AdvEvals'].to_numpy())
            DiffRhsNum += np.nansum(intdata['DiffEvals'].to_numpy())
            numRx = np.nansum(intdata['RxEvals'].to_numpy())
            if (numRx == 0):
                numRx = np.nansum(intdata['AdvEvals'].to_numpy())
            RxRhsNum += numRx
    AdvRhsMean = (AdvRhsTime/AdvRhsNum) if (AdvRhsNum > 0) else 0
    DiffRhsMean = (DiffRhsTime/DiffRhsNum) if (DiffRhsNum > 0) else 0
    RxRhsMean = (RxRhsTime/RxRhsNum) if (RxRhsNum > 0) else 0

    # generate plots
    for integrator in data['inttype'].unique():
        intdata = data.groupby(['inttype',]).get_group((integrator,))

        if (integrator == 'ExtSTS'):
            for extsts in intdata['extststype'].unique():
                extstsdata = intdata.groupby(['extststype',]).get_group((extsts,))
                for sts in extstsdata['ststype'].unique():
                    stsdata = extstsdata.groupby(['ststype',]).get_group((sts,))
                    for rxtype in stsdata['implicitrx'].unique():
                        pdata = stsdata[stsdata['implicitrx'] == rxtype]
                        accuracy = pdata['Accuracy'].to_numpy()
                        numRx = np.sum(pdata['RxEvals'].to_numpy())
                        if (numRx == 0):
                            runtime = (AdvRhsMean * pdata['AdvEvals'].to_numpy() +
                                    DiffRhsMean * pdata['DiffEvals'].to_numpy() +
                                    RxRhsMean * pdata['AdvEvals'].to_numpy())
                        else:
                            runtime = (AdvRhsMean * pdata['AdvEvals'].to_numpy() +
                                    DiffRhsMean * pdata['DiffEvals'].to_numpy() +
                                    RxRhsMean * pdata['RxEvals'].to_numpy())
                        if (len(extstsdata['implicitrx'].unique()) > 1):
                            if (rxtype):
                                rxtxt = ', impl-R'
                            else:
                                rxtxt = ', expl-R'
                        else:
                            rxtxt = ''
                        ltext = '%s+%s+%s%s' % (integrator,extsts,sts,rxtxt)
                        m,c,l = extsts_line_style(extsts,sts,rxtype)
                        DoPlot = True
                        if (integrators is not None):
                            if ltext not in integrators:
                                DoPlot = False
                        if DoPlot:
                            ax1.loglog(runtime, accuracy, marker=m, color=c, linestyle=l, label=ltext)

        elif (integrator == 'PIROCK'):
            for rxtype in intdata['implicitrx'].unique():
                pdata = intdata[intdata['implicitrx'] == rxtype]
                accuracy = pdata['Accuracy'].to_numpy()
                numRx = np.sum(pdata['RxEvals'].to_numpy())
                if (numRx == 0):
                    runtime = (AdvRhsMean * pdata['AdvEvals'].to_numpy() +
                            DiffRhsMean * pdata['DiffEvals'].to_numpy() +
                            RxRhsMean * pdata['AdvEvals'].to_numpy())
                else:
                    runtime = (AdvRhsMean * pdata['AdvEvals'].to_numpy() +
                            DiffRhsMean * pdata['DiffEvals'].to_numpy() +
                            RxRhsMean * pdata['RxEvals'].to_numpy())
                if (len(intdata['implicitrx'].unique()) > 1):
                    if (rxtype):
                        rxtxt = ', impl-R'
                        ls = '--'
                    else:
                        rxtxt = ', expl-R'
                        ls = '-'
                else:
                    rxtxt = ''
                    ls = '-'
                ltext = '%s%s' % (integrator,rxtxt)
                DoPlot = True
                if (integrators is not None):
                    if ltext not in integrators:
                        DoPlot = False
                if DoPlot:
                    ax1.loglog(runtime, accuracy, marker='.', color='k', linestyle=ls, label=ltext)

        elif (integrator == 'Strang'):
            for sts in intdata['ststype'].unique():
                stsdata = extstsdata.groupby(['ststype',]).get_group((sts,))
                for rxtype in stsdata['implicitrx'].unique():
                    pdata = stsdata[stsdata['implicitrx'] == rxtype]
                    accuracy = pdata['Accuracy'].to_numpy()
                    numRx = np.sum(pdata['RxEvals'].to_numpy())
                    if (numRx == 0):
                        runtime = (AdvRhsMean * pdata['AdvEvals'].to_numpy() +
                                DiffRhsMean * pdata['DiffEvals'].to_numpy() +
                                RxRhsMean * pdata['AdvEvals'].to_numpy())
                    else:
                        runtime = (AdvRhsMean * pdata['AdvEvals'].to_numpy() +
                                DiffRhsMean * pdata['DiffEvals'].to_numpy() +
                                RxRhsMean * pdata['RxEvals'].to_numpy())
                    if (len(intdata['implicitrx'].unique()) > 1):
                        if (rxtype):
                            rxtxt = ', impl-R'
                        else:
                            rxtxt = ', expl-R'
                    else:
                        rxtxt = ''
                    ltext = '%s+%s%s' % (integrator,sts,rxtxt)
                    m,c,l = strang_line_style(sts,rxtype)
                    DoPlot = True
                    if (integrators is not None):
                        if ltext not in integrators:
                            DoPlot = False
                    if DoPlot:
                        ax1.loglog(runtime, accuracy, marker=m, color=c, linestyle=l, label=ltext)

        else:
            for table_id in intdata['table_id'].unique():
                tabledata = intdata.groupby(['table_id',]).get_group((table_id,))
                for rxtype in tabledata['implicitrx'].unique():
                    pdata = tabledata[tabledata['implicitrx'] == rxtype]
                    accuracy = pdata['Accuracy'].to_numpy()
                    numRx = np.sum(pdata['RxEvals'].to_numpy())
                    if (numRx == 0):
                        runtime = (AdvRhsMean * pdata['AdvEvals'].to_numpy() +
                                   DiffRhsMean * pdata['DiffEvals'].to_numpy() +
                                   RxRhsMean * pdata['AdvEvals'].to_numpy())
                    else:
                        runtime = (AdvRhsMean * pdata['AdvEvals'].to_numpy() +
                                   DiffRhsMean * pdata['DiffEvals'].to_numpy() +
                                   RxRhsMean * pdata['RxEvals'].to_numpy())
                    if (len(tabledata['implicitrx'].unique()) > 1):
                        if (rxtype):
                            rxtxt = ', impl-R'
                        else:
                            rxtxt = ', expl-R'
                    else:
                        rxtxt = ''
                    ltext = '%s%s' % (ark_table_name(table_id),rxtxt)
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
                    for rxtype in stsdata['implicitrx'].unique():
                        pdata = stsdata[stsdata['implicitrx'] == rxtype]
                        rtol = pdata['rtol'].to_numpy()
                        accuracy = pdata['Accuracy'].to_numpy()
                        if (len(extstsdata['implicitrx'].unique()) > 1):
                            if (rxtype):
                                rxtxt = ', impl-R'
                            else:
                                rxtxt = ', expl-R'
                        else:
                            rxtxt = ''
                        ltext = '%s+%s+%s%s' % (integrator,extsts,sts,rxtxt)
                        m,c,l = extsts_line_style(extsts,sts,rxtype)
                        DoPlot = True
                        if (integrators is not None):
                            if ltext not in integrators:
                                DoPlot = False
                        if DoPlot:
                            ax1.loglog(rtol, accuracy, marker=m, color=c, linestyle=l, label=ltext)

        elif (integrator == 'PIROCK'):
            for rxtype in intdata['implicitrx'].unique():
                pdata = intdata[intdata['implicitrx'] == rxtype]
                rtol = pdata['rtol'].to_numpy()
                accuracy = pdata['Accuracy'].to_numpy()
                if (len(intdata['implicitrx'].unique()) > 1):
                    if (rxtype):
                        rxtxt = ', impl-R'
                        ls = '--'
                    else:
                        rxtxt = ', expl-R'
                        ls = '-'
                else:
                    rxtxt = ''
                    ls = '-'
                ltext = '%s%s' % (integrator,rxtxt)
                DoPlot = True
                if (integrators is not None):
                    if ltext not in integrators:
                        DoPlot = False
                if DoPlot:
                    ax1.loglog(rtol, accuracy, marker='.', color='k', linestyle=ls, label=ltext)

        elif (integrator == 'Strang'):
            for sts in intdata['ststype'].unique():
                stsdata = extstsdata.groupby(['ststype',]).get_group((sts,))
                for rxtype in stsdata['implicitrx'].unique():
                    pdata = stsdata[stsdata['implicitrx'] == rxtype]
                    rtol = pdata['rtol'].to_numpy()
                    accuracy = pdata['Accuracy'].to_numpy()
                    if (len(intdata['implicitrx'].unique()) > 1):
                        if (rxtype):
                            rxtxt = ', impl-R'
                        else:
                            rxtxt = ', expl-R'
                    else:
                        rxtxt = ''
                    ltext = '%s+%s%s' % (integrator,sts,rxtxt)
                    m,c,l = strang_line_style(sts,rxtype)
                    DoPlot = True
                    if (integrators is not None):
                        if ltext not in integrators:
                            DoPlot = False
                    if DoPlot:
                        ax1.loglog(rtol, accuracy, marker=m, color=c, linestyle=l, label=ltext)

        else:
            for table_id in intdata['table_id'].unique():
                tabledata = intdata.groupby(['table_id',]).get_group((table_id,))
                for rxtype in tabledata['implicitrx'].unique():
                    pdata = tabledata[tabledata['implicitrx'] == rxtype]
                    rtol = pdata['rtol'].to_numpy()
                    accuracy = pdata['Accuracy'].to_numpy()
                    if (len(tabledata['implicitrx'].unique()) > 1):
                        if (rxtype):
                            rxtxt = ', impl-R'
                        else:
                            rxtxt = ', expl-R'
                    else:
                        rxtxt = ''
                    ltext = '%s%s' % (ark_table_name(table_id),rxtxt)
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
    dvals=[1e-2, 1e-1]
    Bvals=[3.0, 3e1, 3e2]
    if (Plot_Fixed):
        integrators=None
        #integrators=['ExtSTS+ARS+RKC, expl-R', 'ExtSTS+Giraldo+RKC, expl-R', 'ExtSTS+SSP22+RKC', 'ExtSTS+SSP32+RKC', 'ExtSTS+SSP42+RKC', 'PIROCK, expl-R', 'Strang+RKC, expl-R']
        data = pd.read_excel('AdvDiffRx2D-fixed.xlsx')
        data = data[data["ReturnCode"] == 0]
        for d in dvals:
            for B in Bvals:
                ddata = data[data['d'] == d]
                ddata = ddata[ddata['B'] == B]
                make_convergence_comparison_plot(ddata, r'AdvDiffRx Convergence ($d=%.2f$, $B=%.1e$)' % (d, B), 'adr2d_fixed_convergence_d%.2f_B%.1e' % (d, B), integrators=integrators)
                make_efficiency_comparison_plot(ddata, r'AdvDiffRx Efficiency ($d=%.2f$, $B=%.1e$, Fixed)' % (d, B), 'adr2d_fixed_efficiency_d%.2f_B%.1e' % (d, B), integrators=integrators)
                make_runtime_efficiency_comparison_plot(ddata, r'AdvDiffRx Runtime Efficiency ($d=%.2f$, $B=%.1e$, Fixed)' % (d, B), 'adr2d_fixed_runtime_efficiency_d%.2f_B%.1e' % (d, B), integrators=integrators)

    if (Plot_Adaptive):
        integrators=None
        #integrators=['ARS-ARK21', 'ExtSTS+Giraldo+RKC, expl-R', 'ExtSTS+Giraldo+RKC, impl-R', 'ExtSTS+SSP22+RKC', 'ExtSTS+SSP32+RKC', 'ExtSTS+SSP42+RKC', 'PIROCK, expl-R']
        data = pd.read_excel('AdvDiffRx2D-adapt.xlsx')
        data = data[data["ReturnCode"] == 0]
        for d in dvals:
            for B in Bvals:
                ddata = data[data['d'] == d]
                ddata = ddata[ddata['B'] == B]
                make_accuracy_comparison_plot(ddata, r'AdvDiffRx Accuracy ($d=%.2f$, $B=%.1e$)' % (d, B), 'adr2d_adaptive_accuracy_d%.2f_B%.1e' % (d, B), integrators=integrators)
                make_efficiency_comparison_plot(ddata, r'AdvDiffRx Efficiency ($d=%.2f$, $B=%.1e$)' % (d, B), 'adr2d_adaptive_efficiency_d%.2f_B%.1e' % (d, B), integrators=integrators)
                make_runtime_efficiency_comparison_plot(ddata, r'AdvDiffRx Runtime Efficiency ($d=%.2f$, $B=%.1e$)' % (d, B), 'adr2d_adaptive_runtime_efficiency_d%.2f_B%.1e' % (d, B), integrators=integrators)

if (Plot_RD):
    Bvals=[2.e1, 2.e4, 2.e7]
    dvals=[0.01, 0.1]
    if (Plot_Fixed):
        integrators=None
        #integrators=['ARS, impl-R', 'Giraldo, impl-R', 'ExtSTS+ARS+RKC', 'ExtSTS+Heun-Euler+RKL', 'ExtSTS+Giraldo+RKL', 'PIROCK']
        data = pd.read_excel('RxDiff2D-fixed.xlsx')
        data = data[data["ReturnCode"] == 0]
        for d in dvals:
            for B in Bvals:
                ddata = data[data['d'] == d]
                ddata = ddata[ddata['B'] == B]
                make_convergence_comparison_plot(ddata, r'RxDiff Convergence ($d=%.2f$, $B=%.1e$)' % (d, B), 'rd2d_fixed_convergence_d%.2f_B%.1e' % (d, B), integrators=integrators)
                make_efficiency_comparison_plot(ddata, r'RxDiff Efficiency ($d=%.2f$, $B=%.1e$, Fixed)' % (d, B), 'rd2d_fixed_efficiency_d%.2f_B%.1e' % (d, B), plot_adv=False, integrators=integrators)
                make_runtime_efficiency_comparison_plot(ddata, r'RxDiff Runtime Efficiency ($d=%.2f$, $B=%.1e$, Fixed)' % (d, B), 'rd2d_fixed_runtime_efficiency_d%.2f_B%.1e' % (d, B), integrators=integrators)
    if (Plot_Adaptive):
        integrators=None
        #integrators=['ARS-ARK21', 'ExtSTS+Giraldo+RKC', 'ExtSTS+IRK21a', 'PIROCK']
        data = pd.read_excel('RxDiff2D-adapt.xlsx')
        data = data[data["ReturnCode"] == 0]
        for d in dvals:
            for B in Bvals:
                ddata = data[data['d'] == d]
                ddata = ddata[ddata['B'] == B]
                make_accuracy_comparison_plot(ddata, r'RxDiff Accuracy ($d=%.2f$, $B=%.1e$)' % (d, B), 'rd2d_adaptive_accuracy_d%.2f_B%.1e' % (d, B), integrators=integrators)
                make_efficiency_comparison_plot(ddata, r'RxDiff Efficiency ($d=%.2f$, $B=%.1e$)' % (d, B), 'rd2d_adaptive_efficiency_d%.2f_B%.1e' % (d, B), plot_adv=False, integrators=integrators)
                make_runtime_efficiency_comparison_plot(ddata, r'RxDiff Runtime Efficiency ($d=%.2f$, $B=%.1e$)' % (d, B), 'rd2d_adaptive_runtime_efficiency_d%.2f_B%.1e' % (d, B), integrators=integrators)

# display plots
#plt.show()
