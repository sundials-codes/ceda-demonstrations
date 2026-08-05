#!/usr/bin/env python3
#------------------------------------------------------------
# Programmer(s):  Daniel R. Reynolds @ UMBC
#------------------------------------------------------------
# Copyright (c) 2026, University of Maryland Baltimore County
# All rights reserved.
# For details, see the LICENSE file.
#------------------------------------------------------------

# imports
import argparse
import os
import warnings
from pathlib import Path
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
Plot_Fixed = True
Plot_Adaptive = True
Unknown_ExtSTS_Warnings = set()

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

def legend_method_name(method):
    """Return the display name for a method in plot legends."""
    method_names = {
        'ExtSTS+Giraldo': 'Giraldo-ExtSTS',
        'ExtSTS+ERK22a': 'ERK22a-ExtSTS',
        'ExtSTS+ESDIRK34a': 'ESDIRK324a-ExtSTS',
        'Giraldo-ARK21': 'Giraldo-ARK',
        'ExtSTS+SSP32': 'SSP32-ExtSTS',
        'Giraldo-DIRK21': 'Giraldo-DIRK',
    }
    return method_names.get(method, method)

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
        return 'o', 'C6', ls
    else:
        return 's', 'C6', ls

def extsts_line_style(extsts,sts,implicitrx):
    """Return the marker, color, and line style for plotting the extended STS method type and
       STS method with the given IDs."""
    if (implicitrx):
        ls = '--'
    else:
        ls = '-'
    if (extsts == 'ARS'):
        if (sts == 'RKL'):
            return 'o', 'C2', ls
        else:
            return 's', 'C2', ls
    elif (extsts == 'Giraldo'):
        if (sts == 'RKL'):
            return 'o', 'C3', ls
        else:
            return 's', 'C3', ls
    elif (extsts == 'Ralston'):
        if (sts == 'RKL'):
            return 'o', 'C4', ls
        else:
            return 's', 'C4', ls
    elif (extsts == 'SSPSDIRK2'):
        if (sts == 'RKL'):
            return 'o', 'C5', ls
        else:
            return 's', 'C5', ls
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
            return 'o', 'C4', ls
        else:
            return 's', 'C4', ls
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
        if extsts not in Unknown_ExtSTS_Warnings:
            print(f"Warning: unknown extsts type '{extsts}', using fallback marker/color.")
            Unknown_ExtSTS_Warnings.add(extsts)
        if (sts == 'RKL'):
            return 'o', 'k', ls
        return 's', 'k', ls

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
                        if (len(intdata['ststype'].unique()) > 1):
                            ststxt = '+' + sts
                        else:
                            ststxt = ''
                        ltext = '%s+%s%s%s' % (integrator,extsts,ststxt,rxtxt)
                        rate = ' (rate = %.2f)' % (medrate)
                        m,c,l = extsts_line_style(extsts,sts,rxtype)
                        DoPlot = True
                        if (integrators is not None):
                            if ltext not in integrators:
                                DoPlot = False
                        ltext = '%s+%s%s' % (integrator,extsts,rxtxt)
                        display_ltext = '%s%s' % (legend_method_name('%s+%s' % (integrator,extsts)),rxtxt)
                        if DoPlot:
                            ax1.loglog(stepsize, accuracy, marker=m, color=c, linestyle=l, label=display_ltext+rate)

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
                    ax1.loglog(stepsize, accuracy, marker='o', color='k', linestyle=ls, label=legend_method_name(ltext)+rate)

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
                    if (len(intdata['ststype'].unique()) > 1):
                        ststxt = '+' + sts
                    else:
                        ststxt = ''
                    ltext = '%s%s%s' % (integrator,ststxt,rxtxt)
                    rate = ' (rate = %.2f)' % (medrate)
                    m,c,l = strang_line_style(sts,rxtype)
                    DoPlot = True
                    if (integrators is not None):
                        if ltext not in integrators:
                            DoPlot = False
                    ltext = '%s%s' % (integrator,rxtxt)
                    if DoPlot:
                        ax1.loglog(stepsize, accuracy, marker=m, color=c, linestyle=l, label=legend_method_name(ltext)+rate)

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
                    display_ltext = '%s%s' % (legend_method_name(ark_table_name(table_id)),rxtxt)
                    rate = ' (rate = %.2f)' % (medrate)
                    m,c,l = rk_line_style(table_id,rxtype)
                    DoPlot = True
                    if (integrators is not None):
                        if ltext not in integrators:
                            DoPlot = False
                    if DoPlot:
                        ax1.loglog(stepsize, accuracy, marker=m, color=c, linestyle=l, label=display_ltext+rate)

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
                        if (len(intdata['ststype'].unique()) > 1):
                            ststxt = '+' + sts
                        else:
                            ststxt = ''
                        ltext = '%s+%s%s%s' % (integrator,extsts,ststxt,rxtxt)
                        m,c,l = extsts_line_style(extsts,sts,rxtype)
                        DoPlot = True
                        if (integrators is not None):
                            if ltext not in integrators:
                                DoPlot = False
                        ltext = '%s+%s%s' % (integrator,extsts,rxtxt)
                        display_ltext = '%s%s' % (legend_method_name('%s+%s' % (integrator,extsts)),rxtxt)
                        if DoPlot:
                            ax1.loglog(runtime, accuracy, marker=m, color=c, linestyle=l, label=display_ltext)

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
                    ax1.loglog(runtime, accuracy, marker='s', color='k', linestyle=ls, label=legend_method_name(ltext))

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
                    if (len(intdata['ststype'].unique()) > 1):
                        ststxt = '+' + sts
                    else:
                        ststxt = ''
                    ltext = '%s%s%s' % (integrator,ststxt,rxtxt)
                    m,c,l = strang_line_style(sts,rxtype)
                    DoPlot = True
                    if (integrators is not None):
                        if ltext not in integrators:
                            DoPlot = False
                    ltext = '%s%s' % (integrator,rxtxt)
                    if DoPlot:
                        ax1.loglog(runtime, accuracy, marker=m, color=c, linestyle=l, label=legend_method_name(ltext))

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
                    display_ltext = '%s%s' % (legend_method_name(ark_table_name(table_id)),rxtxt)
                    m,c,l = rk_line_style(table_id,rxtype)
                    DoPlot = True
                    if (integrators is not None):
                        if ltext not in integrators:
                            DoPlot = False
                    if DoPlot:
                        ax1.loglog(runtime, accuracy, marker=m, color=c, linestyle=l, label=display_ltext)

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
                        if (len(intdata['ststype'].unique()) > 1):
                            ststxt = '+' + sts
                        else:
                            ststxt = ''
                        ltext = '%s+%s%s%s' % (integrator,extsts,ststxt,rxtxt)
                        m,c,l = extsts_line_style(extsts,sts,rxtype)
                        DoPlot = True
                        if (integrators is not None):
                            if ltext not in integrators:
                                DoPlot = False
                        ltext = '%s+%s%s' % (integrator,extsts,rxtxt)
                        display_ltext = '%s%s' % (legend_method_name('%s+%s' % (integrator,extsts)),rxtxt)
                        if DoPlot:
                            ax1.loglog(rtol, accuracy, marker=m, color=c, linestyle=l, label=display_ltext)

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
                    ax1.loglog(rtol, accuracy, marker='s', color='k', linestyle=ls, label=legend_method_name(ltext))

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
                    if (len(intdata['ststype'].unique()) > 1):
                        ststxt = '+' + sts
                    else:
                        ststxt = ''
                    ltext = '%s%s%s' % (integrator,ststxt,rxtxt)
                    m,c,l = strang_line_style(sts,rxtype)
                    DoPlot = True
                    if (integrators is not None):
                        if ltext not in integrators:
                            DoPlot = False
                    ltext = '%s%s' % (integrator,rxtxt)
                    if DoPlot:
                        ax1.loglog(rtol, accuracy, marker=m, color=c, linestyle=l, label=legend_method_name(ltext))

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
                    display_ltext = '%s%s' % (legend_method_name(ark_table_name(table_id)),rxtxt)
                    m,c,l = rk_line_style(table_id,rxtype)
                    DoPlot = True
                    if (integrators is not None):
                        if ltext not in integrators:
                            DoPlot = False
                    if DoPlot:
                        ax1.loglog(rtol, accuracy, marker=m, color=c, linestyle=l, label=display_ltext)

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


DATA_DIR = Path(__file__).resolve().parent / '..'
PLOT_DIR = Path(__file__).resolve().parent / '../plots'
BCTYPE_CHOICES = ('all', 'periodic', 'stationary')
PLOT_MODE_CHOICES = ('all', 'final', 'rkc-vs-rkl')
ADR_FINAL_INTEGRATORS = ['Giraldo-ARK21', 'ExtSTS+Giraldo+RKL', 'ExtSTS+SSP32+RKL', 'PIROCK', 'Strang+RKL']
ADR_RKC_INTEGRATORS = ['ExtSTS+Giraldo+RKC', 'ExtSTS+Giraldo+RKL', 'ExtSTS+SSP32+RKC', 'ExtSTS+SSP32+RKL', 'Strang+RKC', 'Strang+RKL']


def parse_args():
    parser = argparse.ArgumentParser(description='Generate ADR 2D plots from unified spreadsheets.')
    parser.add_argument('--bctype', choices=BCTYPE_CHOICES, default='all',
                        help='Boundary condition subset to plot (default: all).')
    parser.add_argument('--plot-mode', choices=PLOT_MODE_CHOICES, default='all',
                        help='Plot subset to generate (default: all).')
    return parser.parse_args()


def selected_values(selected, all_values):
    if selected == 'all':
        return all_values
    return (selected,)


def data_path(stem, bctype):
    return DATA_DIR / f'{stem}_{bctype}.xlsx'


def load_data(stem, bctype):
    path = data_path(stem, bctype)
    if not path.exists():
        raise FileNotFoundError(f'Missing required input file: {path}')
    return pd.read_excel(path)


def unique_sorted(data, column):
    return np.sort(np.unique(data[column].to_numpy(dtype=float)))


def subset(data, dvalue, bvalue):
    dmask = np.isclose(data['d'].to_numpy(dtype=float), dvalue)
    out = data[dmask]
    bmask = np.isclose(out['B'].to_numpy(dtype=float), bvalue)
    return out[bmask]


def integrators_for(plot_mode, final_list, rkc_list):
    if plot_mode == 'rkc-vs-rkl':
        return rkc_list
    return final_list


def generate_for_case(bctype, plot_mode):

    if (Plot_ADR):
        integrators = integrators_for(plot_mode, ADR_FINAL_INTEGRATORS, ADR_RKC_INTEGRATORS)
        if (Plot_Fixed):
            data = load_data('data/AdvDiffRx2D-fixed', bctype)
            data = data[data["ReturnCode"] == 0]
            if plot_mode == 'rkc-vs-rkl':
                d, bval = 1e-1, 3.0
                ddata = subset(data, d, bval)
                if len(ddata) > 0:
                    make_convergence_comparison_plot(ddata, r'AdvDiffRx Convergence ($d=%.2f$, $B=%.1e$)' % (d, bval), '%s/adr2D_%s_fixed_convergence_RKLvRKC' % (PLOT_DIR, bctype), integrators=integrators)
                    make_runtime_efficiency_comparison_plot(ddata, r'AdvDiffRx Runtime Efficiency ($d=%.2f$, $B=%.1e$, Fixed)' % (d, bval), '%s/adr2D_%s_fixed_runtime_efficiency_RKLvRKC' % (PLOT_DIR, bctype), integrators=integrators)
            else:
                for d in unique_sorted(data, 'd'):
                    for bval in unique_sorted(data[np.isclose(data['d'].to_numpy(dtype=float), d)], 'B'):
                        ddata = subset(data, d, bval)
                        make_convergence_comparison_plot(ddata, r'AdvDiffRx Convergence ($d=%.2f$, $B=%.1e$)' % (d, bval), '%s/adr2D_%s_fixed_convergence_d%.2f_B%.1e' % (PLOT_DIR, bctype, d, bval), integrators=integrators)
                        make_runtime_efficiency_comparison_plot(ddata, r'AdvDiffRx Runtime Efficiency ($d=%.2f$, $B=%.1e$, Fixed)' % (d, bval), '%s/adr2D_%s_fixed_runtime_efficiency_d%.2f_B%.1e' % (PLOT_DIR, bctype, d, bval), integrators=integrators)
        if (Plot_Adaptive):
            data = load_data('data/AdvDiffRx2D-adapt', bctype)
            data = data[data["ReturnCode"] == 0]
            if plot_mode == 'rkc-vs-rkl':
                d, bval = 1e-1, 3.0
                ddata = subset(data, d, bval)
                if len(ddata) > 0:
                    make_accuracy_comparison_plot(ddata, r'AdvDiffRx Accuracy ($d=%.2f$, $B=%.1e$)' % (d, bval), '%s/adr2D_%s_adaptive_accuracy_RKLvRKC' % (PLOT_DIR, bctype), integrators=integrators)
                    make_runtime_efficiency_comparison_plot(ddata, r'AdvDiffRx Runtime Efficiency ($d=%.2f$, $B=%.1e$)' % (d, bval), '%s/adr2D_%s_adaptive_runtime_efficiency_RKLvRKC' % (PLOT_DIR, bctype), integrators=integrators)
            else:
                for d in unique_sorted(data, 'd'):
                    for bval in unique_sorted(data[np.isclose(data['d'].to_numpy(dtype=float), d)], 'B'):
                        ddata = subset(data, d, bval)
                        make_accuracy_comparison_plot(ddata, r'AdvDiffRx Accuracy ($d=%.2f$, $B=%.1e$)' % (d, bval), '%s/adr2D_%s_adaptive_accuracy_d%.2f_B%.1e' % (PLOT_DIR, bctype, d, bval), integrators=integrators)
                        make_runtime_efficiency_comparison_plot(ddata, r'AdvDiffRx Runtime Efficiency ($d=%.2f$, $B=%.1e$)' % (d, bval), '%s/adr2D_%s_adaptive_runtime_efficiency_d%.2f_B%.1e' % (PLOT_DIR, bctype, d, bval), integrators=integrators)


def main():
    args = parse_args()
    os.makedirs(PLOT_DIR, exist_ok=True)
    warnings.simplefilter("ignore", RuntimeWarning)
    warnings.simplefilter("ignore", UserWarning)
    for bctype in selected_values(args.bctype, ('periodic', 'stationary')):
        for plot_mode in selected_values(args.plot_mode, ('final', 'rkc-vs-rkl')):
            generate_for_case(bctype, plot_mode)
    print('Plot generation complete. Plots saved in:', PLOT_DIR)

if __name__ == '__main__':
    main()
