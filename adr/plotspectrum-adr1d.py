#!/usr/bin/env python3
#------------------------------------------------------------
# Programmer(s):  Daniel R. Reynolds @ UMBC
#------------------------------------------------------------
# Copyright (c) 2025, University of Maryland Baltimore County
# All rights reserved.
# For details, see the LICENSE file.
#------------------------------------------------------------

# imports
import os
import numpy as np
import subprocess
import shlex
import time
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Set plot defaults: increase default font size, increase plot width, enable LaTeX rendering
plt.rc('font', size=15)
plt.rcParams['figure.figsize'] = [7.2, 4.8]
plt.rcParams['text.usetex'] = True
plt.rcParams['figure.constrained_layout.use'] = True

# flags to turn on/off certain plots
Generate_PDF = True
Generate_PNG = False
DoAdvDiffRx = True
DoAdvDiff = True
DoRxDiff = True
NumOut = 100

#####################
# utility routines

# utility routine to set C++ executable inputs for running a specific integration type
def int_method(probtype, inttype, ststype, extststype, table_id):
    flags = ""
    if (probtype == "AdvDiff"):
        flags += " --no-reaction"
    elif (probtype == "RxDiff"):
        flags += " --no-advection"
    elif (probtype == "AdvDiffRx"):
        flags += " "
    else:
        msg = """
        Error: invalid problem type
        Valid problem types are: AdvDiff, RxDiff, AdvDiffRx
        """
        raise(ValueError, msg)

    if (inttype == "ARK"):
        flags += " --integrator 1 --table_id %d" % table_id

    elif (inttype == "ERK"):
        flags += " --integrator 0"

    elif (inttype == "Strang"):
        flags += " --integrator 3"

        if (ststype == "RKC"):
            flags += " --sts_method 0"
        elif (ststype == "RKL"):
            flags += " --sts_method 1"
        else:
            msg = """
            Error: invalid sts type
            Valid choices are: RKC, RKL
            """
            raise(ValueError, msg)

    elif (inttype == "ExtSTS"):
        flags += " --integrator 2"

        if (ststype == "RKC"):
            flags += " --sts_method 0"
        elif (ststype == "RKL"):
            flags += " --sts_method 1"
        else:
            msg = """
            Error: invalid sts type
            Valid choices are: RKC, RKL
            """
            raise(ValueError, msg)

        if (extststype == "ARS"):
            flags += " --extsts_method 0"
        elif (extststype == "Giraldo"):
            flags += " --extsts_method 1"
        elif (extststype == "Ralston"):
            if (probtype != "AdvDiff"):
                raise(ValueError, "invalid problem + extsts type combination")
            flags += " --extsts_method 2"
        elif (extststype == "HeunEuler"):
            if (probtype != "AdvDiff"):
                raise(ValueError, "invalid problem + extsts type combination")
            flags += " --extsts_method 3"
        elif (extststype == "SSPSDIRK2"):
            if (probtype != "RxDiff"):
                raise(ValueError, "invalid problem + extsts type combination")
            flags += " --extsts_method 2"
        elif (extststype == "IRK21a"):
            flags += " --extsts_method -203"
        elif (extststype == "ESDIRK34a"):
            flags += " --extsts_method -204"
        elif (extststype == "ERK22a"):
            flags += " --extsts_method -211"
        elif (extststype == "ERK22b"):
            flags += " --extsts_method -212"
        elif (extststype == "MERK21"):
            flags += " --extsts_method -219"
        elif (extststype == "MERK32"):
            flags += " --extsts_method -220"
        elif (extststype == "MRISR21"):
            flags += " --extsts_method -223"
        else:
            msg = """
            Error: invalid extsts type
            Valid choices are: ARS, Giraldo, Ralston, HeunEuler, SSPSDIRK2, IRK21a, ESDIRK34a, ERK22a, ERK22b, MERK21, MERK32, MRISR21
            """
            raise(ValueError, msg)

    else:
        msg = """
        Error: invalid integrator
        Valid integrator choices are: ARK, ERK, ExtSTS
        """
        raise(ValueError, msg)

    return flags

# utility routine to read an eigenvalue spectrum file and return the data as a dictionary
def read_spectrum_file(filename):
    data = {'time': [], 'real': [], 'imag': [], 'iters': []}
    with open(filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if (line.startswith('#')):
            continue
        values = line.split()
        data['time'].append(float(values[0]))
        data['real'].append(float(values[1]))
        data['imag'].append(float(values[2]))
        data['iters'].append(int(values[3]))
    return data

# utility routine to run a single C++ test
def runtest(exe='./bin/advection_diffusion_reaction_1D', probtype='AdvDiffRx', inttype='ARK', ststype=None, extststype=None, table_id=0, c=1e-2, d=1e-1, A=0.6, B=2.0, eps=1e-2, nx=512, rtol=1e-4, atol=1e-9, fixedh=0.0, maxl=0, nout=100):
    runcommand = "%s --c %e --d %e --A %e --B %e --eps %e --nx %d --rtol %e --atol %e --fixed_h %e --maxl %d --nout %d --output_domeig --maxsteps 1000000" % (exe, c, d, A, B, eps, nx, rtol, atol, fixedh, maxl, nout) + int_method(probtype, inttype, ststype, extststype, table_id)

    # run the test (and determine runtime)
    tstart = time.perf_counter()
    result = subprocess.run(shlex.split(runcommand), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    runtime = time.perf_counter() - tstart
    if (result.returncode != 0):
        print("Error: test failed with return code %d" % result.returncode)
        print("stdout:")
        print(result.stdout.decode('utf-8'))
        print("stderr:")
        print(result.stderr.decode('utf-8'))
        raise(RuntimeError, "test failed")

    # output information about the run to the screen
    print("Ran command: " + runcommand)
    print("Runtime: " + str(runtime) + " seconds")

    # load the relevant eigenvalue spectrum files and return
    diff_eigs = read_spectrum_file('diffusion_domeig.txt')
    os.remove('diffusion_domeig.txt')
    if (probtype in ['AdvDiffRx', 'AdvDiff']):
        adv_eigs = read_spectrum_file('advection_domeig.txt')
        os.remove('advection_domeig.txt')
    else:
        adv_eigs = None
    if (probtype in ['AdvDiffRx', 'RxDiff']):
        react_eigs = read_spectrum_file('reaction_domeig.txt')
        os.remove('reaction_domeig.txt')
    else:
        react_eigs = None

    return diff_eigs, adv_eigs, react_eigs

# utility routine to plot the spectra
def plot_spectra(diff_eigs, adv_eigs, react_eigs, titletxt, picname):
    plot_diff = True if (diff_eigs is not None) else False
    plot_adv = True if (adv_eigs is not None) else False
    plot_react = True if (react_eigs is not None) else False
    num_plots = plot_diff + plot_adv + plot_react
    spectra_figsize = (1+num_plots*4,4)
    fig = plt.figure(figsize=spectra_figsize)
    gs = GridSpec(1, num_plots, figure=fig)
    idx = 0
    ax = fig.add_subplot(gs[0,idx])
    ax.plot(diff_eigs['real'], diff_eigs['imag'], 'b.')
    ax.plot(diff_eigs['real'], -np.array(diff_eigs['imag']), 'b.')
    ax.set_title(r'diffusion $\lambda_{max}$')
    ax.set_xlabel(r'$\Re(\lambda)$')
    ax.set_ylabel(r'$\Im(\lambda)$')
    if (plot_adv):
        idx += 1
        ax_adv = fig.add_subplot(gs[0,idx])
        ax_adv.plot(adv_eigs['real'], adv_eigs['imag'], 'r.')
        ax_adv.plot(adv_eigs['real'], -np.array(adv_eigs['imag']), 'r.')
        ax_adv.set_title(r'advection $\lambda_{max}$')
        ax_adv.set_xlabel(r'$\Re(\lambda)$')
        ax_adv.set_ylabel(r'$\Im(\lambda)$')
    if (plot_react):
        idx += 1
        ax_react = fig.add_subplot(gs[0,idx])
        ax_react.plot(react_eigs['real'], react_eigs['imag'], 'g.')
        ax_react.plot(react_eigs['real'], -np.array(react_eigs['imag']), 'g.')
        ax_react.set_title(r'reaction $\lambda_{max}$')
        ax_react.set_xlabel(r'$\Re(\lambda)$')
        ax_react.set_ylabel(r'$\Im(\lambda)$')
    plt.suptitle(titletxt)
    if (Generate_PNG):
        plt.savefig(picname + '.png')
    if (Generate_PDF):
        plt.savefig(picname + '.pdf')

#####################
# testing setup

# Shared testing parameters
Executable = './bin/advection_diffusion_reaction_1D'
c = 0.5
dvals = [1e-1, 1e1]
A = 0.6
B = 2.0
eps = 1e-2
nx = 512
rtol = 1e-5
atol = 1e-11

# Advection-diffusion-reaction spectrum over the sets of diffusion coefficients
if (DoAdvDiffRx):
    for d in dvals:
        diff_eigs, adv_eigs, react_eigs = runtest(Executable, probtype='AdvDiffRx', inttype='ARK',
                                                  ststype=None, extststype=None, table_id=1, c=c, d=d,
                                                  A=A, B=B, eps=eps, nx=nx, rtol=rtol, atol=atol,
                                                  fixedh=0.0, nout=NumOut)
        # plot the spectra
        plot_spectra(diff_eigs, adv_eigs, react_eigs, titletxt=r'Advection-Diffusion-Reaction 1D Spectra ($d = %.1f$)' % d, picname='adr1d_spectrum_d%0.1f' % d)


# Advection-diffusion spectrum over the sets of diffusion coefficients
if (DoAdvDiff):
    for d in dvals:
        diff_eigs, adv_eigs, react_eigs = runtest(Executable, probtype='AdvDiff', inttype='ARK',
                                                  ststype=None, extststype=None, table_id=1, c=c, d=d,
                                                  A=A, B=B, eps=eps, nx=nx, rtol=rtol, atol=atol,
                                                  fixedh=0.0, nout=NumOut)
        # plot the spectra
        plot_spectra(diff_eigs, adv_eigs, react_eigs, titletxt=r'Advection-Diffusion 1D Spectra ($d = %.1f$)' % d, picname='ad1d_spectrum_d%0.1f' % d)


# Reaction-diffusion spectrum over the sets of diffusion coefficients
if (DoRxDiff):
    for d in dvals:
        diff_eigs, adv_eigs, react_eigs = runtest(Executable, probtype='RxDiff', inttype='ARK',
                                                  ststype=None, extststype=None, table_id=6, d=d,
                                                  A=A, B=B, eps=eps, nx=nx, rtol=rtol, atol=atol,
                                                  fixedh=0.0, nout=NumOut)
        # plot the spectra
        plot_spectra(diff_eigs, adv_eigs, react_eigs, titletxt=r'Reaction-Diffusion 1D Spectra ($d = %.1f$)' % d, picname='rd1d_spectrum_d%0.1f' % d)


# end of script