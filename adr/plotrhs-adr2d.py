#!/usr/bin/env python
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
DoRxDiff = False
NumOut = 100

#####################
# utility routines

# utility routine to set C++ executable inputs for running a specific integration type
def int_method(probtype, implicitrx, inttype, ststype, extststype, table_id):
    flags = ""
    if (probtype == "RxDiff"):
        flags += " --no-advection "
        flags += " "
    elif (probtype == "AdvDiffRx"):
        flags += " "
    else:
        msg = """
        Error: invalid problem type
        Valid problem types are: RxDiff, AdvDiffRx
        """
        print(msg + "(" + str(probtype) + " specified)")
        raise(ValueError, msg)

    if (implicitrx):
        flags += " --implicit-reaction --maxnewt 5"

    if (inttype == "ARK"):
        flags += " --integrator 1 --table_id %d" % table_id

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
            print(msg + "(" + str(ststype) + " specified)")
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
            print(msg + "(" + str(ststype) + " specified)")
            raise(ValueError, msg)

        if (extststype == "ARS"):
            flags += " --extsts_method 0"
        elif (extststype == "Giraldo"):
            flags += " --extsts_method 1"
        elif (extststype == "Ralston"):
            flags += " --extsts_method 2"
        elif (extststype == "Heun-Euler"):
            flags += " --extsts_method 3"
        elif (extststype == "SSPSDIRK2"):
            flags += " --extsts_method 4"
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
            Valid choices are: ARS, Giraldo, Ralston, Heun-Euler, SSPSDIRK2, IRK21a, ESDIRK34a, ERK22a, ERK22b, MERK21, MERK32, MRISR21
            """
            print(msg + "(" + str(extststype) + " specified)")
            raise(ValueError, msg)

    else:
        msg = """
        Error: invalid integrator
        Valid integrator choices are: ARK, ERK, ExtSTS
        """
        print(msg + "(" + str(inttype) + " specified)")
        raise(ValueError, msg)

    return flags


# utility routine to read a RHS norm file and return the data as a dictionary
def read_rhs_file(filename):
    data = {'time': [], 'fa': [], 'fd': [], 'fr': [], 'fad': [], 'far': [], 'frd': [], 'fadr': []}
    with open(filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if (line.startswith('#')):
            continue
        values = line.split()
        data['time'].append(float(values[0]))
        data['fa'].append(float(values[1]))
        data['fd'].append(float(values[2]))
        data['fr'].append(float(values[3]))
        data['fad'].append(float(values[4]))
        data['far'].append(float(values[5]))
        data['frd'].append(float(values[6]))
        data['fadr'].append(float(values[7]))
    return data

# utility routine to run a single C++ test, storing the run options and solver statistics
def runtest(exe='./bin/advection_diffusion_reaction_2D', probtype='AdvDiffRx', implicitrx=False, inttype='ARK', ststype=None, extststype=None, table_id=0, cux=-0.5, cuy=1.0, cvx=0.4, cvy=0.7, d=1e-2, A=1.3, B=1.0, nx=400, ny=400, tf=1.0, rtol=1e-4, atol=1e-9, fixedh=0.0, nout=100):
    runcommand = "%s --cux %e --cuy %e --cvx %e --cvy %e --d %e --A %e --B %e --nx %d --ny %d --tf %e --rtol %e --atol %e --fixed_h %e --nout %d --output_rhsnorms --maxsteps 10000000" % (exe, cux, cuy, cvx, cvy, d, A, B, nx, ny, tf, rtol, atol, fixedh, nout) + int_method(probtype, implicitrx, inttype, ststype, extststype, table_id)

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

    # load the relevant RHS norm file and return
    rhsnorms = read_rhs_file('rhs_norms.txt')
    return rhsnorms

# utility routine to plot the RHS norms
def plot_rhsnorms(rhs_norms, titletxt, picname):
    num_plots = 2
    rhs_figsize = (1+num_plots*4,4)
    fig = plt.figure(figsize=rhs_figsize)
    gs = GridSpec(1, num_plots, figure=fig)
    idx = 0
    ax = fig.add_subplot(gs[0,idx])
    ax.semilogy(rhs_norms['time'], rhs_norms['fa'], 'b-', label=r'$\|f_a\|$')
    ax.semilogy(rhs_norms['time'], rhs_norms['fd'], 'g-', label=r'$\|f_d\|$')
    ax.semilogy(rhs_norms['time'], rhs_norms['fr'], 'r-', label=r'$\|f_r\|$')
    ax.set_title(r'Individual')
    ax.set_xlabel(r'Time')
    ax.set_ylabel(r'Norm')
    ax.legend()

    idx = 1
    ax = fig.add_subplot(gs[0,idx])
    fmax = np.max([np.array(rhs_norms['fa']), np.array(rhs_norms['fd']), np.array(rhs_norms['fr'])], axis=0)
    ax.semilogy(rhs_norms['time'], fmax, 'c-', label=r'$\max(\|f^A\|, \|f^D\|, \|f^R\|)$')
    ax.semilogy(rhs_norms['time'], rhs_norms['fadr'], 'm-', label=r'$\|f^A+f^D+f^R\|$')
    ax.set_title(r'Combined')
    ax.set_xlabel(r'Time')
    ax.set_ylabel(r'Norm')
    ax.legend()

    plt.suptitle(titletxt)
    if (Generate_PNG):
        plt.savefig(picname + '.png')
    if (Generate_PDF):
        plt.savefig(picname + '.pdf')


#####################
# testing setup

# Shared testing parameters
adrexe='./bin/advection_diffusion_reaction_2D_stationary'
bctype = 'stationary'
#adrexe='./bin/advection_diffusion_reaction_2D'
#bctype = 'periodic'
rtol = 1e-5
atol = 1e-11

# Advection-diffusion-reaction tests
if (DoAdvDiffRx):

    cux=-0.5
    cuy=1.0
    cvx=0.4
    cvy=0.7
    Bvals=[3.0, 3e1, 3e2]
    dvals=[0.01, 0.1]
    A=1.0
    nx=400
    ny=400
    tf=5.0
    for d in dvals:
        for B in Bvals:
            rhsnorms = runtest(exe=adrexe, probtype='AdvDiffRx', implicitrx=True, inttype='ARK',
                               ststype=None, extststype=None, table_id=1, cux=cux, cuy=cuy,
                               cvx=cvx, cvy=cvy, d=d, A=A, B=B, nx=nx, ny=ny, tf=tf, rtol=rtol,
                               atol=atol, fixedh=0.0, nout=NumOut)

            # rename the rhs_norms.txt file to a more descriptive name
            os.rename('rhs_norms.txt', 'adr2d_%s_rhsnorms_d%0.2f_B%0.1e.txt' % (bctype, d, B))

            # plot the RHS norms
            plot_rhsnorms(rhsnorms,
                          titletxt=r'Advection-Diffusion-Reaction 2D RHS Norms (%s, $d = %.2f$, $B = %.1e$)' % (bctype, d, B), picname='adr2d_%s_rhsnorms_d%0.2f_B%0.1e' % (bctype, d, B))

# Reaction-diffusion tests
if (DoRxDiff):

    cux=0.0
    cuy=0.0
    cvx=0.0
    cvy=0.0
    A=1.3
    Bvals=[2.e1, 2.e4, 2.e7]
    dvals=[0.01, 0.1]
    nx=200
    ny=200
    tf=2.0
    for d in dvals:
        for B in Bvals:
            rhsnorms = runtest(exe=adrexe, probtype='RxDiff', implicitrx=True, inttype='ARK',
                               ststype=None, extststype=None, table_id=1, cux=cux, cuy=cuy,
                               cvx=cvx, cvy=cvy, d=d, A=A, B=B, nx=nx, ny=ny, tf=tf, rtol=rtol,
                               atol=atol, fixedh=0.0, nout=NumOut)

            # rename the rhs_norms.txt file to a more descriptive name
            os.rename('rhs_norms.txt', 'rd2d_%s_rhsnorms_d%0.2f_B%0.1e.txt' % (bctype, d, B))

            # plot the RHS norms
            plot_rhsnorms(rhsnorms,
                         titletxt=r'Reaction-Diffusion 2D RHS Norms (%s, $d = %.2f$, $B = %.1e$)' % (bctype, d, B), picname='rd2d_%s_rhsnorms_d%0.2f_B%0.1e' % (bctype, d, B))

# end of script
