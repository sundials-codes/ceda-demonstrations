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
import numpy as np
import subprocess
import shlex
import time

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


# utility routine to compute solution error between two files
def calc_error(nx, solfile, reffile):
    soldata = np.loadtxt(solfile)
    refdata = np.loadtxt(reffile)
    uerr = soldata[1:]-refdata[1:]
    return np.sqrt(np.dot(uerr,uerr) / nx / 3)

# utility routine to run the C++ executable to generate a reference solution for a given problem configuration
def generate_reference(exe='./bin/advection_diffusion_reaction_1D', probtype='AdvDiffRx', c=1e-2, d=1e-1, A=0.6, B=2.0, eps=1e-2, nx=512):
    runcommand = "%s --c %e --d %e --A %e --B %e --eps %e --nx %d --nout 1 --calc_error --write_solution" % (exe, c, d, A, B, eps, nx) + int_method(probtype, "ExtSTS", 'RKC', 'ARS', None)
    result = subprocess.run(shlex.split(runcommand), stdout=subprocess.PIPE)


# utility routine to run the PIROCK executable
def runtest_pirock(exe='./bin/advection_diffusion_reaction_1D_pirock', probtype='AdvDiffRx', c=1e-2, d=1e-1, A=0.6, B=2.0, eps=1e-2, nx=512, rtol=1e-4, atol=1e-9, fixedh=0.0, showcommand=False):
    if (nx != 512):
        raise(ValueError, "To run without 512 spatial nodes, need to edit/recompile pb_adr_1D.f (and this error check)")
    stats = {'probtype': probtype, 'inttype': 'PIROCK', 'ststype': None, 'extststype': None, 'table_id': 0, 'c': c, 'd': d, 'A': A, 'B': B, 'eps': eps, 'nx': nx, 'rtol': rtol, 'atol': atol, 'fixedh': fixedh, 'maxl': 0, 'nout': 1, 'ReturnCode': 1, 'Steps': np.nan, 'Fails': np.nan, 'Accuracy': np.nan, 'AdvEvals': np.nan, 'DiffEvals': np.nan, 'RxEvals': np.nan}

    advec_iwork20 = 1  # True
    reac_iwork21 = 1   # True
    if (probtype == "AdvDiff"):
        reac_iwork21 = 0
    elif (probtype == "RxDiff"):
        advec_iwork20 = 0

    # modify parameters in namelist file and turn on/off advection/reaction
    with open("adr_1D_pirock_params.txt",'w') as namefile:
        namefile.write("&list1\n")
        namefile.write("   alf = " + str(d) + "\n")
        namefile.write("   uxadv = " + str(c) + "\n")
        namefile.write("   uyadv = 0.0\n")
        namefile.write("   vxadv = " + str(c) + "\n")
        namefile.write("   vyadv = 0.0\n")
        namefile.write("   wxadv = " + str(c) + "\n")
        namefile.write("   wyadv = 0.0\n")
        namefile.write("   brussa = " + str(A) + "\n")
        namefile.write("   brussb = " + str(B) + "\n")
        namefile.write("   eps = " + str(eps) + "\n")
        namefile.write("   atol = " + str(atol) + "\n")
        namefile.write("   rtol = " + str(rtol) + "\n")
        namefile.write("   h = " + str(fixedh) + "\n")
        namefile.write("   iwork20 = " + str(advec_iwork20) + "\n")
        namefile.write("   iwork21 = " + str(reac_iwork21) + "\n")
        namefile.write("/\n")

    # run the test (and determine runtime)
    tstart = time.perf_counter()
    result = subprocess.run(shlex.split(exe), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    runtime = time.perf_counter() - tstart

    # process the run results
    stats['ReturnCode'] = result.returncode
    stats['RunTime'] = runtime
    if (result.returncode != 0):
        print("Run command " + runcommand + " FAILURE: " + str(run_result.returncode))
        print(result.stderr)
    else:
        if(showcommand):
            print("Run command: " + runcommand + " SUCCESS\n")

        # compute solution error and store this in the stats
        stats['Accuracy'] = calc_error(nx, "sol.dat", "reference.dat")

        # get remaining stats from stdout
        try:
            lines = str(result.stdout).split('\\n')
            for line in lines:
                if 'Number of f evaluations' in line:
                    txt = line.split()
                    stats['DiffEvals'] = int(txt[4])
                    stats['AdvEvals'] = int(txt[7])
                    stats['Steps'] = int(txt[9])
                    stats['Fails'] = int(txt[13])
                elif 'Number of reaction VF' in line:
                    txt = line.split()
                    stats['RxEvals'] = int(txt[5])
        except:
            print("Error processing PIROCK output:")
            print(lines)
    return stats


# utility routine to run a single C++ test, storing the run options and solver statistics
def runtest(exe='./bin/advection_diffusion_reaction_1D', probtype='AdvDiffRx', inttype='ARK', ststype=None, extststype=None, table_id=0, c=1e-2, d=1e-1, A=0.6, B=2.0, eps=1e-2, nx=512, rtol=1e-4, atol=1e-9, fixedh=0.0, maxl=0, nout=20, showcommand=False):
    stats = {'probtype': probtype, 'inttype': inttype, 'ststype': ststype, 'extststype': extststype, 'table_id': table_id, 'c': c, 'd': d, 'A': A, 'B': B, 'eps': eps, 'nx': nx, 'rtol': rtol, 'atol': atol, 'fixedh': fixedh, 'maxl': maxl, 'nout': nout, 'ReturnCode': 1, 'Steps': np.nan, 'Fails': np.nan, 'Accuracy': np.nan, 'AdvEvals': np.nan, 'DiffEvals': np.nan, 'RxEvals': np.nan}
    runcommand = "%s --c %e --d %e --A %e --B %e --eps %e --nx %d --rtol %e --atol %e --fixed_h %e --maxl %d --nout %d --calc_error --maxsteps 1000000" % (exe, c, d, A, B, eps, nx, rtol, atol, fixedh, maxl, nout) + int_method(probtype, inttype, ststype, extststype, table_id)

    # run the test (and determine runtime)
    tstart = time.perf_counter()
    result = subprocess.run(shlex.split(runcommand), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    runtime = time.perf_counter() - tstart

    # process the run results
    stats['ReturnCode'] = result.returncode
    stats['RunTime'] = runtime
    if (result.returncode != 0):
        print("Run command " + runcommand + " FAILURE: " + str(result.returncode))
    else:
        if (showcommand):
            print("Run command " + runcommand + " SUCCESS")
        lines = str(result.stdout).split('\\n')
        if (inttype == "ARK"):
            for line in lines:
                txt = line.split()
                if ("Steps" in txt):
                    stats['Steps'] = int(txt[2])
                if (("Error" in txt) and ("test" in txt) and ("fails" in txt)):
                    stats['Fails'] = int(txt[4])
                elif (("Solution" in txt) and ("error" in txt)):
                    stats['Accuracy'] = float(txt[3])
                elif (("Explicit" in txt) and ("RHS" in txt) and ("evals" in txt)):
                    if (probtype == "AdvDiffRx" or probtype == "RxDiff"):
                        stats['AdvEvals'] = int(txt[4])
                    if (probtype == "AdvDiff"):
                        stats['AdvEvals'] = int(txt[4])
                elif (("Implicit" in txt) and ("RHS" in txt) and ("evals" in txt)):
                    if (probtype == "AdvDiffRx" or probtype == "RxDiff"):
                        stats['DiffEvals'] = int(txt[4])
                        stats['RxEvals'] = int(txt[4])
                    if (probtype == "AdvDiff"):
                        stats['DiffEvals'] = int(txt[4])
                        stats['RxEvals'] = 0
        elif (inttype == "ERK"):
            for line in lines:
                txt = line.split()
                if ("Steps" in txt):
                    stats['Steps'] = int(txt[2])
                if (("Error" in txt) and ("test" in txt) and ("fails" in txt)):
                    stats['Fails'] = int(txt[4])
                elif (("Solution" in txt) and ("error" in txt)):
                    stats['Accuracy'] = float(txt[3])
                elif (("RHS" in txt) and ("evals" in txt)):
                    if (probtype == "AdvDiffRx"):
                        stats['AdvEvals'] = int(txt[3])
                        stats['DiffEvals'] = int(txt[3])
                        stats['RxEvals'] = int(txt[3])
                    if (probtype == "AdvDiff"):
                        stats['AdvEvals'] = int(txt[3])
                        stats['DiffEvals'] = int(txt[3])
                        stats['RxEvals'] = 0
                    if (probtype == "RxDiff"):
                        stats['AdvEvals'] = 0
                        stats['DiffEvals'] = int(txt[3])
                        stats['RxEvals'] = int(txt[3])
        elif (inttype == "Strang"):
            for line in lines:
                txt = line.split()
                if ("Steps" in txt and stats['Steps'] == 1e10):
                    stats['Steps'] = int(txt[2])
                if (("Error" in txt) and ("test" in txt) and ("fails" in txt)):
                    stats['Fails'] = int(txt[4])
                elif (("Solution" in txt) and ("error" in txt)):
                    stats['Accuracy'] = float(txt[3])
                elif (("Explicit" in txt) and ("RHS" in txt) and ("evals" in txt)):
                    if (probtype == "AdvDiffRx" or probtype == "AdvDiff"):
                        stats['AdvEvals'] = int(txt[5])
                    else:
                        stats['AdvEvals'] = 0
                elif (("Implicit" in txt) and ("RHS" in txt) and ("evals" in txt)):
                    if (probtype == "AdvDiffRx" or probtype == "RxDiff"):
                        stats['RxEvals'] = int(txt[5])
                    else:
                        stats['RxEvals'] = 0
                elif (("RHS" in txt) and ("fn" in txt) and ("evals" in txt) and ("LS" not in txt)):
                    stats['DiffEvals'] = int(txt[4])
        else:
            for line in lines:
                txt = line.split()
                if ("Steps" in txt and stats['Steps'] == 1e10):
                    stats['Steps'] = int(txt[2])
                if (("Error" in txt) and ("test" in txt) and ("fails" in txt) and stats['Fails'] == 1e10):
                    stats['Fails'] = int(txt[4])
                elif (("Solution" in txt) and ("error" in txt)):
                    stats['Accuracy'] = float(txt[3])
                elif (("Explicit" in txt) and ("RHS" in txt) and ("evals" in txt)):
                    stats['AdvEvals'] = int(txt[6])
                elif (("Implicit" in txt) and ("RHS" in txt) and ("evals" in txt)):
                    stats['RxEvals'] = int(txt[6])
                elif (("RHS" in txt) and ("fn" in txt) and ("evals" in txt) and ("LS" not in txt)):
                    stats['DiffEvals'] = int(txt[4])
    return stats


#####################
# testing setup

# Flags to enable/disable categories of tests
DoAdvDiffRx = True
DoAdvDiff = True
DoRxDiff = True
DoFixedTests = True
DoAdaptiveTests = True

# Shared testing parameters
Executable = './bin/advection_diffusion_reaction_1D'
PIROCKExecutable = './bin/advection_diffusion_reaction_1D_pirock'
AdvDiffRxSolvers = [['ARK', None, None, 1],
                    ['ARK', None, None, 2],
                    ['ExtSTS', 'RKC', 'ARS', None],
                    ['ExtSTS', 'RKL', 'ARS', None],
                    ['ExtSTS', 'RKC', 'Giraldo', None],
                    ['ExtSTS', 'RKL', 'Giraldo', None],
                    ['ExtSTS', 'RKC', 'MRISR21', None],
                    ['ExtSTS', 'RKL', 'MRISR21', None]]
AdvDiffSolvers = [['ARK', None, None, 1],
                  ['ARK', None, None, 2],
                  ['ExtSTS', 'RKC', 'ARS', None],
                  ['ExtSTS', 'RKL', 'ARS', None],
                  ['ExtSTS', 'RKC', 'Giraldo', None],
                  ['ExtSTS', 'RKL', 'Giraldo', None],
                  ['ExtSTS', 'RKC', 'MRISR21', None],
                  ['ExtSTS', 'RKL', 'MRISR21', None],
                  ['ExtSTS', 'RKC', 'Ralston', None],
                  ['ExtSTS', 'RKL', 'Ralston', None],
                  ['ExtSTS', 'RKC', 'ERK22a', None],
                  ['ExtSTS', 'RKL', 'ERK22a', None],
                  ['ExtSTS', 'RKC', 'ERK22b', None],
                  ['ExtSTS', 'RKL', 'ERK22b', None],
                  ['ExtSTS', 'RKC', 'MERK21', None],
                  ['ExtSTS', 'RKL', 'MERK21', None],
                  ['ExtSTS', 'RKC', 'MERK32', None],
                  ['ExtSTS', 'RKL', 'MERK32', None]]
RxDiffSolvers = [['ARK', None, None, 5],
                 ['ARK', None, None, 6],
                 ['ExtSTS', 'RKC', 'ARS', None],
                 ['ExtSTS', 'RKL', 'ARS', None],
                 ['ExtSTS', 'RKC', 'Giraldo', None],
                 ['ExtSTS', 'RKL', 'Giraldo', None],
                 ['ExtSTS', 'RKC', 'MRISR21', None],
                 ['ExtSTS', 'RKL', 'MRISR21', None],
                 ['ExtSTS', 'RKC', 'SSPSDIRK2', None],
                 ['ExtSTS', 'RKL', 'SSPSDIRK2', None],
                 ['ExtSTS', 'RKC', 'IRK21a', None],
                 ['ExtSTS', 'RKL', 'IRK21a', None],
                 ['ExtSTS', 'RKC', 'ESDIRK34a', None],
                 ['ExtSTS', 'RKL', 'ESDIRK34a', None]]
StrangSolvers = [['Strang', 'RKC', None, None],
                 ['Strang', 'RKL', None, None]]
#c = 1e-2
c = 0.5
dvals = [1e-1, 1e1]
A = 0.6
B = 2.0
eps = 1e-2
nx = 512
fixed_maxl = 500
nout = 1

# Advection-diffusion-reaction tests
if (DoAdvDiffRx):

    # loop over diffusion coefficients
    FixedStats = []
    AdaptStats = []
    for d in dvals:

        # generate reference solution for PIROCK error
        generate_reference(Executable, probtype='AdvDiffRx', c=c, d=d, A=A, B=B,
                            eps=eps, nx=nx)

        if (DoFixedTests):

            # set step sizes for fixed-step ADR tests
            fixedh        = 0.02 / np.array([4, 8, 16, 32, 64, 128, 256], dtype=float)
            fixedh_strang = 0.01 / np.array([4, 8, 16, 32, 64, 128, 256], dtype=float)
            fixedh_pirock = 0.01 / np.array([4, 8, 16, 32, 64, 128, 256], dtype=float)

            for solver in AdvDiffRxSolvers:
                for h in fixedh:
                    FixedStats.append(runtest(Executable, probtype='AdvDiffRx', inttype=solver[0],
                                        ststype=solver[1], extststype=solver[2],
                                        table_id=solver[3], c=c, d=d, A=A, B=B, eps=eps,
                                        nx=nx, fixedh=h, rtol=max(1e-3*(h*h),1e-9), maxl=fixed_maxl, nout=nout))
            for solver in StrangSolvers:
                for h in fixedh_strang:
                    FixedStats.append(runtest(Executable, probtype='AdvDiffRx', inttype=solver[0],
                                        ststype=solver[1], extststype=solver[2],
                                        table_id=solver[3], c=c, d=d, A=A, B=B, eps=eps,
                                        nx=nx, fixedh=h, rtol=max(1e-3*(h*h),1e-9), maxl=fixed_maxl, nout=nout))
            for h in fixedh_pirock:
                FixedStats.append(runtest_pirock(PIROCKExecutable, probtype='AdvDiffRx', c=c, d=d, A=A, B=B,
                                    eps=eps, nx=nx, rtol=max(1e-3*(h*h),1e-9), fixedh=h))

        if (DoAdaptiveTests):

            # set tolerances for adaptive ADR tests
            rtol = np.logspace(-2.5, -6.5, 7)
            atol = 1e-11

            for solver in AdvDiffRxSolvers:
                for rt in rtol:
                    AdaptStats.append(runtest(Executable, probtype='AdvDiffRx', inttype=solver[0],
                                        ststype=solver[1], extststype=solver[2],
                                        table_id=solver[3], c=c, d=d, A=A, B=B,
                                        eps=eps, nx=nx, rtol=rt, atol=atol, fixedh=0.0, nout=nout))
            for rt in rtol:
                AdaptStats.append(runtest_pirock(PIROCKExecutable, probtype='AdvDiffRx', c=c, d=d, A=A, B=B,
                                            eps=eps, nx=nx, rtol=rt, atol=atol, fixedh=0.0))

    if (DoFixedTests):
        Df = pd.DataFrame.from_records(FixedStats)
        print("Fixed step AdvDiffRx test Df:")
        print(Df)
        print("Saving as Excel")
        Df.to_excel('AdvDiffRx-fixed.xlsx', index=False)

    if (DoAdaptiveTests):
        Df = pd.DataFrame.from_records(AdaptStats)
        print("Adaptive step AdvDiffRx test Df:")
        print(Df)
        print("Saving as Excel")
        Df.to_excel('AdvDiffRx-adapt.xlsx', index=False)

# Advection-diffusion tests
if (DoAdvDiff):

    # loop over diffusion coefficients
    FixedStats = []
    AdaptStats = []
    for d in dvals:

        # generate reference solution for PIROCK error
        generate_reference(Executable, probtype='AdvDiff', c=c, d=d, nx=nx)

        if (DoFixedTests):

            # set step sizes for fixed-step AD tests
            fixedh        = 0.02 / np.array([4, 8, 16, 32, 64, 128, 256], dtype=float)
            fixedh_strang = 0.01 / np.array([4, 8, 16, 32, 64, 128, 256], dtype=float)
            fixedh_pirock = 0.01 / np.array([4, 8, 16, 32, 64, 128, 256], dtype=float)

            for solver in AdvDiffSolvers:
                for h in fixedh:
                    FixedStats.append(runtest(Executable, probtype='AdvDiff', inttype=solver[0],
                                        ststype=solver[1], extststype=solver[2],
                                        table_id=solver[3], c=c, d=d,
                                        nx=nx, fixedh=h, rtol=max(1e-3*(h*h),1e-9), maxl=fixed_maxl, nout=nout))
            for solver in StrangSolvers:
                for h in fixedh_strang:
                    FixedStats.append(runtest(Executable, probtype='AdvDiff', inttype=solver[0],
                                        ststype=solver[1], extststype=solver[2],
                                        table_id=solver[3], c=c, d=d,
                                        nx=nx, fixedh=h, rtol=max(1e-3*(h*h),1e-9), maxl=fixed_maxl, nout=nout))
            for h in fixedh_pirock:
                FixedStats.append(runtest_pirock(PIROCKExecutable, probtype='AdvDiff', c=c, d=d,
                                            nx=nx, rtol=max(1e-3*(h*h),1e-9), fixedh=h))


        if (DoAdaptiveTests):

            # set tolerances for adaptive AD tests
            rtol = np.logspace(-2.5, -6.5, 7)
            atol = 1e-11

            for solver in AdvDiffSolvers:
                for rt in rtol:
                    AdaptStats.append(runtest(Executable, probtype='AdvDiff', inttype=solver[0],
                                        ststype=solver[1], extststype=solver[2],
                                        table_id=solver[3], c=c, d=d,
                                        nx=nx, rtol=rt, atol=atol, fixedh=0.0, nout=nout))
            for rt in rtol:
                AdaptStats.append(runtest_pirock(PIROCKExecutable, probtype='AdvDiff', c=c, d=d,
                                            nx=nx, rtol=rt, atol=atol, fixedh=0.0))

    if (DoFixedTests):
        Df = pd.DataFrame.from_records(FixedStats)
        print("Fixed step AdvDiff test Df:")
        print(Df)
        print("Saving as Excel")
        Df.to_excel('AdvDiff-fixed.xlsx', index=False)

    if (DoAdaptiveTests):
        Df = pd.DataFrame.from_records(AdaptStats)
        print("Adaptive step AdvDiff test Df:")
        print(Df)
        print("Saving as Excel")
        Df.to_excel('AdvDiff-adapt.xlsx', index=False)

# Reaction-diffusion tests
if (DoRxDiff):

    # loop over diffusion coefficients
    FixedStats = []
    AdaptStats = []
    for d in dvals:

        # generate reference solution for PIROCK error
        generate_reference(Executable, probtype='RxDiff', d=d, A=A, B=B,
                            eps=eps, nx=nx)

        if (DoFixedTests):

            # set step sizes for fixed-step RD tests
            fixedh        = 0.05 / np.array([4, 8, 16, 32, 64, 128, 256], dtype=float)
            fixedh_strang = 0.05 / np.array([4, 8, 16, 32, 64, 128, 256], dtype=float)
            fixedh_pirock = 0.05 / np.array([4, 8, 16, 32, 64, 128, 256], dtype=float)

            for solver in RxDiffSolvers:
                for h in fixedh:
                    FixedStats.append(runtest(Executable, probtype='RxDiff', inttype=solver[0],
                                        ststype=solver[1], extststype=solver[2],
                                        table_id=solver[3], d=d, A=A, B=B, eps=eps,
                                        nx=nx, fixedh=h, rtol=max(1e-3*(h*h),1e-9), maxl=fixed_maxl, nout=nout))
            for solver in StrangSolvers:
                for h in fixedh_strang:
                    FixedStats.append(runtest(Executable, probtype='RxDiff', inttype=solver[0],
                                        ststype=solver[1], extststype=solver[2],
                                        table_id=solver[3], d=d, A=A, B=B, eps=eps,
                                        nx=nx, fixedh=h, rtol=max(1e-3*(h*h),1e-9), maxl=fixed_maxl, nout=nout))
            for h in fixedh_pirock:
                FixedStats.append(runtest_pirock(PIROCKExecutable, probtype='RxDiff', d=d, A=A, B=B, eps=eps,
                                            nx=nx, rtol=max(1e-3*(h*h),1e-9), fixedh=h))

        if (DoAdaptiveTests):

            # set tolerances for adaptive RD tests
            rtol = np.logspace(-2.5, -6.5, 7)
            atol = 1e-11

            for solver in RxDiffSolvers:
                for rt in rtol:
                    AdaptStats.append(runtest(Executable, probtype='RxDiff', inttype=solver[0],
                                        ststype=solver[1], extststype=solver[2],
                                        table_id=solver[3], d=d, A=A, B=B, eps=eps,
                                        nx=nx, rtol=rt, atol=atol, fixedh=0.0, nout=nout))
            for rt in rtol:
                AdaptStats.append(runtest_pirock(PIROCKExecutable, probtype='RxDiff', d=d,
                                            A=A, B=B, eps=eps, nx=nx, rtol=rt, atol=atol, fixedh=0.0))

    if (DoFixedTests):
        Df = pd.DataFrame.from_records(FixedStats)
        print("Fixed step RxDiff test Df:")
        print(Df)
        print("Saving as Excel")
        Df.to_excel('RxDiff-fixed.xlsx', index=False)

    if (DoAdaptiveTests):
        Df = pd.DataFrame.from_records(AdaptStats)
        print("Adaptive step RxDiff test Df:")
        print(Df)
        print("Saving as Excel")
        Df.to_excel('RxDiff-adapt.xlsx', index=False)

# end of script