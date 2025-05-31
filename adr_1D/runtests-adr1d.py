#!/usr/bin/env python3
#------------------------------------------------------------
# Programmer(s):  Daniel R. Reynolds @ SMU
#------------------------------------------------------------
# Copyright (c) 2025, Southern Methodist University.
# All rights reserved.
# For details, see the LICENSE file.
#------------------------------------------------------------

# imports
import pandas as pd
import numpy as np
import subprocess
import shlex

#####################
# utility routines

# utility routine to set inputs for running a specific integration type
def int_method(probtype, inttype, ststype, extststype):
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
        flags += " --integrator 1"

    elif (inttype == "ERK"):
        flags += " --integrator 0"

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
            if (probtype != "AdvDiffRx"):
                raise(ValueError, "invalid problem + extsts type combination")
            flags += " --extsts_method 0"
        elif (extststype == "Ralston"):
            if (probtype != "AdvDiff"):
                raise(ValueError, "invalid problem + extsts type combination")
            flags += " --extsts_method 0"
        elif (extststype == "SSPSDIRK2"):
            if (probtype != "RxDiff"):
                raise(ValueError, "invalid problem + extsts type combination")
            flags += " --extsts_method 0"
        elif (extststype == "SSPSDIRK2"):
            if (probtype != "RxDiff"):
                raise(ValueError, "invalid problem + extsts type combination")
            flags += " --extsts_method 0"
        elif (extststype == "GiraldoARK"):
            if (probtype != "AdvDiffRx"):
                raise(ValueError, "invalid problem + extsts type combination")
            flags += " --extsts_method 1"
        elif (extststype == "GiraldoDIRK"):
            if (probtype != "RxDiff"):
                raise(ValueError, "invalid problem + extsts type combination")
            flags += " --extsts_method 1"
        elif (extststype == "HeunEuler"):
            if (probtype != "AdvDiff"):
                raise(ValueError, "invalid problem + extsts type combination")
            flags += " --extsts_method 1"
        else:
            msg = """
            Error: invalid extsts type
            Valid choices are: ARS, Ralston, SSPSDIRK2, GiraldoARK, GiraldoDIRK, HeunEuler
            """
            raise(ValueError, msg)

    else:
        msg = """
        Error: invalid integrator
        Valid integrator choices are: ARK, ERK, ExtSTS
        """
        raise(ValueError, msg)

    return flags

# utility routine to run a single nested KPR test, storing the run options and solver statistics
def runtest(exe='./bin/advection_diffusion_reaction', probtype='AdvDiffRx', inttype='ARK', ststype=None, extststype=None, c=1e-2, d=1e-1, eps=1e-2, nx=512, rtol=1e-4, atol=1e-9, fixedh=0.0, showcommand=False):
    stats = {'probtype': probtype, 'inttype': inttype, 'ststype': ststype, 'extststype': extststype, 'c': c, 'd': d, 'eps': eps, 'nx': nx, 'rtol': rtol, 'atol': atol, 'fixedh': fixedh, 'ReturnCode': 1, 'Steps': 1e10, 'Fails': 1e10, 'Accuracy': 1e10, 'AdvEvals': 1e10, 'DiffEvals': 1e10, 'RxEvals': 1e10}
    runcommand = "%s --c %e --d %e --eps %e --nx %d --rtol %e --atol %e --fixed_h %e --calc_error" % (exe, c, d, eps, nx, rtol, atol, fixedh) + int_method(probtype, inttype, ststype, extststype)
    result = subprocess.run(shlex.split(runcommand), stdout=subprocess.PIPE)
    stats['ReturnCode'] = result.returncode
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
DoAdaptiveTests = False

# Shared testing parameters
Executable = './bin/advection_diffusion_reaction'
AdvDiffRxSolvers = [['ARK', None, None],
                    ['ExtSTS', 'RKC', 'ARS'],
                    ['ExtSTS', 'RKL', 'ARS'],
                    ['ExtSTS', 'RKC', 'GiraldoARK'],
                    ['ExtSTS', 'RKL', 'GiraldoARK']]
AdvDiffSolvers = [['ARK', None, None],
                  ['ExtSTS', 'RKC', 'Ralston'],
                  ['ExtSTS', 'RKL', 'Ralston'],
                  ['ExtSTS', 'RKC', 'HeunEuler'],
                  ['ExtSTS', 'RKL', 'HeunEuler']]
RxDiffSolvers = [['ARK', None, None],
                 ['ExtSTS', 'RKC', 'SSPSDIRK2'],
                 ['ExtSTS', 'RKL', 'SSPSDIRK2'],
                 ['ExtSTS', 'RKC', 'GiraldoDIRK'],
                 ['ExtSTS', 'RKL', 'GiraldoDIRK']]
c = 1e-2
d = 1e-1
eps = 1e-2
nx = 512
rtol = [1e-3, 1e-4, 1e-5]
atol = 1e-11
fixedh = 0.1 / np.array([4, 8, 16, 32, 64, 128, 256], dtype=float)

# Advection-diffusion-reaction tests
if (DoAdvDiffRx):

    if (DoFixedTests):

        Stats = []
        for solver in AdvDiffRxSolvers:
            for h in fixedh:
                Stats.append(runtest(Executable, probtype='AdvDiffRx', inttype=solver[0],
                                     ststype=solver[1], extststype=solver[2], c=c, d=d, eps=eps,
                                     nx=nx, fixedh=h))
        Df = pd.DataFrame.from_records(Stats)
        print("Fixed step AdvDiffRx test Df:")
        print(Df)
        print("Saving as Excel")
        Df.to_excel('AdvDiffRx-fixed.xlsx', index=False)

    if (DoAdaptiveTests):

        Stats = []
        for solver in AdvDiffRxSolvers:
            for rt in rtol:
                Stats.append(runtest(Executable, probtype='AdvDiffRx', inttype=solver[0],
                                     ststype=solver[1], extststype=solver[2], c=c, d=d,
                                     eps=eps, nx=nx, rtol=rt, atol=atol, fixedh=0.0))
        Df = pd.DataFrame.from_records(Stats)
        print("Adaptive step AdvDiffRx test Df:")
        print(Df)
        print("Saving as Excel")
        Df.to_excel('AdvDiffRx-adapt.xlsx', index=False)

# Advection-diffusion tests
if (DoAdvDiff):

    if (DoFixedTests):

        Stats = []
        for solver in AdvDiffSolvers:
            for h in fixedh:
                Stats.append(runtest(Executable, probtype='AdvDiff', inttype=solver[0],
                                     ststype=solver[1], extststype=solver[2], c=c, d=d,
                                     nx=nx, fixedh=h))
        Df = pd.DataFrame.from_records(Stats)
        print("Fixed step AdvDiff test Df:")
        print(Df)
        print("Saving as Excel")
        Df.to_excel('AdvDiff-fixed.xlsx', index=False)

    if (DoAdaptiveTests):

        Stats = []
        for solver in AdvDiffSolvers:
            for rt in rtol:
                Stats.append(runtest(Executable, probtype='AdvDiff', inttype=solver[0],
                                     ststype=solver[1], extststype=solver[2], c=c, d=d,
                                     nx=nx, rtol=rt, atol=atol, fixedh=0.0))
        Df = pd.DataFrame.from_records(Stats)
        print("Adaptive step AdvDiff test Df:")
        print(Df)
        print("Saving as Excel")
        Df.to_excel('AdvDiff-adapt.xlsx', index=False)

# Reaction-diffusion tests
if (DoRxDiff):

    if (DoFixedTests):

        Stats = []
        for solver in RxDiffSolvers:
            for h in fixedh:
                Stats.append(runtest(Executable, probtype='RxDiff', inttype=solver[0],
                                     ststype=solver[1], extststype=solver[2], d=d, eps=eps,
                                     nx=nx, fixedh=h))
        Df = pd.DataFrame.from_records(Stats)
        print("Fixed step RxDiff test Df:")
        print(Df)
        print("Saving as Excel")
        Df.to_excel('RxDiff-fixed.xlsx', index=False)

    if (DoAdaptiveTests):

        Stats = []
        for solver in RxDiffSolvers:
            for rt in rtol:
                Stats.append(runtest(Executable, probtype='RxDiff', inttype=solver[0],
                                     ststype=solver[1], extststype=solver[2], d=d, eps=eps,
                                     nx=nx, rtol=rt, atol=atol, fixedh=0.0))
        Df = pd.DataFrame.from_records(Stats)
        print("Adaptive step RxDiff test Df:")
        print(Df)
        print("Saving as Excel")
        Df.to_excel('RxDiff-adapt.xlsx', index=False)

