#!/usr/bin/env python3
#------------------------------------------------------------
# Programmer(s):  Daniel R. Reynolds @ UMBC
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
        else:
            msg = """
            Error: invalid extsts type
            Valid choices are: ARS, Giraldo, Ralston, HeunEuler, SSPSDIRK2
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
    usol = np.reshape(soldata[1:],[nx,3])
    uref = np.reshape(refdata[1:],[nx,3])
    uerr = usol-uref
    return np.sqrt(np.mean(np.square(uerr)))

# utility routine to run the C++ executable to generate a reference solution for a given problem configuration
def generate_reference(exe='./bin/advection_diffusion_reaction', probtype='AdvDiffRx', c=1e-2, d=1e-1, eps=1e-2, nx=512):
    runcommand = "%s --c %e --d %e --eps %e --nx %d --nout 1 --calc_error --write_solution" % (exe, c, d, eps, nx) + int_method(probtype, "ExtSTS", 'RKC', 'ARS', None)
    result = subprocess.run(shlex.split(runcommand), stdout=subprocess.PIPE)


# utility routine to run the PIROCK executable
def runtest_pirock(exe='./bin/advection_diffusion_reaction_pirock', probtype='AdvDiffRx', c=1e-2, d=1e-1, eps=1e-2, nx=512, rtol=1e-4, atol=1e-9, fixedh=0.0, showcommand=False):
    if (nx != 512):
        raise(ValueError, "To run without 512 spatial nodes, need to edit/recompile pb_adr_1D.f (and this error check)")
    stats = {'probtype': probtype, 'inttype': 'PIROCK', 'ststype': None, 'extststype': None, 'table_id': 0, 'c': c, 'd': d, 'eps': eps, 'nx': nx, 'rtol': rtol, 'atol': atol, 'fixedh': fixedh, 'maxl': 0, 'nout': 1, 'ReturnCode': 1, 'Steps': 1e10, 'Fails': 1e10, 'Accuracy': 1e10, 'AdvEvals': 1e10, 'DiffEvals': 1e10, 'RxEvals': 1e10}

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
        namefile.write("   brussa = 0.6\n")
        namefile.write("   brussb = 2.0\n")
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
    return stats


# utility routine to run a single C++ test, storing the run options and solver statistics
def runtest(exe='./bin/advection_diffusion_reaction', probtype='AdvDiffRx', inttype='ARK', ststype=None, extststype=None, table_id=0, c=1e-2, d=1e-1, eps=1e-2, nx=512, rtol=1e-4, atol=1e-9, fixedh=0.0, maxl=0, nout=20, showcommand=False):
    stats = {'probtype': probtype, 'inttype': inttype, 'ststype': ststype, 'extststype': extststype, 'table_id': table_id, 'c': c, 'd': d, 'eps': eps, 'nx': nx, 'rtol': rtol, 'atol': atol, 'fixedh': fixedh, 'maxl': maxl, 'nout': nout, 'ReturnCode': 1, 'Steps': 1e10, 'Fails': 1e10, 'Accuracy': 1e10, 'AdvEvals': 1e10, 'DiffEvals': 1e10, 'RxEvals': 1e10}
    runcommand = "%s --c %e --d %e --eps %e --nx %d --rtol %e --atol %e --fixed_h %e --maxl %d --nout %d --calc_error" % (exe, c, d, eps, nx, rtol, atol, fixedh, maxl, nout) + int_method(probtype, inttype, ststype, extststype, table_id)

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
Executable = './bin/advection_diffusion_reaction'
PIROCKExecutable = './bin/advection_diffusion_reaction_pirock'
AdvDiffRxSolvers = [['ARK', None, None, 1],
                    ['ARK', None, None, 2],
                    ['ExtSTS', 'RKC', 'ARS', None],
                    ['ExtSTS', 'RKL', 'ARS', None],
                    ['ExtSTS', 'RKC', 'Giraldo', None],
                    ['ExtSTS', 'RKL', 'Giraldo', None]]
AdvDiffSolvers = [['ARK', None, None, 1],
                  ['ARK', None, None, 2],
                  ['ExtSTS', 'RKC', 'ARS', None],
                  ['ExtSTS', 'RKL', 'ARS', None],
                  ['ExtSTS', 'RKC', 'Giraldo', None],
                  ['ExtSTS', 'RKL', 'Giraldo', None],
                  ['ExtSTS', 'RKC', 'Ralston', None],
                  ['ExtSTS', 'RKL', 'Ralston', None]]
RxDiffSolvers = [['ARK', None, None, 5],
                 ['ARK', None, None, 6],
                 ['ExtSTS', 'RKC', 'ARS', None],
                 ['ExtSTS', 'RKL', 'ARS', None],
                 ['ExtSTS', 'RKC', 'Giraldo', None],
                 ['ExtSTS', 'RKL', 'Giraldo', None],
                 ['ExtSTS', 'RKC', 'SSPSDIRK2', None],
                 ['ExtSTS', 'RKL', 'SSPSDIRK2', None]]
StrangSolvers = [['Strang', 'RKC', None, None],
                 ['Strang', 'RKL', None, None]]
c = 1e-2
d = 1e-1
eps = 1e-2
nx = 512
rtol = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
atol = 1e-11
fixed_maxl = 500
fixedh = 0.1 / np.array([4, 8, 16, 32, 64, 128, 256], dtype=float)
nout = 1

# Advection-diffusion-reaction tests
if (DoAdvDiffRx):

    # generate reference solution for PIROCK error
    generate_reference(Executable, probtype='AdvDiffRx', c=c, d=d, eps=eps, nx=nx)

    if (DoFixedTests):

        Stats = []
        for solver in AdvDiffRxSolvers:
            for h in fixedh:
                Stats.append(runtest(Executable, probtype='AdvDiffRx', inttype=solver[0],
                                     ststype=solver[1], extststype=solver[2],
                                     table_id=solver[3], c=c, d=d, eps=eps,
                                     nx=nx, fixedh=h, rtol=1e-3*(h*h), maxl=fixed_maxl, nout=nout))
        for solver in StrangSolvers:
            for h in fixedh:
                Stats.append(runtest(Executable, probtype='AdvDiffRx', inttype=solver[0],
                                     ststype=solver[1], extststype=solver[2],
                                     table_id=solver[3], c=c, d=d, eps=eps,
                                     nx=nx, fixedh=h, rtol=1e-3*(h*h), maxl=fixed_maxl, nout=nout))
        for h in fixedh:
            Stats.append(runtest_pirock(PIROCKExecutable, probtype='AdvDiffRx', c=c, d=d, eps=eps,
                                 nx=nx, rtol=1e-3*(h*h), fixedh=h))

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
                                     ststype=solver[1], extststype=solver[2],
                                     table_id=solver[3], c=c, d=d,
                                     eps=eps, nx=nx, rtol=rt, atol=atol, fixedh=0.0, nout=nout))
        for rt in rtol:
            Stats.append(runtest_pirock(PIROCKExecutable, probtype='AdvDiffRx', c=c, d=d,
                                        eps=eps, nx=nx, rtol=rt, atol=atol, fixedh=0.0))

        Df = pd.DataFrame.from_records(Stats)
        print("Adaptive step AdvDiffRx test Df:")
        print(Df)
        print("Saving as Excel")
        Df.to_excel('AdvDiffRx-adapt.xlsx', index=False)

# Advection-diffusion tests
if (DoAdvDiff):

    # generate reference solution for PIROCK error
    generate_reference(Executable, probtype='AdvDiff', c=c, d=d, eps=eps, nx=nx)

    if (DoFixedTests):

        Stats = []
        for solver in AdvDiffSolvers:
            for h in fixedh:
                Stats.append(runtest(Executable, probtype='AdvDiff', inttype=solver[0],
                                     ststype=solver[1], extststype=solver[2],
                                     table_id=solver[3], c=c, d=d,
                                     nx=nx, fixedh=h, rtol=1e-3*(h*h), maxl=fixed_maxl, nout=nout))
        for solver in StrangSolvers:
            for h in fixedh:
                Stats.append(runtest(Executable, probtype='AdvDiff', inttype=solver[0],
                                     ststype=solver[1], extststype=solver[2],
                                     table_id=solver[3], c=c, d=d,
                                     nx=nx, fixedh=h, rtol=1e-3*(h*h), maxl=fixed_maxl, nout=nout))
        for h in fixedh:
            Stats.append(runtest_pirock(PIROCKExecutable, probtype='AdvDiff', c=c, d=d,
                                        nx=nx, rtol=1e-3*(h*h), fixedh=h))

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
                                     ststype=solver[1], extststype=solver[2],
                                     table_id=solver[3], c=c, d=d,
                                     nx=nx, rtol=rt, atol=atol, fixedh=0.0, nout=nout))
        for rt in rtol:
            Stats.append(runtest_pirock(PIROCKExecutable, probtype='AdvDiff', c=c, d=d,
                                        nx=nx, rtol=rt, atol=atol, fixedh=0.0))

        Df = pd.DataFrame.from_records(Stats)
        print("Adaptive step AdvDiff test Df:")
        print(Df)
        print("Saving as Excel")
        Df.to_excel('AdvDiff-adapt.xlsx', index=False)

# Reaction-diffusion tests
if (DoRxDiff):

    # generate reference solution for PIROCK error
    generate_reference(Executable, probtype='RxDiff', c=c, d=d, eps=eps, nx=nx)

    if (DoFixedTests):

        Stats = []
        for solver in RxDiffSolvers:
            for h in fixedh:
                Stats.append(runtest(Executable, probtype='RxDiff', inttype=solver[0],
                                     ststype=solver[1], extststype=solver[2],
                                     table_id=solver[3], d=d, eps=eps,
                                     nx=nx, fixedh=h, rtol=1e-3*(h*h), maxl=fixed_maxl, nout=nout))
        for solver in StrangSolvers:
            for h in fixedh:
                Stats.append(runtest(Executable, probtype='RxDiff', inttype=solver[0],
                                     ststype=solver[1], extststype=solver[2],
                                     table_id=solver[3], d=d, eps=eps,
                                     nx=nx, fixedh=h, rtol=1e-3*(h*h), maxl=fixed_maxl, nout=nout))
        for h in fixedh:
            Stats.append(runtest_pirock(PIROCKExecutable, probtype='RxDiff', d=d, eps=eps,
                                        nx=nx, rtol=1e-3*(h*h), fixedh=h))

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
                                     ststype=solver[1], extststype=solver[2],
                                     table_id=solver[3], d=d, eps=eps,
                                     nx=nx, rtol=rt, atol=atol, fixedh=0.0, nout=nout))
        for rt in rtol:
            Stats.append(runtest_pirock(PIROCKExecutable, probtype='RxDiff', d=d,
                                        eps=eps, nx=nx, rtol=rt, atol=atol, fixedh=0.0))

        Df = pd.DataFrame.from_records(Stats)
        print("Adaptive step RxDiff test Df:")
        print(Df)
        print("Saving as Excel")
        Df.to_excel('RxDiff-adapt.xlsx', index=False)

# end of script