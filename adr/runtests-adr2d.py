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
import os


# set maximum runtime before a test is considered a failure and canceled
maxruntime = 30*60 # 30 minutes, converted to seconds

#####################
# utility routines

# utility routine to set C++ executable inputs for running a specific integration type
def int_method(probtype, implicitrx, inttype, ststype, extststype, table_id):
    flags = ""
    if (probtype == "RxDiff"):
        flags += " --no-advection"
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
        flags += " --implicit-reaction"

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
        else:
            msg = """
            Error: invalid extsts type
            Valid choices are: ARS, Giraldo, Ralston, Heun-Euler, SSPSDIRK2
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


# utility routine to compute solution error between two files
def calc_error(nx, solfile, reffile):
    soldata = np.loadtxt(solfile)
    refdata = np.loadtxt(reffile)
    uerr = soldata[1:]-refdata[1:]
    return np.sqrt(np.dot(uerr,uerr) / nx / nx / 2)

# utility routine to generate an ADR reference solution for a given problem configuration
def generate_ADR_reference(exe='./bin/advection_diffusion_reaction_2D', cux=-0.5, cuy=1.0, cvx=0.4, cvy=0.7, d=1e-2, A=1.3, B=1.0, nx=400, ny=400, tf=1.0):
    if not os.path.isfile("adr_reference.dat"):
        runcommand = "%s --cux %e --cuy %e --cvx %e --cvy %e --d %e --A %e --B %e --nx %d --ny %d --tf %e --nout 1 --calc_error --write_solution" % (exe, cux, cuy, cvx, cvy, d, A, B, nx, ny, tf) + int_method("AdvDiffRx", False, "ARK", None, None, 1)
        print("Generating ADR reference solution with command: " + runcommand)
        result = subprocess.run(shlex.split(runcommand), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode())
        os.rename('reference.dat', 'adr_reference.dat')

# utility routine to generate a RD reference solution for a given problem configuration
def generate_RD_reference(exe='./bin/advection_diffusion_reaction_2D', d=0.1, A=1.3, B=2.e7, nx=200, ny=200, tf=2.0):
    if not os.path.isfile("rd_reference.dat"):
        runcommand = "%s --d %e --A %e --B %e --nx %d --ny %d --tf %e --cux 0.0 --cuy 0.0 --cvx 0.0 --cvy 0.0 --nout 1 --calc_error --write_solution" % (exe, d, A, B, nx, ny, tf) + int_method("RxDiff", True, "ARK", None, None, 1)
        print("Generating RD reference solution with command: " + runcommand)
        result = subprocess.run(shlex.split(runcommand), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode())
        os.rename('reference.dat', 'rd_reference.dat')

# utility routine to run the PIROCK ADR executable
def runtest_ADR_pirock(exe='./bin/advection_diffusion_reaction_2D_pirock', cux=-0.5, cuy=1.0, cvx=0.4, cvy=0.7, d=1e-2, A=1.3, B=1.0, nx=400, ny=400, tf=1.0, rtol=1e-4, atol=1e-9, fixedh=0.0, showcommand=False, showoutput=False):
    if (nx != 400):
        raise(ValueError, "To run without 400 spatial nodes, need to edit/recompile pb_bruss2dadv.f (and this error check)")
    stats = {'probtype': 'AdvDiffRx', 'implicitrx': False, 'inttype': 'PIROCK', 'ststype': None, 'extststype': None, 'table_id': 0, 'cux': cux, 'cuy': cuy, 'cvx': cvx, 'cvy': cvy, 'd': d, 'A': A, 'B': B, 'nx': nx, 'ny': ny, 'tf': tf, 'rtol': rtol, 'atol': atol, 'fixedh': fixedh, 'ReturnCode': 1, 'Steps': np.nan, 'Fails': np.nan, 'Accuracy': np.nan, 'AdvEvals': np.nan, 'DiffEvals': np.nan, 'RxEvals': np.nan}

    # modify parameters in namelist file and turn on/off advection/reaction
    with open("adr_2D_pirock_params.txt",'w') as namefile:
        namefile.write("&inputs\n")
        namefile.write("   alf = " + str(d) + "\n")
        namefile.write("   uxadv = " + str(cux) + "\n")
        namefile.write("   uyadv = " + str(cuy) + "\n")
        namefile.write("   vxadv = " + str(cvx) + "\n")
        namefile.write("   vyadv = " + str(cvy) + "\n")
        namefile.write("   brussa = " + str(A) + "\n")
        namefile.write("   brussb = " + str(B) + "\n")
        namefile.write("   atol = " + str(atol) + "\n")
        namefile.write("   rtol = " + str(rtol) + "\n")
        namefile.write("   h = " + str(fixedh) + "\n")
        namefile.write("/\n")

    # run the test (and determine runtime)
    tstart = time.perf_counter()
    try:
        result = subprocess.run(shlex.split(exe), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=maxruntime)
        stats['ReturnCode'] = result.returncode
        # print output to screen if requested
        if (showoutput):
            print(result.stdout.decode())
    except subprocess.TimeoutExpired:
        print('Test exceeded maximum run time and was terminated')
        stats['ReturnCode'] = -1
    stats['RunTime'] = time.perf_counter() - tstart

    # process the run results
    if (stats['ReturnCode'] != 0):
        print("Run command " + exe + " FAILURE: " + str(stats['ReturnCode']))
        print(result.stderr)
    else:
        if(showcommand):
            print("Run command: " + exe + " SUCCESS\n")

        # compute solution error and store this in the stats
        stats['Accuracy'] = calc_error(nx, "sol.dat", "adr_reference.dat")
        if os.path.isfile("sol.dat"):
            os.remove("sol.dat")

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


# utility routine to run the PIROCK RD executable
def runtest_RD_pirock(exe='./bin/reaction_diffusion_2D_pirock', d=1e-1, A=1.3, B=2.e7, nx=200, ny=200, tf=2.0, rtol=1e-4, atol=1e-9, fixedh=0.0, showcommand=False, showoutput=False):
    if (nx != 200):
        raise(ValueError, "To run without 200 spatial nodes, need to edit/recompile pb_bruss2dreac.f (and this error check)")
    stats = {'probtype': 'RxDiff', 'implicitrx': True, 'inttype': 'PIROCK', 'ststype': None, 'extststype': None, 'table_id': 0, 'cux': 0.0, 'cuy': 0.0, 'cvx': 0.0, 'cvy': 0.0, 'd': d, 'A': A, 'B': B, 'nx': nx, 'ny': ny, 'tf': tf, 'rtol': rtol, 'atol': atol, 'fixedh': fixedh, 'ReturnCode': 1, 'Steps': np.nan, 'Fails': np.nan, 'Accuracy': np.nan, 'AdvEvals': np.nan, 'DiffEvals': np.nan, 'RxEvals': np.nan}

    # modify parameters in namelist file and turn on/off advection/reaction
    with open("rd_2D_pirock_params.txt",'w') as namefile:
        namefile.write("&inputs\n")
        namefile.write("   alf = " + str(d) + "\n")
        namefile.write("   brussa = " + str(A) + "\n")
        namefile.write("   brussb = " + str(B) + "\n")
        namefile.write("   atol = " + str(atol) + "\n")
        namefile.write("   rtol = " + str(rtol) + "\n")
        namefile.write("   h = " + str(fixedh) + "\n")
        namefile.write("/\n")

    # run the test (and determine runtime)
    tstart = time.perf_counter()
    try:
        result = subprocess.run(shlex.split(exe), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=maxruntime)
        stats['ReturnCode'] = result.returncode
        # print output to screen if requested
        if (showoutput):
            print(result.stdout.decode())
    except subprocess.TimeoutExpired:
        print('Test exceeded maximum run time and was terminated')
        stats['ReturnCode'] = -1
    stats['RunTime'] = time.perf_counter() - tstart

    # process the run results
    if (stats['ReturnCode'] != 0):
        print("Run command " + exe + " FAILURE: " + str(stats['ReturnCode']))
        print(result.stderr)
    else:
        if(showcommand):
            print("Run command: " + exe + " SUCCESS\n")

        # compute solution error and store this in the stats
        stats['Accuracy'] = calc_error(nx, "sol.dat", "rd_reference.dat")
        if os.path.isfile("sol.dat"):
            os.remove("sol.dat")

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
def runtest(exe='./bin/advection_diffusion_reaction_2D', probtype='AdvDiffRx', implicitrx=False, inttype='ARK', ststype=None, extststype=None, table_id=0, cux=-0.5, cuy=1.0, cvx=0.4, cvy=0.7, d=1e-2, A=1.3, B=1.0, nx=400, ny=400, rtol=1e-4, atol=1e-9, fixedh=0.0, showcommand=False, showoutput=False):
    stats = {'probtype': probtype, 'implicitrx': implicitrx, 'inttype': inttype, 'ststype': ststype, 'extststype': extststype, 'table_id': table_id, 'cux': cux, 'cuy': cuy, 'cvx': cvx, 'cvy': cvy, 'd': d, 'A': A, 'B': B, 'nx': nx, 'ny': ny, 'rtol': rtol, 'atol': atol, 'fixedh': fixedh, 'ReturnCode': 1, 'Steps': np.nan, 'Fails': np.nan, 'Accuracy': np.nan, 'AdvEvals': np.nan, 'DiffEvals': np.nan, 'RxEvals': np.nan}
    runcommand = "%s --cux %e --cuy %e --cvx %e --cvy %e --d %e --A %e --B %e --nx %d --ny %d --rtol %e --atol %e --fixed_h %e --nout 1 --output 1 --maxsteps 1000000" % (exe, cux, cuy, cvx, cvy, d, A, B, nx, ny, rtol, atol, fixedh) + int_method(probtype, implicitrx, inttype, ststype, extststype, table_id)

    # run the test (and determine runtime)
    tstart = time.perf_counter()
    try:
        result = subprocess.run(shlex.split(runcommand), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=maxruntime)
        stats['ReturnCode'] = result.returncode
        # print output to screen if requested
        if (showoutput):
            print(result.stdout.decode())
    except subprocess.TimeoutExpired:
        print('Test exceeded maximum run time and was terminated')
        print('Run command was: ' + runcommand)
        stats['ReturnCode'] = -1
    stats['RunTime'] = time.perf_counter() - tstart

    # process the run results
    if (stats['ReturnCode'] != 0):
        print("Run command " + runcommand + " FAILURE: " + str(stats['ReturnCode']))
    else:
        if (showcommand):
            print("Run command " + runcommand + " SUCCESS")

        # compute solution error and store this in the stats
        if (probtype == "AdvDiffRx"):
            stats['Accuracy'] = calc_error(nx, "solution.dat", "adr_reference.dat")
        else:
            stats['Accuracy'] = calc_error(nx, "solution.dat", "rd_reference.dat")
        if os.path.isfile("solution.dat"):
            os.remove("solution.dat")

        lines = str(result.stdout).split('\\n')
        if (inttype == "ARK"):
            for line in lines:
                txt = line.split()
                if ("Steps" in txt):
                    stats['Steps'] = int(txt[2])
                if (("Error" in txt) and ("test" in txt) and ("fails" in txt)):
                    stats['Fails'] = int(txt[4])
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
        elif (inttype == "Strang"):
            for line in lines:
                txt = line.split()
                if ("Steps" in txt and stats['Steps'] == 1e10):
                    stats['Steps'] = int(txt[2])
                if (("Error" in txt) and ("test" in txt) and ("fails" in txt)):
                    stats['Fails'] = int(txt[4])
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
DoImplicitRx = True
DoExplicitRx = True
DoADRFixedTests = True
DoRDFixedTests = True
DoADRAdaptiveTests = True
DoRDAdaptiveTests = True
ShowCommand = True
ShowOutput = True

# Shared testing parameters: [inttype, ststype, extststype, table_id]
AdvDiffRxSolvers = [['ARK', None, None, 1],
                    ['ARK', None, None, 2]]
AdvDiffRxSolversExpOnly = [
                           ['ExtSTS', 'RKC', 'ARS', None],
                           ['ExtSTS', 'RKL', 'ARS', None],
                           ['ExtSTS', 'RKC', 'Giraldo', None],
                           ['ExtSTS', 'RKL', 'Giraldo', None],
                           ['ExtSTS', 'RKC', 'Ralston', None],
                           ['ExtSTS', 'RKL', 'Ralston', None],
                           ['ExtSTS', 'RKC', 'Heun-Euler', None],
                           ['ExtSTS', 'RKL', 'Heun-Euler', None]]
ADRStrangSolvers = [['Strang', 'RKC', None, None],
                    ['Strang', 'RKL', None, None]]
RxDiffSolvers = [['ARK', None, None, 1],
                 ['ARK', None, None, 2],
                 ['ARK', None, None, 5],
                 ['ExtSTS', 'RKC', 'ARS', None],
                 ['ExtSTS', 'RKL', 'ARS', None],
                 ['ExtSTS', 'RKC', 'Giraldo', None],
                 ['ExtSTS', 'RKL', 'Giraldo', None],
                 ['ExtSTS', 'RKC', 'SSPSDIRK2', None],
                 ['ExtSTS', 'RKL', 'SSPSDIRK2', None]]
RDStrangSolvers = [['Strang', 'RKC', None, None],
                   ['Strang', 'RKL', None, None]]

# Advection-diffusion-reaction tests
if (DoADRFixedTests or DoADRAdaptiveTests):

    # generate reference solution
    generate_ADR_reference()

    if (DoADRFixedTests):
        Stats = []

        # set step sizes for fixed-step ADR tests
        fixedh        = 0.01 / np.array([16, 32, 64, 128], dtype=float)
        fixedh_strang = 0.01 / np.array([16, 32, 64, 128], dtype=float)
        fixedh_pirock = 0.01 / np.array([16, 32, 64, 128], dtype=float)

        # run solvers, treating reactions implicitly
        if (DoImplicitRx):
            for solver in AdvDiffRxSolvers:
                for h in fixedh:
                    Stats.append(runtest(probtype='AdvDiffRx', inttype=solver[0], ststype=solver[1],
                                        extststype=solver[2], table_id=solver[3],
                                        implicitrx=True, fixedh=h, rtol=max(1e-3*(h*h),1e-9),
                                        showcommand=ShowCommand, showoutput=ShowOutput))
            for solver in ADRStrangSolvers:
                for h in fixedh_strang:
                    Stats.append(runtest(probtype='AdvDiffRx', inttype=solver[0], ststype=solver[1],
                                        extststype=solver[2], table_id=solver[3], implicitrx=True,
                                        fixedh=h, rtol=max(1e-3*(h*h),1e-9),
                                        showcommand=ShowCommand, showoutput=ShowOutput))

        # run solvers, treating reactions explicitly
        if (DoExplicitRx):
            for solver in AdvDiffRxSolvers:
                for h in fixedh:
                    Stats.append(runtest(probtype='AdvDiffRx', inttype=solver[0], ststype=solver[1],
                                        extststype=solver[2], table_id=solver[3], implicitrx=False,
                                        fixedh=h, rtol=max(1e-3*(h*h),1e-9),
                                        showcommand=ShowCommand, showoutput=ShowOutput))
            for solver in AdvDiffRxSolversExpOnly:
                for h in fixedh:
                    Stats.append(runtest(probtype='AdvDiffRx', inttype=solver[0], ststype=solver[1],
                                        extststype=solver[2], table_id=solver[3], implicitrx=False,
                                        fixedh=h, rtol=max(1e-3*(h*h),1e-9),
                                        showcommand=ShowCommand, showoutput=ShowOutput))
            for solver in ADRStrangSolvers:
                for h in fixedh_strang:
                    Stats.append(runtest(probtype='AdvDiffRx', inttype=solver[0], ststype=solver[1],
                                        extststype=solver[2], table_id=solver[3], implicitrx=False,
                                        fixedh=h, rtol=max(1e-3*(h*h),1e-9),
                                        showcommand=ShowCommand, showoutput=ShowOutput))

            for h in fixedh_pirock:
                Stats.append(runtest_ADR_pirock(rtol=max(1e-3*(h*h),1e-9), fixedh=h,
                                                showcommand=ShowCommand, showoutput=ShowOutput))

        # store results
        Df = pd.DataFrame.from_records(Stats)
        print("Fixed step AdvDiffRx2D test Df:")
        print(Df)
        print("Saving as Excel")
        Df.to_excel('AdvDiffRx2D-fixed.xlsx', index=False)

    if (DoADRAdaptiveTests):
        Stats = []

        # set tolerances for adaptive ADR tests
        rtol = np.logspace(-2.5, -6.5, 7)
        atol = 1e-11

        # run solvers, treating reactions implicitly
        for solver in AdvDiffRxSolvers:
            for rt in rtol:
                Stats.append(runtest(probtype='AdvDiffRx', inttype=solver[0],
                                     ststype=solver[1], extststype=solver[2],
                                     table_id=solver[3], implicitrx=True, rtol=rt,
                                     atol=atol, showcommand=ShowCommand, showoutput=ShowOutput))

        # run solvers, treating reactions explicitly
        for solver in AdvDiffRxSolvers:
            for rt in rtol:
                Stats.append(runtest(probtype='AdvDiffRx', inttype=solver[0],
                                     ststype=solver[1], extststype=solver[2],
                                     table_id=solver[3], implicitrx=False, rtol=rt,
                                     atol=atol, showcommand=ShowCommand, showoutput=ShowOutput))
        for solver in AdvDiffRxSolversExpOnly:
            for rt in rtol:
                Stats.append(runtest(probtype='AdvDiffRx', inttype=solver[0],
                                     ststype=solver[1], extststype=solver[2],
                                     table_id=solver[3], implicitrx=False, rtol=rt,
                                     atol=atol, showcommand=ShowCommand, showoutput=ShowOutput))

        for rt in rtol:
            Stats.append(runtest_ADR_pirock(rtol=rt, atol=atol, showcommand=ShowCommand, showoutput=ShowOutput))

        Df = pd.DataFrame.from_records(Stats)
        print("Adaptive step AdvDiffRx2D test Df:")
        print(Df)
        print("Saving as Excel")
        Df.to_excel('AdvDiffRx2D-adapt.xlsx', index=False)


# Reaction-diffusion tests
if (DoRDFixedTests or DoRDAdaptiveTests):

    # shared problem parameters
    d = 0.1
    cux = 0.0
    cuy = 0.0
    cvx = 0.0
    cvy = 0.0
    A = 1.3
    B = 2.e7

    # generate reference solution
    generate_RD_reference(d=d, A=A, B=B)

    if (DoRDFixedTests):
        Stats = []

        # set step sizes for fixed-step RD tests
        fixedh        = 0.01 / np.array([16, 32, 64, 128], dtype=float)
        fixedh_strang = 0.01 / np.array([16, 32, 64, 128], dtype=float)
        fixedh_pirock = 0.01 / np.array([16, 32, 64, 128], dtype=float)

        for solver in RxDiffSolvers:
            for h in fixedh:
                Stats.append(runtest(probtype='RxDiff', inttype=solver[0],
                                     ststype=solver[1], extststype=solver[2],
                                     table_id=solver[3], implicitrx=True,
                                     cux=cux, cuy=cuy, cvx=cvx, cvy=cvy,
                                     d=d, A=A, B=B, fixedh=h, rtol=max(1e-3*(h*h),1e-9)))
        for solver in RDStrangSolvers:
            for h in fixedh_strang:
                Stats.append(runtest(probtype='RxDiff', inttype=solver[0],
                                    ststype=solver[1], extststype=solver[2],
                                    table_id=solver[3], implicitrx=True,
                                    cux=cux, cuy=cuy, cvx=cvx, cvy=cvy,
                                    d=d, A=A, B=B, fixedh=h, rtol=max(1e-3*(h*h),1e-9)))
        for h in fixedh_pirock:
            Stats.append(runtest_RD_pirock(d=d, A=A, B=B, rtol=max(1e-3*(h*h),1e-9), fixedh=h))

        Df = pd.DataFrame.from_records(Stats)
        print("Fixed step RxDiff2D test Df:")
        print(Df)
        print("Saving as Excel")
        Df.to_excel('RxDiff2D-fixed.xlsx', index=False)

    if (DoRDAdaptiveTests):
        Stats = []

        # set tolerances for adaptive RD tests
        rtol = np.logspace(-2.5, -6.5, 7)
        atol = 1e-11

        for solver in RxDiffSolvers:
            for rt in rtol:
                Stats.append(runtest(probtype='RxDiff', inttype=solver[0],
                                     ststype=solver[1], extststype=solver[2],
                                     table_id=solver[3], implicitrx=True,
                                     cux=cux, cuy=cuy, cvx=cvx, cvy=cvy,
                                     d=d, A=A, B=B, rtol=rt, atol=atol))
        for rt in rtol:
            Stats.append(runtest_RD_pirock(d=d, A=A, B=B, rtol=rt, atol=atol))

        Df = pd.DataFrame.from_records(Stats)
        print("Adaptive step RxDiff2D test Df:")
        print(Df)
        print("Saving as Excel")
        Df.to_excel('RxDiff2D-adapt.xlsx', index=False)

# end of script