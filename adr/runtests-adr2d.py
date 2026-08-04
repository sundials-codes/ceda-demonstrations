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
import multiprocessing
import tempfile


# set maximum runtime before a test is considered a failure and canceled
maxruntime = 90*60 # 90 minutes, converted to seconds

# Directory from which the script is being run
topdir = os.getcwd()

# Maximum number of processes to use for multiprocessor tests
maxprocs = multiprocessing.cpu_count()//2 - 1

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
        #flags += " --implicit-reaction --maxnewt 10 --nlscoef 0.01 --error_bias 2.0"
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
        elif (extststype == "SSP22"):
            flags += " --extsts_method 5"
        elif (extststype == "SSP32"):
            flags += " --extsts_method 6"
        elif (extststype == "SSP42"):
            flags += " --extsts_method 7"
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
            Valid choices are: ARS, Giraldo, Ralston, Heun-Euler, SSPSDIRK2, IRK21a, ESDIRK34a, ERK22a, ERK22b, MERK21, MERK32, MRISR21, SSP22, SSP32, SSP42
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
    if (soldata.ndim > 1):
        sol = soldata[-1,1:]
    else:
        sol = soldata[1:]
    if (refdata.ndim > 1):
        ref = refdata[-1,1:]
    else:
        ref = refdata[1:]
    uerr = sol-ref
    return np.sqrt(np.dot(uerr,uerr) / nx / nx / 2)

# utility routine to generate an ADR reference solution for a given problem configuration
def generate_ADR_reference(exe='./bin/advection_diffusion_reaction_2D', cux=-0.5, cuy=1.0, cvx=0.4, cvy=0.7, d=1e-2, A=1.3, B=1.0, nx=400, ny=400, tf=1.0):
    runcommand = "%s --cux %e --cuy %e --cvx %e --cvy %e --d %e --A %e --B %e --nx %d --ny %d --tf %e --rtol 1e-6 --atol 1e-11 --nout 1 --calc_error --write_solution" % (exe, cux, cuy, cvx, cvy, d, A, B, nx, ny, tf) + int_method("AdvDiffRx", False, "ARK", None, None, 1)
    print("Generating ADR reference solution")
    result = subprocess.run(shlex.split(runcommand), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(result.stdout.decode())
    os.rename('reference.dat', 'adr_reference.dat')

# utility routine to run the PIROCK ADR executable
def runtest_ADR_pirock(exe='./bin/advection_diffusion_reaction_2D_pirock', cux=-0.5, cuy=1.0, cvx=0.4, cvy=0.7, d=1e-2, A=1.3, B=1.0, nx=400, ny=400, tf=1.0, rtol=1e-4, atol=1e-9, fixedh=0.0, nout=20, showcommand=False, showoutput=False):
    if (nx != 400):
        raise(ValueError, "To run without 400 spatial nodes, need to edit/recompile pb_bruss2dadv.f (and this error check)")
    stats = {'probtype': 'AdvDiffRx', 'implicitrx': False, 'inttype': 'PIROCK', 'ststype': None, 'extststype': None, 'table_id': None, 'cux': cux, 'cuy': cuy, 'cvx': cvx, 'cvy': cvy, 'd': d, 'A': A, 'B': B, 'nx': nx, 'ny': ny, 'tf': tf, 'rtol': rtol, 'atol': atol, 'fixedh': fixedh, 'nout': nout, 'ReturnCode': 1, 'Steps': np.nan, 'Fails': np.nan, 'Accuracy': np.nan, 'AdvEvals': np.nan, 'DiffEvals': np.nan, 'RxEvals': np.nan, 'AdvTime': np.nan, 'DiffTime': np.nan, 'RxTime': np.nan, 'RxJacTime': np.nan}

    # create a temporary directory to run the test
    with tempfile.TemporaryDirectory() as tempdir:
        pwd = os.getcwd()
        os.chdir(tempdir)

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
            namefile.write("   tend = " + str(tf) + "\n")
            namefile.write("   nout = " + str(nout) + "\n")
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
        if(showcommand):
            if (stats['ReturnCode'] != 0):
                print("Run command " + exe + " FAILURE: " + str(stats['ReturnCode']))
            else:
                print("Run command: " + exe + " SUCCESS\n")

        if (stats['ReturnCode'] == 0):
            # compute solution error and store this in the stats
            stats['Accuracy'] = calc_error(nx, "sol.dat", topdir + "/adr_reference.dat")

            # get remaining stats from stdout
            lines = str(result.stdout).split('\\n')
            for line in lines:
                if 'Number of f evaluations' in line:
                    txt = line.split()
                    stats['DiffEvals'] = abs(int(txt[4]))
                    stats['AdvEvals'] = abs(int(txt[7]))
                    stats['RxEvals'] = abs(int(txt[7]))
                    stats['Steps'] = abs(int(txt[9]))
                    stats['Fails'] = abs(int(txt[13]))
        os.chdir(pwd)
    return stats


# utility routine to run a single C++ test, storing the run options and solver statistics
def runtest(exe='./bin/advection_diffusion_reaction_2D', probtype='AdvDiffRx', implicitrx=False, inttype='ARK', ststype=None, extststype=None, table_id=0, cux=-0.5, cuy=1.0, cvx=0.4, cvy=0.7, d=1e-2, A=1.3, B=1.0, nx=400, ny=400, tf=1.0, rtol=1e-4, atol=1e-9, fixedh=0.0, maxl=0, nout=20, showcommand=False, showoutput=False):
    stats = {'probtype': probtype, 'implicitrx': implicitrx, 'inttype': inttype, 'ststype': ststype, 'extststype': extststype, 'table_id': table_id, 'cux': cux, 'cuy': cuy, 'cvx': cvx, 'cvy': cvy, 'd': d, 'A': A, 'B': B, 'nx': nx, 'ny': ny, 'tf': tf, 'rtol': rtol, 'atol': atol, 'fixedh': fixedh, 'maxl': maxl, 'nout': nout, 'ReturnCode': 1, 'Steps': np.nan, 'Fails': np.nan, 'Accuracy': np.nan, 'AdvEvals': np.nan, 'DiffEvals': np.nan, 'RxEvals': np.nan, 'AdvTime': np.nan, 'DiffTime': np.nan, 'RxTime': np.nan, 'RxJacTime': np.nan}
    runcommand = "%s --cux %e --cuy %e --cvx %e --cvy %e --d %e --A %e --B %e --nx %d --ny %d --tf %e --rtol %e --atol %e --fixed_h %e --maxl %d --nout %d --output 3 --maxsteps 10000000" % (exe, cux, cuy, cvx, cvy, d, A, B, nx, ny, tf, rtol, atol, fixedh, maxl, nout) + int_method(probtype, implicitrx, inttype, ststype, extststype, table_id)

    # create a temporary directory to run the test
    with tempfile.TemporaryDirectory() as tempdir:
        pwd = os.getcwd()
        os.chdir(tempdir)

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
        if (showcommand):
            if (stats['ReturnCode'] != 0):
                print("Run command " + runcommand + " FAILURE: " + str(stats['ReturnCode']))
            else:
                print("Run command " + runcommand + " SUCCESS")

        if (stats['ReturnCode'] == 0):
            # compute solution error and store this in the stats
            stats['Accuracy'] = calc_error(nx, "solution.dat", topdir + "/adr_reference.dat")

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
                            if (implicitrx == False):
                                stats['RxEvals'] = int(txt[4])
                    elif (("Implicit" in txt) and ("RHS" in txt) and ("evals" in txt)):
                        if (probtype == "AdvDiffRx" or probtype == "RxDiff"):
                            stats['DiffEvals'] = int(txt[4])
                            if (implicitrx == True):
                                stats['RxEvals'] = int(txt[4])
                    if ("fDtime" in txt):
                        stats['DiffTime'] = float(txt[2])
                    if ("fAtime" in txt):
                        stats['AdvTime'] = float(txt[2])
                    if ("fRtime" in txt):
                        stats['RxTime'] = float(txt[2])
                    if ("JRtime" in txt):
                        stats['RxJacTime'] = float(txt[2])
            elif (inttype == "Strang"):
                for line in lines:
                    txt = line.split()
                    if ("Steps" in txt and np.isnan(stats['Steps'])):
                        stats['Steps'] = int(txt[2])
                    if (("Error" in txt) and ("test" in txt) and ("fails" in txt)):
                        stats['Fails'] = int(txt[4])
                    elif (("Explicit" in txt) and ("RHS" in txt) and ("evals" in txt)):
                        if (probtype == "AdvDiffRx"):
                            stats['AdvEvals'] = int(txt[5])
                            if (implicitrx == False):
                                stats['RxEvals'] = int(txt[5])
                        else:
                            stats['AdvEvals'] = 0
                    elif (("Implicit" in txt) and ("RHS" in txt) and ("evals" in txt)):
                        if (probtype == "AdvDiffRx" or probtype == "RxDiff"):
                            stats['RxEvals'] = int(txt[5])
                        else:
                            stats['RxEvals'] = 0
                    elif (("RHS" in txt) and ("fn" in txt) and ("evals" in txt) and ("LS" not in txt)):
                        stats['DiffEvals'] = int(txt[4])
                    if ("fDtime" in txt):
                        stats['DiffTime'] = float(txt[2])
                    if ("fAtime" in txt):
                        stats['AdvTime'] = float(txt[2])
                    if ("fRtime" in txt):
                        stats['RxTime'] = float(txt[2])
                    if ("JRtime" in txt):
                        stats['RxJacTime'] = float(txt[2])
            else:
                for line in lines:
                    txt = line.split()
                    if ("Steps" in txt and np.isnan(stats['Steps'])):
                        stats['Steps'] = int(txt[2])
                    if (("Error" in txt) and ("test" in txt) and ("fails" in txt) and np.isnan(stats['Fails'])):
                        stats['Fails'] = int(txt[4])
                    elif (("Explicit" in txt) and ("RHS" in txt) and ("evals" in txt)):
                        stats['AdvEvals'] = int(txt[6])
                        if (implicitrx == False):
                            stats['RxEvals'] = int(txt[6])
                    elif (("Implicit" in txt) and ("RHS" in txt) and ("evals" in txt)):
                        stats['RxEvals'] = int(txt[6])
                    elif (("RHS" in txt) and ("fn" in txt) and ("evals" in txt) and ("LS" not in txt)):
                        stats['DiffEvals'] = int(txt[4])
                    if ("fDtime" in txt):
                        stats['DiffTime'] = float(txt[2])
                    if ("fAtime" in txt):
                        stats['AdvTime'] = float(txt[2])
                    if ("fRtime" in txt):
                        stats['RxTime'] = float(txt[2])
                    if ("JRtime" in txt):
                        stats['RxJacTime'] = float(txt[2])
        os.chdir(pwd)
    return stats


def main():
    global topdir
    topdir = os.getcwd()

    #####################
    # testing setup

    # Flags to enable/disable categories of tests
    DoImplicitRx = False
    DoExplicitRx = True
    DoADRFixedTests = True
    DoADRAdaptiveTests = True
    ShowCommand = True
    ShowOutput = True
    ShowArgs = False

    # Shared testing parameters: [inttype, ststype, extststype, table_id]

    # ExtSTS solvers:
    #    ImEx: ARS, Giraldo, MRISR21
    #    Expl: Ralston, Heun-Euler, ERK22a, ERK22b, MERK21, MERK32, SSP22, SSP32, SSP42
    #    Impl: SSPSDIRK2, IRK21a, ESDIRK34a
    AdvDiffRxSolvers = []
    AdvDiffRxSolversExpRx = [['ARK', None, None, 2],
                             ['ExtSTS', 'RKC', 'Giraldo', None],
                             ['ExtSTS', 'RKL', 'Giraldo', None],
                             ['ExtSTS', 'RKC', 'SSP32', None],
                             ['ExtSTS', 'RKL', 'SSP32', None]]
    ADRStrangSolvers = [['Strang', 'RKC', None, None],
                        ['Strang', 'RKL', None, None]]

    # Advection-diffusion-reaction tests
    if (DoADRFixedTests or DoADRAdaptiveTests):

        for bctype in ['periodic', 'stationary']:

            # set appropriate executables for the boundary condition type
            if (bctype == 'periodic'):
                adrexe=topdir + '/bin/advection_diffusion_reaction_2D'
                adrpirockexe=topdir + '/bin/advection_diffusion_reaction_2D_pirock'
            else:
                adrexe=topdir + '/bin/advection_diffusion_reaction_2D_stationary'
                adrpirockexe=topdir + '/bin/advection_diffusion_reaction_2D_stationary_pirock'

            # shared problem parameters
            probtype='AdvDiffRx'
            cux=-0.5
            cuy=1.0
            cvx=0.4
            cvy=0.7
            A=1.0
            Bvals=[3.0, 3e1, 3e2]
            dvals=[1e-2, 1e-1]
            nx=400
            ny=400
            tf=5.0
            atol=1e-9
            fixed_maxl=500
            nout = 1

            # loop over diffusion coefficients
            FixedStats = []
            AdaptStats = []
            for d in dvals:
                for B in Bvals:

                    # generate reference solution
                    generate_ADR_reference(adrexe, cux, cuy, cvx, cvy, d, A, B, nx, ny, tf)

                    if (DoADRAdaptiveTests):
                        runtest_args = []
                        runtest_pirock_args = []

                        # set tolerances for adaptive ADR tests
                        rtol = np.logspace(-2.5, -6.5, 9)
                        atol = 1e-11
                        adapt_maxl = 0


                        # set up tests that treat reactions implicitly
                        for solver in AdvDiffRxSolvers:
                            for rt in rtol:
                                runtest_args.append((adrexe, probtype, True, solver[0], solver[1], solver[2],
                                                    solver[3], cux, cuy, cvx, cvy, d, A, B, nx, ny, tf, rt,
                                                    atol, 0.0, adapt_maxl, nout, ShowCommand, ShowOutput))

                        # set up tests that treat reactions explicitly
                        for solver in AdvDiffRxSolversExpRx:
                            for rt in rtol:
                                runtest_args.append((adrexe, probtype, False, solver[0], solver[1], solver[2],
                                                    solver[3], cux, cuy, cvx, cvy, d, A, B, nx, ny, tf, rt,
                                                    atol, 0.0, adapt_maxl, nout, ShowCommand, ShowOutput))

                        for rt in rtol:
                            runtest_pirock_args.append((adrpirockexe, cux, cuy, cvx, cvy, d, A, B, nx, ny, tf,
                                                        rt, atol, 0.0, nout, ShowCommand, ShowOutput))

                        # output argument lists if requested
                        if (ShowArgs):
                            print("ADR Adaptive Tests for d = " + str(d) + " and B = " + str(B) + ":")
                            print("runtest_args:")
                            for argset in runtest_args:
                                print(argset)
                            print("runtest_pirock_args:")
                            for argset in runtest_pirock_args:
                                print(argset)

                        # run tests in parallel
                        with multiprocessing.Pool(processes=maxprocs) as pool:
                            ar = []
                            for args in runtest_args:
                                ar.append(pool.apply_async(runtest, args=args))
                            for args in runtest_pirock_args:
                                ar.append(pool.apply_async(runtest_ADR_pirock, args=args))
                            for a in ar:
                                st = a.get()
                                print(st)
                                AdaptStats.append(st)

                    if (DoADRFixedTests):
                        runtest_args = []
                        runtest_pirock_args = []

                        # set step sizes for fixed-step ADR tests
                        fixedh        = 0.01 / np.array([16, 32, 64, 128], dtype=float)
                        fixedh_strang = 0.01 / np.array([16, 32, 64, 128], dtype=float)
                        fixedh_pirock = 0.01 / np.array([16, 32, 64, 128], dtype=float)

                        # set up tests that treat reactions implicitly
                        if (DoImplicitRx):
                            for solver in AdvDiffRxSolvers:
                                for h in fixedh:
                                    runtest_args.append((adrexe, probtype, True, solver[0], solver[1], solver[2],
                                                        solver[3], cux, cuy, cvx, cvy, d, A, B, nx, ny, tf, max
                                                        (1e-3*(h*h),1e-9), atol, h, fixed_maxl, nout, ShowCommand,
                                                        ShowOutput))
                            for solver in ADRStrangSolvers:
                                for h in fixedh_strang:
                                    runtest_args.append((adrexe, probtype, True, solver[0], solver[1], solver[2],
                                                        solver[3], cux, cuy, cvx, cvy, d, A, B, nx, ny, tf, max
                                                        (1e-3*(h*h),1e-9), atol, h, fixed_maxl, nout, ShowCommand,
                                                        ShowOutput))

                        # set up tests that treat reactions explicitly
                        if (DoExplicitRx):
                            for solver in AdvDiffRxSolversExpRx:
                                for h in fixedh:
                                    runtest_args.append((adrexe, probtype, False, solver[0], solver[1],
                                                        solver[2], solver[3], cux, cuy, cvx, cvy, d, A, B, nx,
                                                        ny, tf, max(1e-3*(h*h),1e-9), atol, h,
                                                        fixed_maxl, nout, ShowCommand, ShowOutput))
                            for solver in ADRStrangSolvers:
                                for h in fixedh_strang:
                                    runtest_args.append((adrexe, probtype, False, solver[0], solver[1],
                                                        solver[2], solver[3], cux, cuy, cvx, cvy, d, A, B, nx,
                                                        ny, tf, max(1e-3*(h*h),1e-9), atol, h,
                                                        fixed_maxl, nout, ShowCommand, ShowOutput))

                            for h in fixedh_pirock:
                                runtest_pirock_args.append((adrpirockexe, cux, cuy, cvx, cvy, d, A, B, nx, ny, tf,
                                                            max(1e-3*(h*h),1e-9), atol, h, nout, ShowCommand, ShowOutput))

                        # output argument lists if requested
                        if (ShowArgs):
                            print("ADR Fixed Tests for d = " + str(d) + ":")
                            print("runtest_args:")
                            for argset in runtest_args:
                                print(argset)
                            print("runtest_pirock_args:")
                            for argset in runtest_pirock_args:
                                print(argset)

                        # run tests in parallel
                        with multiprocessing.Pool(processes=maxprocs) as pool:
                            ar = []
                            for args in runtest_args:
                                ar.append(pool.apply_async(runtest, args=args))
                            for args in runtest_pirock_args:
                                ar.append(pool.apply_async(runtest_ADR_pirock, args=args))
                            for a in ar:
                                st = a.get()
                                print(st)
                                FixedStats.append(st)

            # store results
            if (DoADRAdaptiveTests):
                Df = pd.DataFrame.from_records(AdaptStats)
                print("Adaptive step AdvDiffRx2D %s tests finished" % bctype)
                fname = 'data/AdvDiffRx2D-adapt_' + bctype + '.xlsx'
                Df.to_excel(fname, index=False)

            if (DoADRFixedTests):
                Df = pd.DataFrame.from_records(FixedStats)
                print("Fixed step AdvDiffRx2D %s tests finished" % bctype)
                fname = 'data/AdvDiffRx2D-fixed_' + bctype + '.xlsx'
                Df.to_excel(fname, index=False)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
# end of script
