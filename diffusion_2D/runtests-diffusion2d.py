#!/usr/bin/env python
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
import os

# utility routine to run a test, storing the run options and solver statistics
def runtest(solver, pg, rtol, h, kxy, commonargs, showcommand=False):
    stats = {'method': solver['name'], 'procs': pg['np'], 'grid': pg['grid'], 'rtol': rtol,
             'h': h, 'kx': kxy['kx'], 'ky': kxy['ky'],
             'ReturnCode': 1, 'Steps': 1e10, 'Fails': 1e10, 'Accuracy': 1e10,
             'FEvals': 0, 'Runtime': 1e10, 'args': commonargs}
    MPIEXEC = os.getenv("MPIEXEC", "mpiexec")
    runcommand = "%s -n %i %s --nx %i --ny %i --rtol %e --kx %e --ky %e %s" % (MPIEXEC, pg['np'], solver['exe'], pg['grid'], pg['grid'], rtol, kxy['kx'], kxy['ky'], commonargs)
    if (h > 0):
        runcommand += " --fixedstep %e" %(h)
    result = subprocess.run(shlex.split(runcommand), stdout=subprocess.PIPE)
    stats['ReturnCode'] = result.returncode
    if (result.returncode != 0):
        print("Run command " + runcommand + " FAILURE: " + str(result.returncode))
        print(result.stderr)
    else:
        if (showcommand):
            print("Run command " + runcommand + " SUCCESS")
        lines = str(result.stdout).split('\\n')
        for line in lines:
            txt = line.split()
            if ("Steps" in txt):
                stats['Steps'] = int(txt[2])
            elif (("Error" in txt) and ("test" in txt)):
                stats['Fails'] = int(txt[4])
            elif (("Maximum" in txt) and ("relative" in txt)):
                stats['Accuracy'] = float(txt[4])
            elif (("Explicit" in txt) and ("RHS" in txt)):
                stats['FEvals'] += int(txt[5])
            elif (("Implicit" in txt) and ("RHS" in txt)):
                stats['FEvals'] += int(txt[5])
            elif (("LS" in txt) and ("RHS" in txt)):
                stats['FEvals'] += int(txt[5])
            elif (("RHS" in txt) and ("fn" in txt)):
                stats['FEvals'] += int(txt[4])
            elif ("simulation" in txt):
                stats['Runtime'] = float(txt[4])
    return stats

# path to executables
bindir = "./bin/"

# shortcuts to executable/configuration of different solver types
DIRK2Solver = bindir + "diffusion_2D_mpi --integrator dirk --order 2"
DIRK3Solver = bindir + "diffusion_2D_mpi --integrator dirk --order 3"
ERK2Solver = bindir + "diffusion_2D_mpi --integrator erk --order -2"
ERK3Solver = bindir + "diffusion_2D_mpi --integrator erk --order -3"
ERK4Solver = bindir + "diffusion_2D_mpi --integrator erk --order -4"
RKCSolver = bindir + "diffusion_2D_mpi --integrator rkc"
RKLSolver = bindir + "diffusion_2D_mpi --integrator rkl"

# test groups to run
RunAdaptiveTests = True
RunFixedStepTests = False

# common testing parameters
homo = " --inhomogeneous"
atol = " --atol 1.e-11"
controller = " --controller 2"
calcerror = " --error"
precsetup = " --nonlinear --msbp 1"
maxsteps = " --maxsteps 100000"
eigest = " --internaleig"
common = homo + atol + controller + calcerror + precsetup + maxsteps + eigest

# parameter arrays to iterate over
kxky = [{'kx': 0.1,  'ky': 0.0},
        {'kx': 1.0,  'ky': 0.0},
        {'kx': 10.0, 'ky': 0.0}]
procgrids = [{'np': 1,   'grid': 32,  'nrelax': 3},
             {'np': 4,   'grid': 64,  'nrelax': 8},
             {'np': 16,  'grid': 128, 'nrelax': 20},
             {'np': 64,  'grid': 256, 'nrelax': 75}]
rtols = [1.e-2, 1.e-3, 1.e-4, 1.e-5, 1.e-6]
hvals = 1.e-2 / np.array([2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
solvertype = [{'name': 'dirk2-Jacobi', 'exe': DIRK2Solver},
              {'name': 'dirk3-Jacobi', 'exe': DIRK3Solver},
              {'name': 'erk2', 'exe': ERK2Solver},
              {'name': 'erk3', 'exe': ERK3Solver},
              {'name': 'erk4', 'exe': ERK4Solver},
              {'name': 'rkc', 'exe': RKCSolver},
              {'name': 'rkl', 'exe': RKLSolver}]

# run adaptive tests and collect results as a pandas data frame
if (RunAdaptiveTests):
    fname = "results_diffusion_2D.xlsx"
    RunStats = []
    for kxy in kxky:
        for rtol in rtols:
            for pg in procgrids:
                for solver in solvertype:
                    stat = runtest(solver, pg, rtol, 0.0, kxy, common)
                    RunStats.append(stat)
    RunStatsDf = pd.DataFrame.from_records(RunStats)

    # save dataframe as Excel file
    print("RunStatsDf object (adaptive step):")
    print(RunStatsDf)
    print("Saving as Excel")
    RunStatsDf.to_excel(fname, index=False)

# run fixed-step tests and collect results as a pandas data frame
if (RunFixedStepTests):
    fname = "results_diffusion_2D_fixedstep.xlsx"
    RunStats = []
    for kxy in kxky:
        for h in hvals:
            for pg in procgrids:
                for solver in solvertype:
                    stat = runtest(solver, pg, 1.e-9, h, kxy, common)
                    RunStats.append(stat)
    RunStatsDf = pd.DataFrame.from_records(RunStats)

    # save dataframe as Excel file
    print("RunStatsDf object (fixed step):")
    print(RunStatsDf)
    print("Saving as Excel")
    RunStatsDf.to_excel(fname, index=False)
