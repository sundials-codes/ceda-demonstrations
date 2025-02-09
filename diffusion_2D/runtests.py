#!/usr/bin/env python
#------------------------------------------------------------
# Programmer(s):  Daniel R. Reynolds @ SMU
#------------------------------------------------------------
# Copyright (c) 2025, Southern Methodist University.
# All rights reserved.
# For details, see the LICENSE file.
#------------------------------------------------------------

# imports
import pandas as pd
import subprocess
import shlex
import sys

# utility routine to run a test, storing the run options and solver statistics
def runtest(solver, pg, rtol, kxy, commonargs, showcommand=False):
    stats = {'method': solver['name'], 'procs': pg['np'], 'grid': pg['grid'], 'rtol': rtol,
             'kx': kxy['kx'], 'ky': kxy['ky'],
             'ReturnCode': 0, 'Steps': 0, 'Fails': 0, 'Accuracy': 0.0,
             'FEvals': 0, 'Runtime': 0.0, 'args': commonargs}
    runcommand = "mpiexec -n %i %s --nx %i --ny %i --rtol %e --kx %e --ky %e %s" % (pg['np'], solver['exe'], pg['grid'], pg['grid'], rtol, kxy['kx'], kxy['ky'], commonargs)
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
            elif ("Relative" in txt):
                stats['Accuracy'] = float(txt[3])
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

# filename to hold run statistics
fname = "results_diffusion_2D"

# shortcuts to executable/configuration of different solver types
DIRKSolver = "./diffusion_2D_mpi --integrator dirk --order 2"
DIRKSolverHypre = "./diffusion_2D_mpi_hypre --integrator dirk --order 2"
ERKSolver = "./diffusion_2D_mpi --integrator erk --order 2"
RKCSolver = "./diffusion_2D_mpi --integrator rkc"
RKLSolver = "./diffusion_2D_mpi --integrator rkl"

# common testing parameters
homo = " --inhomogeneous"
atol = " --atol 1.e-11"
controller = " --controller 2"
common = homo + atol + controller

# parameter arrays to iterate over
kxky = [{'kx': 0.01, 'ky': 0.0},
        {'kx': 0.1,  'ky': 0.0},
        {'kx': 1.0,  'ky': 0.0}]
# procgrids = [{'np': 1,   'grid': 64},
#              {'np': 4,   'grid': 128},
#              {'np': 16,  'grid': 256},
#              {'np': 64,  'grid': 512},
#              {'np': 256, 'grid': 1024}]
procgrids = [{'np': 1,   'grid': 64},
             {'np': 4,   'grid': 128}]
rtols = [1.e-2, 1.e-4, 1.e-6]
solvertype = [{'name': 'dirk-Jacobi', 'exe': DIRKSolver},
              {'name': 'dirk-hypre', 'exe': DIRKSolverHypre},
              {'name': 'erk', 'exe': ERKSolver},
              {'name': 'rkc', 'exe': RKCSolver},
              {'name': 'rkl', 'exe': RKLSolver}]

# run tests and collect results as a pandas data frame
RunStats = []
for solver in solvertype:
    for pg in procgrids:
        for kxy in kxky:
            for rtol in rtols:
                stat = runtest(solver, pg, rtol, kxy, common)
                RunStats.append(stat)
RunStatsDf = pd.DataFrame.from_records(RunStats)

# save dataframe as Excel file
print("RunStatsDf object:")
print(RunStatsDf)
print("Saving as Excel")
RunStatsDf.to_excel(fname + '.xlsx', index=False)
