#!/usr/bin/env python
#------------------------------------------------------------
# Programmer(s):  Daniel R. Reynolds @ UMBC
#------------------------------------------------------------
# Copyright (c) 2025, University of Maryland Baltimore County
# All rights reserved.
# For details, see the LICENSE file.
#------------------------------------------------------------

# imports
import pandas as pd
import subprocess
import shlex
import numpy as np

# utility routine to run a test, storing the run options and solver statistics
def runtest(exe, k, rtol, atol, h, normtype, method, eigfreq, eigsafety, user_dom_eig,
            dee_id, dee_init_wups, dee_succ_wups, dee_maxiters, dee_reltol, commonargs,
            showcommand=False):
    stats = {'k': k, 'rtol': rtol, 'atol': atol, 'h': h, 'normtype': normtype,
             'method': method['name'], 'sspstages': method['stages'], 'eigfreq': eigfreq,
             'eigsafety': eigsafety, 'user_dom_eig': user_dom_eig, 'dee_id': dee_id,
             'dee_init_wups': dee_init_wups, 'dee_succ_wups': dee_succ_wups,
             'dee_maxiters': dee_maxiters, 'dee_reltol': dee_reltol,
             'ReturnCode': 0, 'Steps': 0, 'Fails': 0, 'Accuracy': 0.0, 'Accuracy_Tf': 0.0,
             'RHSEvals': 0, 'RHSEvalsDEE': 0, 'MaxStages': 0, 'MaxSpectralRadius': 0,
             'Runtime': 0.0, 'commonargs': commonargs}
    if (method['name'] == 'RKC'):
        methodnum = 0
    elif (method['name'] == 'RKL'):
        methodnum = 1
    elif (method['name'] == 'SSP2'):
        methodnum = 2
    elif (method['name'] == 'SSP3'):
        methodnum = 3
    elif (method['name'] == 'SSP4'):
        methodnum = 4
    runcommand = "%s --k %e --rtol %e --atol %e --fixedstep %e --wrms_norm_type %i --method %i --eigfrequency %i --eigsafety %e --dee_id %i --dee_num_init_wups %i --dee_num_succ_wups %i --dee_max_iters %i --dee_reltol %e %s" % (exe, k, rtol, atol, h, normtype, methodnum, eigfreq, eigsafety, dee_id, dee_init_wups, dee_succ_wups, dee_maxiters, dee_reltol, commonargs)
    if (user_dom_eig):
        runcommand += " --user_dom_eig"
    if (methodnum > 1):
        runcommand += " --num_SSP_stages %i" % (method['stages'])
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
            elif (("max-in-space" in txt) and ("at" in txt)):
                stats['Accuracy_Tf'] = float(txt[3])
            elif (("max-in-time" in txt) and ("max-in-space" in txt)):
                stats['Accuracy'] = float(txt[5])
            elif (("RHS" in txt) and ("fn" in txt)):
                stats['RHSEvals'] += int(txt[4])
            elif (("fe" in txt) and ("DEE" in txt)):
                stats['RHSEvalsDEE'] += int(txt[7])
            elif (("stages" in txt) and ("used" in txt) and ("Max." in txt)):
                stats['MaxStages'] += int(txt[6])
            elif (("stages" in txt) and ("used" in txt) and ("Number" in txt)):
                stats['MaxStages'] += int(txt[5])
            elif (("Max." in txt) and ("spectral" in txt)):
                stats['MaxSpectralRadius'] += float(txt[4])
            elif (("Computed" in txt) and ("CPU" in txt)):
                stats['Runtime'] = float(txt[4])
    return stats

# filename to hold run statistics
fname_fixed = "results_gk_diffusion_1x1v_p1_fixed.xlsx"
fname_adaptive = "results_gk_diffusion_1x1v_p1_adaptive.xlsx"

# executable
exe = "./gk_diffusion_1x1v_p1"

# common testing parameters
tf = " --tf 1.0"
maxstages = " --stage_max_limit 1000"
nout = " --nout 10"
maxsteps = " --maxsteps 100000"
common = tf + maxstages + nout + maxsteps

# parameter arrays to iterate over
kvals = [0.1, 1.0, 10.0]
rtols = [1.e-2, 1.e-3, 1.e-4, 1.e-5, 1.e-6, 1.e-7, 1.e-8]
atol = 1e-12
hvals = 0.1/np.array([10, 20, 40, 80, 160])
normtypes = [1, 2]
STSSolvers = [{'name': 'RKC', 'stages': 0},
              {'name': 'RKL', 'stages': 0}]
SSPSolvers = [{'name': 'SSP2', 'stages': 2},
              {'name': 'SSP3', 'stages': 4},
              {'name': 'SSP4', 'stages': 10}]
eigfreq = 0
eigsafety = [1.01, 1.05, 1.1, 1.2]
user_dom_eig = [False, True]
dee_id = 0
dee_init_wups = 0
dee_succ_wups = 0
dee_maxiters_val = [100]
dee_reltol = 1e-1

# flags to enable/disable classes of tests
DoFixedTests = True
DoAdaptiveTests = True

# run fixed-step tests, collect results as a pandas data frame, and save to Excel
if (DoFixedTests):
    FixedStats = []
    for k in kvals:
        for h in hvals:
            # STS methods
            for method in STSSolvers:
                for eigsafety_val in eigsafety:
                    for dee_maxiters in dee_maxiters_val:
                        # automatic eigenvalue computation
                        stat = runtest(exe, k, 1e-4, atol, h, 1, method,
                                    eigfreq, eigsafety_val, False,
                                    dee_id, dee_init_wups, dee_succ_wups,
                                    dee_maxiters, dee_reltol, common)
                        FixedStats.append(stat)
                        # user supplied eigenvalue
                        stat = runtest(exe, k, 1e-4, atol, h, 1, method,
                                    eigfreq, eigsafety_val, True,
                                    dee_id, dee_init_wups, dee_succ_wups,
                                    dee_maxiters, dee_reltol, common)
                        FixedStats.append(stat)
            # SSP methods
            for method in SSPSolvers:
                stat = runtest(exe, k, 1e-4, atol, h, 1, method,
                               eigfreq, 1.15, False,
                               dee_id, dee_init_wups, dee_succ_wups,
                               100, dee_reltol, common)
                FixedStats.append(stat)
    FixedStatsDf = pd.DataFrame.from_records(FixedStats)

    # save dataframe as Excel file
    print("FixedStatsDf object:")
    print(FixedStatsDf)
    print("Saving as Excel")
    FixedStatsDf.to_excel(fname_fixed, index=False)

# run adaptive tests, collect results as a pandas data frame, and save to Excel
if (DoAdaptiveTests):
    AdaptiveStats = []
    for k in kvals:
        for rtol in rtols:
            for normtype in normtypes:
                # STS methods
                for method in STSSolvers:
                    for eigsafety_val in eigsafety:
                        for dee_maxiters in dee_maxiters_val:
                            # automatic eigenvalue computation
                            stat = runtest(exe, k, rtol, atol, 0.0, normtype,
                                        method, eigfreq, eigsafety_val, False,
                                        dee_id, dee_init_wups, dee_succ_wups,
                                        dee_maxiters, dee_reltol, common)
                            AdaptiveStats.append(stat)
                            # user supplied eigenvalue
                            stat = runtest(exe, k, rtol, atol, 0.0, normtype,
                                        method, eigfreq, eigsafety_val, True,
                                        dee_id, dee_init_wups, dee_succ_wups,
                                        dee_maxiters, dee_reltol, common)
                            AdaptiveStats.append(stat)
                # SSP methods
                for method in SSPSolvers:
                    stat = runtest(exe, k, rtol, atol, 0.0, normtype,
                                   method, eigfreq, 1.15, False,
                                   dee_id, dee_init_wups, dee_succ_wups,
                                   100, dee_reltol, common)
                    AdaptiveStats.append(stat)
    AdaptiveStatsDf = pd.DataFrame.from_records(AdaptiveStats)

    # save dataframe as Excel file
    print("AdaptiveStatsDf object:")
    print(AdaptiveStatsDf)
    print("Saving as Excel")
    AdaptiveStatsDf.to_excel(fname_adaptive, index=False)
