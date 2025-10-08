#!/usr/bin/env python
#------------------------------------------------------------------------------------------------------------------------------------
# Programmer(s):  Sylvia Amihere @ UMBC
#------------------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2025, University of Maryland, Baltimore County.
# All rights reserved.
# For details, see the LICENSE file.
#------------------------------------------------------------------------------------------------------------------------------------
# This is a python script that runs the 1d Brusselator (advection, reaction, diffusion) problem using the PIROCK package.
# The files involved in running this problem are 'namelist_read.txt' and 'dr_adr_1D.f'.
# The test problem is defined in the file 'pb_adr_1D.f'.
# This script runs the test problem for different advection, diffusion, reaction coefficients, and spatial dimensions.
# Advection and reaction can also be turned on and off.
# The statistics for running the test problem and the results (including number of stages, steps and function evalutions)
# are saved to the excel file 'PIROCK_adr_1D.xlsx'.
# Note: Run the Makefile to generate the executable used in this script.
#-------------------------------------------------------------------------------------------------------------------------------------

# # imports
import pandas as pd
import subprocess
import shlex
import sys, os
import numpy as np

def runtest(nsdV,alfV,uxadvV,vxadvV,wxadvV,brussaV,brussbV,epsV,hV,atolV,rtolV,showcommand=True):
    stats = {'ReturnCode': 0, 'reac': 0, 'advec': 0, 'spatial_dim': 0, 'diff_coef': 0.0,
             'u_advec_coef': 0.0, 'v_advec_coef': 0.0, 'w_advec_coef': 0.0, 'a_rec_coef': 0.0,
             'b_rec_coef': 0.0, 'eps': 0.0, 'CPU_time':0.0, 'time_step': " ", 'intial_h': 0.0,
             'total_steps': 0, 'accpt_steps': 0, 'reject_steps': 0, 'stages':0,'fD_evals': 0,
             'fA_evals': 0, 'fR_evals': 0, 'fRJac_evals': 0}

    advection_OnOff = True #True (1 in PIROCK): advection, False (0 in PIROCK): no advection
    reaction_OnOff  = True #True (1 in PIROCK): reaction,  False (0 in PIROCK): no reaction

    if (advection_OnOff):
        advec_iwork20 = 1
    else:
        advec_iwork20 = 0
    if (reaction_OnOff):
        reac_iwork21 = 1
    else:
        reac_iwork21 = 0

    # Automatically find the directory where this Python script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # modify spatial dimension in fortran file
    executable_filename = 'dr_adr_1D.f'
    executable_file = os.path.join(script_dir, executable_filename)

    nsd_params = []
    with open(executable_file,'r') as exfile:
        lines = exfile.readlines()

    #look for the line with nsd
    for line in lines:
        newline = line
        if 'parameter(nsd=' in line:
            start = line.find('parameter(nsd=') + len('parameter(nsd=')
            end = line.find(',', start)
            newline = line[:start] + str(nsdV) + line[end:]
        nsd_params.append(newline)

    # Write modified content back
    with open(executable_file, 'w') as excfile:
        excfile.writelines(nsd_params)

    ### TO-DO: compile updated source code file

    # modify parameters in namelist file and turn on/off advection/reaction
    namelist_filename = 'namelist_read.txt'
    namelist_file = os.path.join(script_dir, namelist_filename)

    namelist_params = []
    with open(namelist_file,'r') as namefile:
        lines = namefile.readlines()
    for line in lines:
        new_line = line
        if 'iwork20' in line:
            start = line.find('iwork20 = ') + len('iwork20 = ')
            end = line.find('\n', start)
            new_line = line[:start] + str(advec_iwork20) + line[end:]
        if 'iwork21' in line:
            start = line.find('iwork21 = ') + len('iwork21 = ')
            end = line.find('\n', start)
            new_line = line[:start] + str(reac_iwork21) + line[end:]
        if 'alf' in line:
            start = line.find('alf = ') + len('alf = ')
            end = line.find('\n', start)
            new_line = line[:start] + str(alfV) + line[end:]
        if 'uxadv' in line:
            start = line.find('uxadv = ') + len('uxadv = ')
            end = line.find('\n', start)
            new_line = line[:start] + str(uxadvV) + line[end:]
        if 'vxadv' in line:
            start = line.find('vxadv = ') + len('vxadv = ')
            end = line.find('\n', start)
            new_line = line[:start] + str(vxadvV) + line[end:]
        if 'wxadv' in line:
            start = line.find('wxadv = ') + len('wxadv = ')
            end = line.find('\n', start)
            new_line = line[:start] + str(wxadvV) + line[end:]
        if 'brussa' in line:
            start = line.find('brussa = ') + len('brussa = ')
            end = line.find('\n', start)
            new_line = line[:start] + str(brussaV) + line[end:]
        if 'brussb' in line:
            start = line.find('brussb = ') + len('brussb = ')
            end = line.find('\n', start)
            new_line = line[:start] + str(brussbV) + line[end:]
        if 'eps' in line:
            start = line.find('eps = ') + len('eps = ')
            end = line.find('\n', start)
            new_line = line[:start] + str(epsV) + line[end:]
        if 'h' in line:
            start = line.find('h = ') + len('h = ')
            end = line.find('\n', start)
            new_line = line[:start] + str(hV) + line[end:]
        if 'atol' in line:
            start = line.find('atol = ') + len('atol = ')
            end = line.find('\n', start)
            new_line = line[:start] + str(atolV) + line[end:]
        if 'rtol' in line:
            start = line.find('rtol = ') + len('rtol = ')
            end = line.find('\n', start)
            new_line = line[:start] + str(rtolV) + line[end:]
        namelist_params.append(new_line)

    # Write modified content back
    with open(namelist_file, 'w') as namefile:
        namefile.writelines(namelist_params)

    # the folder should contain the executable after running the Makefile
    runcommand = "./adr1D"
    run_result = subprocess.run(shlex.split(runcommand),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if (run_result.returncode != 0):
        print("Run command " + runcommand + " FAILURE: " + str(run_result.returncode))
        print(run_result.stderr)
        sys.exit(1)  # Exit program with error code 1
    else:
        if(showcommand):
            print("Run command: " + runcommand + " SUCCESS\n")
            if (hV<=0.0):
                stats['time_step']= "adaptive"
            else:
                stats['time_step']= "fixed_h"
            # end
            stats['advec']        = advec_iwork20
            stats['reac']         = reac_iwork21
            stats['ReturnCode']   = run_result.returncode
            stats['spatial_dim']  = nsdV
            stats['diff_coef']    = alfV
            stats['u_advec_coef'] = uxadvV
            stats['v_advec_coef'] = vxadvV
            stats['w_advec_coef'] = wxadvV
            stats['a_rec_coef']   = brussaV
            stats['b_rec_coef']   = brussbV
            stats['eps']          = epsV
            stats['atol']         = atolV
            stats['rtol']         = rtolV
            stats['h']            = hV

            ### TO-DO: load the "sol.dat" and "ref.dat" files, compute solution error, and store this in the stats

            # # Show output
            # print("Program output:")
            # print(run_result.stdout.decode())
            lines = run_result.stdout.decode().splitlines()
            for line in lines:
                if 'initial step size h= ' in line:
                    txt = line.split()
                    stats['intial_h'] = float(txt[4])
                elif 'CPU time ' in line:
                    txt = line.split()
                    stats['CPU_time'] = float(txt[2])
                elif 'Max number of stages used= ' in line:
                    txt = line.split()
                    stats['stages'] = txt[5]
                elif ' Number of f evaluations= ' in line:
                    txt = line.split()
                    stats['fD_evals'] = txt[4]
                    stats['fA_evals'] = txt[7]
                    stats['total_steps'] = txt[9]
                    stats['accpt_steps'] = txt[11]
                    stats['reject_steps'] = txt[13]
                elif 'Number of reaction VF' in line:
                    txt = line.split()
                    stats['fR_evals'] = txt[5]
                elif 'Number of reaction Jacobian' in line:
                    txt = line.split()
                    stats['fRJac_evals'] = txt[5]

    return stats
# end of function


# ------------------------------ Run 1d-adv-3var example ------------------------------
nsd_values = [512]  #spatial dimension
alf_values = [1e-1] #diffusion coefficient

# advection values in the x-direction for u, v and w
uxadv_values = [1e-2] #advection coefficient for u in the x-direction
vxadv_values = [1e-2] #advection coefficient for v in the x-direction
wxadv_values = [1e-2] #advection coefficient for w in the x-direction

# advection values in the y-direction for u, v and w, these values will remain unchanged (0) because the example is 1D
uyadv_values = [0.0] #advection coefficient for u in the y-direction
vyadv_values = [0.0] #advection coefficient for v in the y-direction
wyadv_values = [0.0] #advection coefficient for w in the y-direction

# reaction values
brussa = [0.6]  #A
brussb = [2.0]  #B
eps    = [1e-2] #epsilon

# absolute and relative tolerances
atol = [1e-3]
rtol = [1e-3]

# adaptive/fixed step size: adaptive step size: <=0.0, fixed step size: >0.0
fixed_h = [0.0]

# filename to hold run statistics
fname = "PIROCK_adr_1D"

RunStats = []
for nsdVal in nsd_values:
    for alfVal in alf_values:
        for uxadvVal in uxadv_values:
            for vxadvVal in vxadv_values:
                for wxadvVal in wxadv_values:
                    for brussaVal in brussa:
                        for brussbVal in brussb:
                            for epsVal in eps:
                                for hVal in fixed_h:
                                    for atolVal in atol:
                                        for rtolVal in rtol:
                                            stat = runtest(nsdVal,alfVal,uxadvVal,vxadvVal,wxadvVal,brussaVal,brussbVal,epsVal,hVal,atolVal,rtolVal,showcommand=True)
                                            RunStats.append(stat)
# print(RunStats)
RunStatsDf = pd.DataFrame.from_records(RunStats)

# save dataframe as Excel file
print("RunStatsDf object:")
print(RunStatsDf)
print("Saving as Excel")
RunStatsDf.to_excel(fname + '.xlsx', index=False)

# end of script











