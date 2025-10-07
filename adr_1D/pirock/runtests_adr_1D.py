#!/usr/bin/env python
#------------------------------------------------------------------------------------------------------------------------------------
# Programmer(s):  Sylvia Amihere @ UMBC
#------------------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2025, University of Maryland, Baltimore County.
# All rights reserved.
# For details, see the LICENSE file.
#------------------------------------------------------------------------------------------------------------------------------------
# ReadME: This is a python script that runs the 1d Brusselator (advection, reaction, diffusion) problem in the PIROCK package.
#         The files involved in running this problem are 'namelist_read.txt' and 'dr_adr_1D.f'. The test problem is 
#         defined in the file 'pb_adr_1D.f'
#         In this script, the parameters used to run the test problem can be edited as needed and new values added. 
# 
#-------------------------------------------------------------------------------------------------------------------------------------

# # imports
import pandas as pd
import subprocess
import shlex
import sys, os,re
import numpy as np
import matplotlib.pyplot as plt
import itertools
from itertools import cycle
from matplotlib.gridspec import GridSpec


def runtest(nsdV,alfV,uxadvV,vxadvV,wxadvV,brussaV,brussbV,epsV,atolV,rtolV,showcommand=True):
    stats = {'ReturnCode': 0, 'reaction': 0, 'advection': 0, 'spatial_dim': 0, 'diff_coef': 0.0, 
             'u_advec_coef': 0.0, 'v_advec_coef': 0.0, 'w_advec_coef': 0.0, 'a_rec_coef': 0.0,
             'b_rec_coef': 0.0, 'eps': 0.0, 'CPU_time':0.0,
             'intial_h': 0.0, 'total_steps': 0, 'accpt_steps': 0, 'reject_steps': 0, 'stages':0,
             'fD_evals': 0, 'fA_evals': 0,'fR_evals': 0, 'fRJac_evals': 0}
    
    advection_OnOff = True #iwork20 = 0: no advection, iwork20 = 1: advection
    reaction_OnOff  = True #iwork21 = 0: no reaction,  iwork21 = 1: reaction

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

    # for nsdVal in nsd_values:
    nsd_params = []
    with open(executable_file,'r') as exfile:
        lines = exfile.readlines()
    
    #look for the line with nsd 
    for line in lines:
        newline = line
        if 'parameter(nsd=' in line:
            # print(line)
            start = line.find('parameter(nsd=') + len('parameter(nsd=')
            end = line.find(',', start)
            newline = line[:start] + str(nsdV) + line[end:]
            # print(new_line)
        nsd_params.append(newline)

    # Write modified content back
    with open(executable_file, 'w') as excfile:
        excfile.writelines(nsd_params)

    # modify parameters in namelist file and turn on/off advection/reaction
    namelist_filename = 'namelist_read.txt'
    namelist_file = os.path.join(script_dir, namelist_filename)

    namelist_params = []
    with open(namelist_file,'r') as namefile:
        lines = namefile.readlines()
    for line in lines:
        new_line = line
        if 'iwork20' in line:
            # print(line)
            start = line.find('iwork20 = ') + len('iwork20 = ')
            end = line.find('\n', start)
            new_line = line[:start] + str(advec_iwork20) + line[end:]
            # print(new_line)
        if 'iwork21' in line:
            # print(line)
            start = line.find('iwork21 = ') + len('iwork21 = ')
            end = line.find('\n', start)
            new_line = line[:start] + str(reac_iwork21) + line[end:]
            # print(new_line)
        # for alfVal in alf_values:
        if 'alf' in line:
            start = line.find('alf = ') + len('alf = ')
            end = line.find('\n', start)
            new_line = line[:start] + str(alfV) + line[end:]
            # print(new_line)
        # for uxadvVal in uxadv_values:
        if 'uxadv' in line:
            start = line.find('uxadv = ') + len('uxadv = ')
            end = line.find('\n', start)
            new_line = line[:start] + str(uxadvV) + line[end:]
            # print(new_line)
        # for vxadvVal in vxadv_values:
        if 'vxadv' in line:
            start = line.find('vxadv = ') + len('vxadv = ')
            end = line.find('\n', start)
            new_line = line[:start] + str(vxadvV) + line[end:]
            # print(new_line)
        # for wxadvVal in wxadv_values:
        if 'wxadv' in line:
            start = line.find('wxadv = ') + len('wxadv = ')
            end = line.find('\n', start)
            new_line = line[:start] + str(wxadvV) + line[end:]
            # print(new_line)
        # for brussaVal in brussa:
        if 'brussa' in line:
            start = line.find('brussa = ') + len('brussa = ')
            end = line.find('\n', start)
            new_line = line[:start] + str(brussaV) + line[end:]
            # print(new_line)
        # for brussbVal in brussb:
        if 'brussb' in line:
            start = line.find('brussb = ') + len('brussb = ')
            end = line.find('\n', start)
            new_line = line[:start] + str(brussbV) + line[end:]
            # print(new_line)
        # for epsVal in eps:
        if 'eps' in line:
            start = line.find('eps = ') + len('eps = ')
            end = line.find('\n', start)
            new_line = line[:start] + str(epsV) + line[end:]
            # print(new_line)
        # for atolVal in atol:
        if 'atol' in line:
            start = line.find('atol = ') + len('atol = ')
            end = line.find('\n', start)
            new_line = line[:start] + str(atolV) + line[end:]
            # print(new_line)
        # for rtolVal in rtol:
        if 'rtol' in line:
            start = line.find('rtol = ') + len('rtol = ')
            end = line.find('\n', start)
            new_line = line[:start] + str(rtolV) + line[end:]
            # print(new_line)
        namelist_params.append(new_line)

    # Write modified content back
    with open(namelist_file, 'w') as namefile:
        namefile.writelines(namelist_params)

    # Compile the Fortran code
    compile_cmd = "gfortran -std=legacy -o pirock_adr_1D dr_adr_1D.f"
    compile_result = subprocess.run(shlex.split(compile_cmd),stdout=subprocess.PIPE,stderr=subprocess.PIPE)

    # Check if compilation failed
    if compile_result.returncode != 0:
        print("Compilation failed:" + str(compile_result.returncode))
        print(compile_result.stderr)
        sys.exit(1)  # Exit program with error code 1
    # Run the compiled executable
    else:
        if(showcommand):
            runcommand = "./pirock_adr_1D"
            run_result = subprocess.run(shlex.split(runcommand),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            print("Run command: " + runcommand + " SUCCESS\n")
            stats['advection']    = advec_iwork20
            stats['reaction']     = reac_iwork21
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

            # # Show output
            # print("Program output:")
            # print(run_result.stdout.decode())
            lines = run_result.stdout.decode().splitlines()#str(run_result.stdout).splitlines()
            for line in lines:
                # print(line)
                if 'initial step size h= ' in line:
                    txt = line.split()
                    # print(txt)
                    # print(txt[4])
                    stats['intial_h'] = float(txt[4])
                elif 'CPU time ' in line:
                    txt = line.split()
                    # print(txt)
                    # print(txt[2])
                    stats['CPU_time'] = float(txt[2])
                elif 'Max number of stages used= ' in line:
                    txt = line.split()
                    # print(txt)
                    # print(txt[5])
                    stats['stages'] = txt[5]
                elif ' Number of f evaluations= ' in line:
                    txt = line.split()
                    # print(txt)
                    # print(txt[4])
                    stats['fD_evals'] = txt[4]
                    # print(txt[7])
                    stats['fA_evals'] = txt[7]
                    # print(txt[9])
                    stats['total_steps'] =  txt[9]
                    # print(txt[11])
                    stats['accpt_steps'] = txt[11]
                    # print(txt[13])
                    stats['reject_steps'] = txt[13]
                elif 'Number of reaction VF' in line:
                    txt = line.split()
                    # print(txt)
                    # print(txt[5])
                    stats['fR_evals'] = txt[5]
                elif 'Number of reaction Jacobian' in line:
                    txt = line.split()
                    # print(txt)
                    # print(txt[5])
                    stats['fRJac_evals'] = txt[5]
    
    return stats
# end of function


# ------------------------------ Run 1d-adv-3var example ------------------------------
nsd_values = [512]   #spatial dimension
alf_values = [1e-1] #diffusion coefficient

# advection values in the x-direction for u, v and w
uxadv_values = [1e-2] #advection coefficient for u in the x-direction 
vxadv_values = [1e-2] #advection coefficient for v in the x-direction 
wxadv_values = [1e-2] #advection coefficient for w in the x-direction 

# advection values in the x-direction for u, v and w, these values will remain 0 because the example is 1D
uyadv_values = [0.0] #advection coefficient for u in the y-direction 
vyadv_values = [0.0] #advection coefficient for v in the y-direction 
wyadv_values = [0.0] #advection coefficient for w in the y-direction 

# reaction values
brussa = [0.6] #A
brussb = [2.0]  #B 
eps    = [1e-2] #epsilon

# absolute and relative tolerances
atol     = [1e-3]
rtol     = [1e-3]

# filename to hold run statistics
fname = "PIROCK"

RunStats = []
for nsdVal in nsd_values:
    for alfVal in alf_values:
        for uxadvVal in uxadv_values:
            for vxadvVal in vxadv_values:
                for wxadvVal in wxadv_values:
                    for brussaVal in brussa:
                        for brussbVal in brussb:
                            for epsVal in eps:
                                for atolVal in atol:
                                    for rtolVal in rtol:
                                        stat = runtest(nsdVal,alfVal,uxadvVal,vxadvVal,wxadvVal,brussaVal,brussbVal,epsVal,atolVal,rtolVal,showcommand=True)
                                        RunStats.append(stat)
# print(RunStats)
RunStatsDf = pd.DataFrame.from_records(RunStats)

# save dataframe as Excel file
print("RunStatsDf object:")
print(RunStatsDf)
print("Saving as Excel")
RunStatsDf.to_excel(fname + '.xlsx', index=False)

# end of script











