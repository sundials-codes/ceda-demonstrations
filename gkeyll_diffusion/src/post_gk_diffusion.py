import postgkyl as pg
import numpy as np
import matplotlib.pyplot as plt

#...........................................................................#
#.
#.Post process 1x1v diffusion test.
#.
#.Started by: Manaure Francisquez.
#.Started on: October 2024.
#.
#...........................................................................#

dataDir = '../'  #.Directory where data is located.
outDir  = '../'  #.Directory to place output files.

simName = 'gk_diffusion_1x1v_p1'  #.Simulation name.
fieldName = 'f'  #.Quantity to plot.

frame = 10  #.Frame number to plot.

outFigureFile    = False    #.Output a figure file?.
figureFileFormat = '.png'    #.Can be .png, .pdf, .ps, .eps, .svg.

#..................... NO MORE USER INPUTS BELOW (maybe) ....................#

basisType = 'gkhyb'  #.'ms': modal serendipity, or 'ns': nodal serendipity.
polyOrder = 1  #.Polynomial order.

#.Full path and name of data file.
dataFile = dataDir + simName + '-' + fieldName + '_' + str(frame) + '.gkyl'

pgData = pg.GData(dataFile) #.Get gkeyll data.
pgInterp = pg.GInterpModal(pgData, polyOrder, basisType) #.Get an interpolating object for Gkeyll data.
xInt, fldInt = pgInterp.interpolate() #.Get interpolated grid and field.
fldInt = np.squeeze(fldInt)

#.2D nodal mesh.
Xnodal = [np.outer(xInt[0], np.ones(np.shape(xInt[1]))), \
          np.outer(np.ones(np.shape(xInt[0])), xInt[1])]

#.Make figure and plot.
figProp1a = (7., 5)
ax1aPos   = [ [0.08, 0.09, 0.78, 0.85], ]
cax1aPos  = [0.865, 0.09, 0.02, 0.85]
fig1a     = plt.figure(figsize=figProp1a)
ax1a      = list()
for i in range(len(ax1aPos)):
  ax1a.append(fig1a.add_axes(ax1aPos[i]))
cbar_ax1a = fig1a.add_axes(cax1aPos)

hpl1a = list()
hpl1a.append(ax1a[0].pcolormesh(Xnodal[0], Xnodal[1], fldInt, cmap='inferno'))

ax1a[0].set_xlabel(r'$x$', fontsize=16, labelpad=-2)
ax1a[0].set_ylabel(r'$v_\parallel$', fontsize=16, labelpad=-1)
cbar = plt.colorbar(hpl1a[0], ax=ax1a, cax=cbar_ax1a)
cbar.set_label(r'$f(x,v_\parallel)$', rotation=270, labelpad=18, fontsize=16)

if outFigureFile:
  outFile = outDir + simeName + '-' + fieldName + '_' + str(frame) + figureFileFormat
  plt.savefig(outFile)
  plt.close()
else:
  plt.show()
