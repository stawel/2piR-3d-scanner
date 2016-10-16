#!/usr/bin/python
from vtk_visualizer import *
import numpy as np
 
# Generate 1000 random points
xyz = np.random.rand(1000,3)
 
# Plot them
plotxyz(xyz)
