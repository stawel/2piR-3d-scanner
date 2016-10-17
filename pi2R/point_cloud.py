#!/usr/bin/python
import vtk
from vtk_visualizer.pointobject import *


class PointCloud:

    def __init__(self):
        self.renderer = vtk.vtkRenderer()
        self.pointObjects = []
        self.p = []
        self.c = []
    def addPoints(self, points, colors):
        if len(points) > 0:
            self.p.append(points)
            self.c.append(colors)

    def addPointsAktor(self, points, colors):
        if len(points) > 0:
            obj = VTKObject()
            obj.CreateFromArray(points)
            obj.AddColors(colors.astype(np.uint8))
            self.pointObjects.append(obj)
            self.renderer.AddActor(obj.GetActor())

def run(pointCloud):
# Renderer
    renderer = pointCloud.renderer
#    renderer.AddActor(pointCloud.vtkActor)
    renderer.SetBackground(.2, .3, .4)
    renderer.ResetCamera()

    pointCloud.addPointsAktor(np.concatenate(pointCloud.p),np.concatenate(pointCloud.c))
# Render Window
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)

# Interactor
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    style = vtk.vtkInteractorStyleTrackballCamera()
    renderWindowInteractor.SetInteractorStyle(style)


# Begin Interaction
    renderWindow.Render()
    renderWindowInteractor.Start()
