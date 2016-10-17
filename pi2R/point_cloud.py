#!/usr/bin/python
import vtk
from numpy import random

class PointCloud:

    def __init__(self, zMin=-100.0, zMax=100.0, maxNumPoints=1e6):
        self.maxNumPoints = maxNumPoints
        self.vtkPolyData = vtk.vtkPolyData()
        self.clearPoints()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.vtkPolyData)
#        mapper.SetColorModeToDefault()
#        mapper.SetScalarRange(zMin, zMax)
#        mapper.SetScalarVisibility(1)
        self.vtkActor = vtk.vtkActor()
        self.vtkActor.SetMapper(mapper)

    def addPoints(self, points, colors = []):
        for i in range(0,len(points)):
            if(len(colors) <= i):
                color = [255,255,0]
            else:
                color = colors[i]
            self.addPoint(points[i], color)


    def addPoint(self, point, color = [255, 0, 0]):
        pointId = self.vtkPoints.InsertNextPoint(point[:])
        self.Colors.InsertNextTuple3(color[2], color[1], color[0])
#        self.vtkDepth.InsertNextValue(-point[2])
        self.vtkCells.InsertNextCell(1)
        self.vtkCells.InsertCellPoint(pointId)
        self.vtkCells.Modified()
        self.vtkPoints.Modified()
#        self.vtkDepth.Modified()

    def clearPoints(self):
        self.vtkPoints = vtk.vtkPoints()
        self.vtkCells = vtk.vtkCellArray()
        self.Colors = vtk.vtkUnsignedCharArray()
        self.Colors.SetNumberOfComponents(3)
        self.Colors.SetName('Colors')
        self.vtkPolyData.SetPoints(self.vtkPoints)
        self.vtkPolyData.SetVerts(self.vtkCells)
        self.vtkPolyData.GetPointData().SetScalars(self.Colors)
#        self.vtkPolyData.GetPointData().SetActiveScalars('DepthArray')


def run(pointCloud):
# Renderer
    renderer = vtk.vtkRenderer()
    renderer.AddActor(pointCloud.vtkActor)
    renderer.SetBackground(.2, .3, .4)
    renderer.ResetCamera()

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
