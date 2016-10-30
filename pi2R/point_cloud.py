#!/usr/bin/python
import vtk
from vtk_visualizer.pointobject import *


class PointCloud:

    def __init__(self):
    # Renderer
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(.2, .3, .4)
        self.renderer.ResetCamera()

        self.renderWindow = vtk.vtkRenderWindow()
        self.iren = vtk.vtkRenderWindowInteractor()
    # Render Window
        self.renderWindow.AddRenderer(self.renderer)
    # Interactor
        self.iren.SetRenderWindow(self.renderWindow)
    # Interactor style
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.iren.SetInteractorStyle(style)

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
            if colors.dtype == np.float32:
                colors*=256
            obj.AddColors(colors.astype(np.uint8))
            self.pointObjects.append(obj)
            self.renderer.AddActor(obj.GetActor())

    def addSlider(self, callback):
        SliderWidget = vtk.vtkSliderWidget()
        SliderWidget.SetInteractor(self.iren)
#        SliderWidget.SetRepresentation(SliderRepres)
        SliderWidget.KeyPressActivationOff()
        SliderWidget.SetAnimationModeToAnimate()
        SliderWidget.SetEnabled(True)
        SliderWidget.AddObserver("EndInteractionEvent", callback)
        self.SliderWidget = SliderWidget

    def getSliderValue(self, obj):
        sliderRepres = obj.GetRepresentation()
        return sliderRepres.GetValue()

    def removeActors(self):
        for a in self.pointObjects:
            self.renderer.RemoveActor(a.GetActor())
        self.c = []
        self.p = []
        self.pointObjects = []

    def addActors(self):
        self.addPointsAktor(np.concatenate(self.p),np.concatenate(self.c))

    def run(self):

        self.addActors()

        transform = vtk.vtkTransform()
        transform.Translate(1.0, 0.0, 0.0)

        axes = vtk.vtkAxesActor()
        #  The axes are positioned with a user transform
#        axes.SetUserTransform(transform)
        self.renderer.AddActor(axes)

    #    renderer.AddActor(pointCloud.vtkActor)
    # Begin Interaction
        self.renderWindow.Render()
        self.iren.Start()
