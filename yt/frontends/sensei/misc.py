"""
Miscellaneous classes and functions that are Sensei-specific



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2017, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

from vtk import vtkDataObject, mutable
from mpi4py import MPI
comm = MPI.COMM_WORLD

class amrdata():
    '''

    A class that emulates the new metadata in sensei 3.

    '''
    
    def __str__(self):
        return \
        'self.MeshName = %s\n'%(str(self.MeshName)) + \
        'self.NumBlocks = %s\n'%(str(self.NumBlocks)) + \
        'self.NumLevels = %s\n'%(str(self.NumLevels)) + \
        'self.NumGhostCells = %s\n'%(str(self.NumGhostCells)) + \
        'self.NumGhostNodes = %s\n'%(str(self.NumGhostNodes)) + \
        'self.RefRatio = %s\n'%(str(self.RefRatio)) + \
        'self.NumArrays = %s\n'%(str(self.NumArrays)) + \
        'self.ArrayNames = %s\n'%(str(self.ArrayNames)) + \
        'self.ArrayCentering = %s\n'%(str(self.ArrayCentering)) + \
        'self.Extent = %s\n'%(str(self.Extent)) + \
        'self.Bounds = %s\n'%(str(self.Bounds)) + \
        'self.Origin = %s\n'%(str(self.Origin)) + \
        'self.Spacing = %s\n'%(str(self.Spacing)) + \
        'self.BlockOwner = %s\n'%(str(self.BlockOwner)) + \
        'self.BlockIds = %s\n'%(str(self.BlockIds)) + \
        'self.NumBlocksLocal = %s\n'%(str(self.NumBlocksLocal)) + \
        'self.BlockExtents = %s\n'%(str(self.BlockExtents)) + \
        'self.BlockLevels = %s\n'%(str(self.BlockLevels))

    def __init__(self, adaptor, meshName):
        self.MeshName = ''
        self.NumBlocks = 0
        self.NumLevels = 0
        self.NumGhostCells = 0
        self.NumGhostNodes = 0
        self.RefRatio = []
        self.NumArrays = 0
        self.ArrayNames = []
        self.ArrayCentering = []
        self.Extent = [0]*6
        self.Bounds = [0.]*6
        self.Origin = [0.]*3
        self.Spacing = [1.]*3
        self.BlockOwner = []
        self.BlockIds = []
        self.NumBlocksLocal = []
        self.BlockExtents = []
        self.BlockLevels = []

        # until sensei 3 we need to pull the mesh across to access the
        # metadata. because this can be expensive, cache it here
        self.Mesh = None # data

        r = comm.Get_rank()
        self.MeshName = meshName
        # get array metadata
        cens = [vtkDataObject.POINT, vtkDataObject.CELL]
        for cen in cens:
            self.NumArrays = adaptor.GetNumberOfArrays(meshName, cen)
            i = 0
            while i < self.NumArrays:
                self.ArrayNames.append(adaptor.GetArrayName(meshName, cen, i))
                self.ArrayCentering.append(cen)
                i += 1

        # ghosts
        self.NumGhostCells = adaptor.GetMeshHasGhostCells(meshName)
        self.NumGhostNodes = adaptor.GetMeshHasGhostNodes(meshName)

        # get mesh, in sensei 3 data and metadata are separate
        # until then pull the mesh an probe it directly
        mesh = adaptor.GetMesh(meshName, False)

        # get mesh metadata
        mesh.GetBounds(self.Bounds)
        mesh.GetSpacing(0, self.Spacing)

        self.NumLevels = mesh.GetNumberOfLevels()

        i = 0
        while i < self.NumLevels:
            self.RefRatio.append([mesh.GetRefinementRatio(i)]*3)
            i += 1

        self.NumBlocks = mesh.GetTotalNumberOfBlocks()
        self.NumBlocksLocal = [0]

        it = mesh.NewIterator()
        it.SetSkipEmptyNodes(1)
        it.InitTraversal()
        while not it.IsDoneWithTraversal():
            self.NumBlocksLocal[0] += 1
            it.GoToNextItem()

        it.SetSkipEmptyNodes(0)
        it.InitTraversal()
        while not it.IsDoneWithTraversal():
            cid = it.GetCurrentFlatIndex()
            lev = mutable(0)
            idx = mutable(0)
            mesh.GetLevelAndIndex(cid, lev, idx)
            vbox = mesh.GetAMRBox(lev, idx)
            lc = [-123]*3
            hc = [-123]*3
            lo = vbox.GetDimensions(lc, hc)
            self.BlockExtents.append(lc + hc)
            self.BlockLevels.append(lev.get())
            self.BlockIds.append(idx.get())
            patch = it.GetCurrentDataObject()
            self.BlockOwner.append(r if patch else -1)
            it.GoToNextItem()

        self.Mesh = mesh
        return
