"""
Data structures for Sensei



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2017, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

from __future__ import absolute_import

from yt.utilities.on_demand_imports import _h5py as h5py
import io
import weakref
import numpy as np
import os
import stat
import string
import time

from collections import defaultdict
from yt.extern.six.moves import zip as izip

from yt.funcs import \
    ensure_list, \
    ensure_tuple, \
    get_pbar, \
    setdefaultattr
from yt.data_objects.grid_patch import \
    AMRGridPatch
from yt.geometry.grid_geometry_handler import \
    GridIndex
from yt.geometry.geometry_handler import \
    YTDataChunk
from yt.data_objects.static_output import \
    Dataset
from yt.fields.field_info_container import \
    NullFunc
from yt.utilities.logger import ytLogger as mylog
from yt.utilities.lib.misc_utilities import \
    get_box_grids_level

from .fields import \
    SenseiFieldInfo

from.misc import SenseiMetaData
from vtk import mutable

# from yt.utilities.parallel_tools.parallel_analysis_interface \
#     import communication_system
# comm = communication_system.communicators[-1]


class SenseiGridInSitu(AMRGridPatch):
    """

    Class representing a single Sensei Grid instance.

    """

    __slots__ = ['proc_num']
    _id_offset = 0

    def __init__(self, global_id, local_id, index):
        """

        Returns an instance of SenseiGrid with *id*, associated with
        *filename* and *index*.

        """
        
        AMRGridPatch.__init__(self, global_id, filename = None, index = index)
        self._children_ids = []
        self._parent_id = []
        self.Level = -1
        self._idx = local_id

    def __repr__(self):
        return "SenseiGrid_%04i" % (self.id)

    @property
    def Parent(self):
        if len(self._parent_id) == 0:
            return None
        return [self.index.grids[pid - self._id_offset]
                for pid in self._parent_id]

    @property
    def Children(self):
        return [self.index.grids[cid - self._id_offset]
                for cid in self._children_ids]


class SenseiHierarchyInSitu(GridIndex):

    grid = SenseiGridInSitu

    def __init__(self, ds, dataset_type = None):
        self.dataset_type = dataset_type
        self.float_type = 'float64'
        self.dataset = weakref.proxy(ds)
        self.directory = os.getcwd()
        GridIndex.__init__(self, ds, dataset_type)
    
    def _initialize_data_storage(self):
        pass

    def _count_grids(self):
        self.num_grids = self.dataset.amrdata.NumBlocks

    def _parse_index(self):
        self.max_level = self.dataset.amrdata.NumLevels -1 
        self.num_ghost = self.dataset.amrdata.NumGhostCells

        self.grids = []
        si, ei, LE, RE = [], [], [], []

        origin = np.array(self.dataset.amrdata.Origin, dtype=np.float64)
        dx0    = np.array(self.dataset.amrdata.Spacing, dtype=np.float64)
        ref_ratio = np.array(self.dataset.amrdata.RefRatio, dtype=np.int32)
        
        mesh = self.dataset.mesh
        it = mesh.NewIterator()
        gi = 0
        it.SetSkipEmptyNodes(0)
        it.InitTraversal()
        while not it.IsDoneWithTraversal():
            # get the local data
            patch = it.GetCurrentDataObject()
            
            # get level and level id
            cid = it.GetCurrentFlatIndex()
            lev = mutable(0)
            idx = mutable(0)
            mesh.GetLevelAndIndex(cid, lev, idx)

            self.grid_levels[gi, :] = lev
            
            extent = self.dataset.amrdata.BlockExtents[gi]
            lo = np.array(extent[:3], dtype=np.int32)
            hi = np.array(extent[3:], dtype=np.int32)

            si.append(lo)
            ei.append(hi)
            
            dx_lev = dx0/np.product(ref_ratio[:lev], axis=0)[0]
            xhi = origin + dx_lev*(hi+1)
            xlo = origin + dx_lev*lo
            
            LE.append(xlo)
            RE.append(xhi)
            
            go = self.grid(gi, idx, self)
            go.Level = lev.get()
            go.start_index = lo
            gi += 1
            self.grids.append(go)
            
            it.GoToNextItem()

        self._fill_arrays(ei, si, LE, RE)
        self.grid_procs = np.array(self.dataset.amrdata.BlockOwner, dtype=np.int32)
        
    def _populate_grid_objects(self):
        mylog.debug("Creating grid objects")
        self.grids = np.array(self.grids, dtype='object')
        self._reconstruct_parent_child()
        for i, grid in enumerate(self.grids):
            grid._prepare_grid()
            grid._setup_dx()
            grid.proc_num = self.grid_procs[i]

    def _reconstruct_parent_child(self):
        if (self.max_level == 0):
            return
        mask = np.empty(len(self.grids), dtype='int32')
        mylog.debug("First pass; identifying child grids")
        for i, grid in enumerate(self.grids):
            get_box_grids_level(self.grid_left_edge[i,:],
                                self.grid_right_edge[i,:],
                                self.grid_levels[i] + 1,
                                self.grid_left_edge, self.grid_right_edge,
                                self.grid_levels, mask)
            ids = np.where(mask.astype("bool"))  # where is a tuple
            grid._children_ids = ids[0] + grid._id_offset
        mylog.debug("Second pass; identifying parents")
        for i, grid in enumerate(self.grids):  # Second pass
            for child in grid.Children:
                child._parent_id.append(i + grid._id_offset)

    def save_data(self, *args, **kwargs):
        pass
        
    def _initialize_grid_arrays(self):
        super(SenseiHierarchyInSitu, self)._initialize_grid_arrays()
        self.grid_procs = np.zeros((self.num_grids,1),'int32')
        
    def _fill_arrays(self, ei, si, LE, RE):
        self.grid_dimensions.flat[:] = ei
        self.grid_dimensions -= np.array(si, dtype='i4')
        self.grid_dimensions += 1
        self.grid_left_edge.flat[:] = LE
        self.grid_right_edge.flat[:] = RE

    def _detect_output_fields(self):
        field_names = self.dataset.amrdata.ArrayNames
        self.field_list = []
        for fn in field_names:
            self.field_list.append(('sensei', fn))
            
    def _chunk_io(self, dobj, cache = True, local_only = False):
        gfiles = defaultdict(list)
        gobjs = getattr(dobj._current_chunk, "objs", dobj._chunk_info)
        for g in gobjs:
            gfiles[g.filename].append(g)
        for fn in sorted(gfiles):
            if local_only:
                gobjs = [g for g in gfiles[fn] if g.proc_num == self.comm.rank]
                gfiles[fn] = gobjs
            gs = gfiles[fn]
            count = self._count_selection(dobj, gs)
            yield YTDataChunk(dobj, "io", gs, count, cache = cache)


class SenseiHierarchyInSitu1D(SenseiHierarchyInSitu):

    def _fill_arrays(self, ei, si, LE, RE):
        self.grid_dimensions[:,:1] = ei[:1]
        self.grid_dimensions[:,:1] -= np.array(si[:1], dtype='i4')
        self.grid_dimensions += 1
        self.grid_left_edge[:,:1] = LE[:1]
        self.grid_right_edge[:,:1] = RE[:1]
        self.grid_left_edge[:,1:] = 0.0
        self.grid_right_edge[:,1:] = 1.0
        self.grid_dimensions[:,1:] = 1

        
class SenseiHierarchyInSitu2D(SenseiHierarchyInSitu):

    def _fill_arrays(self, ei, si, LE, RE):
        self.grid_dimensions[:,:2] = np.array(ei)[:,:2]
        self.grid_dimensions[:,:2] -= np.array(np.array(si)[:,:2], dtype='i4')
        self.grid_dimensions += 1
        self.grid_left_edge[:,:2] = np.array(LE)[:,:2]
        self.grid_right_edge[:,:2] = np.array(RE)[:,:2]
        self.grid_left_edge[:,2] = 0.0
        self.grid_right_edge[:,2] = 1.0
        self.grid_dimensions[:,2] = 1

        
class SenseiDatasetInSitu(Dataset):
    """

    Sensei-specific output, set at a fixed time.

    """

    _index_class = SenseiHierarchyInSitu
    _field_info_class = SenseiFieldInfo
    _dataset_type = 'sensei_insitu'
    
    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj.__init__(*args, **kwargs)
        return obj

    def __init__(self, adaptor, meshName, arrayCen, arrayName):

        self.fluid_types += ("sensei",)

        self.adaptor = adaptor
        self.names = []
        self.meshes = {}
        nMeshes = adaptor.GetNumberOfMeshes()
        i = 0
        while i < nMeshes:
            name = adaptor.GetMeshName(i)
            self.names.append(name)
            self.meshes[name] = SenseiMetaData(adaptor, name)
            i += 1
        # print it for debugging
        mylog.debug('==== metadata ====\n')
        for name in self.names:
            mylog.debug('mesh "%s" = {%s }\n'%(name, str(self.meshes[name])))

        # get the local data
        self.amrdata = self.meshes[meshName]
        self.mesh = self.meshes[meshName].Mesh

        # pull the array from the sim
        self.adaptor.AddArray(self.mesh, meshName, arrayCen, arrayName)

        self.current_time = self.adaptor.GetDataTime()
        self.iteration = self.adaptor.GetDataTimeStep()
        self.unique_identifier = self.current_time
        self.cosmological_simulation = False

        super(SenseiDatasetInSitu, self).__init__("SenseiInSituDataset_%.5d" % self.iteration, self._dataset_type)
        
    def _setup_1d(self):
        self._index_class = SenseiHierarchyInSitu1D
        self.domain_left_edge = \
            np.concatenate([[self.domain_left_edge], [0.0, 0.0]])
        self.domain_right_edge = \
            np.concatenate([[self.domain_right_edge], [1.0, 1.0]])
        self.base_grid_dx = \
            np.concatenate([self.base_grid_dx, [1.0, 1.0]])

    def _setup_2d(self):
        self._index_class = SenseiHierarchyInSitu2D
        self.domain_left_edge = \
            np.concatenate([self.domain_left_edge, [0.0]])
        self.domain_right_edge = \
            np.concatenate([self.domain_right_edge, [1.0]])
        self.base_grid_dx = \
            np.concatenate([self.base_grid_dx, [1.0]])
        
    def _parse_parameter_file(self):
        self.refine_by = self.amrdata.RefRatio[0][0]
        self.periodicity = ensure_tuple([True, True, True])
        self.dimensionality = 2
        self.domain_left_edge, self.domain_right_edge = self.amrdata.get_domain_edges(self.dimensionality)
        self.base_grid_dx = np.array(self.amrdata.Spacing)[:self.dimensionality]

        if self.dimensionality == 1:
            self._setup_1d()
        elif self.dimensionality == 2:
            self._setup_2d()

        domain_dimensions = (self.domain_right_edge - self.domain_left_edge) / self.base_grid_dx
        self.domain_dimensions = np.array(domain_dimensions, dtype=np.int32)
            
    def _set_code_unit_attributes(self):
        mylog.warning("Setting 1.0 in code units to be 1.0 cm")
        mylog.warning("Setting 1.0 in code units to be 1.0 s")
        length_unit = mass_unit = time_unit = 1.0
            
        setdefaultattr(self, 'length_unit', self.quan(length_unit, "cm"))
        setdefaultattr(self, 'mass_unit', self.quan(mass_unit, "g"))
        setdefaultattr(self, 'time_unit', self.quan(time_unit, "s"))
        setdefaultattr(
            self, 'velocity_unit', self.length_unit / self.time_unit)
        
        density_unit = self.mass_unit / self.length_unit**3
        magnetic_unit = np.sqrt(4*np.pi * density_unit) * self.velocity_unit
        magnetic_unit = np.float64(magnetic_unit.in_cgs())
        setdefaultattr(self, 'magnetic_unit', self.quan(magnetic_unit, "gauss"))

    @classmethod
    def _is_valid(cls, *args, **kwargs):
        return False
