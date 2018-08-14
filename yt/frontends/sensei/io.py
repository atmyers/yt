"""
Sensei-specific IO functions



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2017, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

from yt.utilities.io_handler import \
    BaseIOHandler
from yt.utilities.logger import ytLogger as mylog
from yt.geometry.selection_routines import GridSelector
import vtk.util.numpy_support as vtknp
import numpy as np


class IOHandlerInSitu(BaseIOHandler):

    _dataset_type = "sensei_insitu"

    def __init__(self, ds):
        self.ds = ds
        self.mesh = self.ds.mesh
        BaseIOHandler.__init__(self, ds)

    def _read_fluid_selection(self, chunks, selector, fields, size):
        rv = {}
        # Now we have to do something unpleasant
        chunks = list(chunks)
        if isinstance(selector, GridSelector):
            if not (len(chunks) == len(chunks[0].objs) == 1):
                raise RuntimeError
            g = chunks[0].objs[0]
            for ftype, fname in fields:
                patch = self.mesh.GetDataSet(g.Level, g._idx)
                data = vtknp.vtk_to_numpy(patch.GetCellData().GetArray(fname))
                rv[(ftype, fname)] = data.reshape(g.ActiveDimensions, order='F')
            return rv
        if size is None:
            size = sum((g.count(selector) for chunk in chunks
                        for g in chunk.objs))
        for field in fields:
            ftype, fname = field
            fsize = size
            rv[field] = np.empty(fsize, dtype="float64")
        ng = sum(len(c.objs) for c in chunks)
        mylog.debug("Reading %s cells of %s fields in %s grids",
                   size, [f2 for f1, f2 in fields], ng)
        ind = 0
        for chunk in chunks:
            for g in chunk.objs:
                for field in fields:
                    ftype, fname = field
                    patch = self.mesh.GetDataSet(g.Level, g._idx)
                    data = vtknp.vtk_to_numpy(patch.GetCellData().GetArray(fname))
                    data = data.reshape(g.ActiveDimensions, order='F')
                    nd = g.select(selector, data, rv[field], ind)
                ind += nd
        assert(ind == fsize)
        return rv
