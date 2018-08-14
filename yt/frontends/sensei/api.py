"""
API for yt.frontends.sensei



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2017, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

from .data_structures import \
      SenseiGridInSitu, \
      SenseiHierarchyInSitu, \
      SenseiDatasetInSitu

from .fields import \
      SenseiFieldInfo
add_sensei_field = SenseiFieldInfo.add_field

from .io import \
      IOHandlerInSitu

from .misc import \
    amrdata
