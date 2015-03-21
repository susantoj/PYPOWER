# Copyright (C) 1996-2011 Power System Engineering Research Center (PSERC)
# Copyright (C) 2011 Richard Lincoln
#
# PYPOWER is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# PYPOWER is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PYPOWER. If not, see <http://www.gnu.org/licenses/>.

"""
Calculate three-phase faults on all busbars
"""

from scipy.sparse.linalg import splu
import numpy as np

from printfault import printfault

from pypower.ext2int import ext2int
from pypower.int2ext import int2ext
from pypower.makeYbus import makeYbus
from pypower.loadcase import loadcase
from pypower.idx_bus import BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, FAULT_MVA, \
    VM, VA, VMAX, VMIN, LAM_P, LAM_Q, MU_VMAX, MU_VMIN, REF
from pypower.idx_gen import RGEN, XDPP, GEN_BUS

def runfault_3ph(ppc):
    """
    Runs a three-phase fault calculation
    Based on per-unit ANSI method with pre-fault voltage of 1pu on all buses
    """
    
    # Build Ybus matrix
    ppc_int = ext2int(ppc)
    baseMVA, bus, branch = ppc_int["baseMVA"], ppc_int["bus"], ppc_int["branch"]
    Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)

    # Check if any generator impedances are zero
    Zg = ppc_int['gen'][:, RGEN] + ppc_int['gen'][:, XDPP]

    if np.count_nonzero(Zg) == len(Zg):
        # Adjust Ybus matrix with generator shunt admittances
        Yg = 1 / (ppc_int['gen'][:, RGEN] + 1j * ppc_int['gen'][:, XDPP])
        gen_bus = ppc_int['gen'][:, GEN_BUS]

        # Add shunts to Ybus main diagonal
        for i in range(len(Yg)):
            j = gen_bus[i]
            Ybus[j,j] = Ybus[j,j] + Yg[i]

        # Factorise Ybus matrix
        Ybus_inv = splu(Ybus)

        # Calculate Thevenin equivalent impedance and fault MVA
        no_bus = len(ppc_int["bus"])
        Sf = np.zeros(no_bus, dtype='float')
        for i in range(no_bus):
            I = np.zeros(no_bus, dtype='complex')
            I[i] = 1
            Zth = Ybus_inv.solve(I) 
            
            # Calculate fault MVA
            Sf[i] = np.abs(1 / Zth[i] * baseMVA)
        
        ppc_int["bus"][:, FAULT_MVA] = Sf
        results = int2ext(ppc_int)
        
        success = True
        
        # Print results
        printfault(baseMVA, results["bus"], results["gen"], results["branch"])
        
    else:
        # Zero elements in generator impedances
        print('One or more generator impedances are zero...')
        success = False
        results = None
    
    return success, results
    
if __name__ == '__main__':
    # Load PYPOWER case
    ppc = loadcase('case9.py')
    success, results = runfault_3ph(ppc)