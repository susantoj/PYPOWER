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
Prints fault analysis results.
"""

from sys import stdout

from numpy import \
    ones, zeros, r_, sort, exp, pi, diff, arange, min, \
    argmin, argmax, logical_or, real, imag, any, sqrt

from numpy import flatnonzero as find

from pypower.idx_bus import BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, \
    BASE_KV, VM, VA, FAULT_MVA
from pypower.idx_gen import GEN_BUS, PG, QG, QMAX, QMIN, GEN_STATUS, \
    PMAX, PMIN, MU_PMAX, MU_PMIN, MU_QMAX, MU_QMIN
from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, \
    TAP, SHIFT, BR_STATUS, PF, QF, PT, QT, MU_SF, MU_ST

from pypower.isload import isload
from pypower.run_userfcn import run_userfcn
from pypower.ppoption import ppoption


def printfault(baseMVA, bus=None, gen=None, branch=None, f=None, success=None,
            et=None, fd=None, ppopt=None):
    """
    Prints fault analysis results.
    """
    ##----- initialization -----
    ## default arguments
    if isinstance(baseMVA, dict):
        have_results_struct = 1
        results = baseMVA
        if gen is None:
            ppopt = ppoption()   ## use default options
        else:
            ppopt = gen
        if (ppopt['OUT_ALL'] == 0):
            return     ## nothin' to see here, bail out now
        if bus is None:
            fd = stdout         ## print to stdout by default
        else:
            fd = bus
        baseMVA, bus, gen, branch, success, et = \
            results["baseMVA"], results["bus"], results["gen"], \
            results["branch"], results["success"], results["et"]
        if 'f' in results:
            f = results["f"]
        else:
            f = None
    else:
        have_results_struct = 0
        if ppopt is None:
            ppopt = ppoption()   ## use default options
            if fd is None:
                fd = stdout         ## print to stdout by default
        if ppopt['OUT_ALL'] == 0:
            return     ## nothin' to see here, bail out now

    isOPF = f is not None    ## FALSE -> only simple PF data, TRUE -> OPF data

    ## options
    isDC            = ppopt['PF_DC']        ## use DC formulation?
    OUT_ALL         = ppopt['OUT_ALL']
    OUT_ANY         = OUT_ALL == 1     ## set to true if any pretty output is to be generated
    OUT_SYS_SUM     = (OUT_ALL == 1) or ((OUT_ALL == -1) and ppopt['OUT_SYS_SUM'])
    OUT_AREA_SUM    = (OUT_ALL == 1) or ((OUT_ALL == -1) and ppopt['OUT_AREA_SUM'])
    OUT_BUS         = (OUT_ALL == 1) or ((OUT_ALL == -1) and ppopt['OUT_BUS'])
    OUT_BRANCH      = (OUT_ALL == 1) or ((OUT_ALL == -1) and ppopt['OUT_BRANCH'])
    OUT_GEN         = (OUT_ALL == 1) or ((OUT_ALL == -1) and ppopt['OUT_GEN'])
    OUT_ANY         = OUT_ANY | ((OUT_ALL == -1) and
                        (OUT_SYS_SUM or OUT_AREA_SUM or OUT_BUS or
                         OUT_BRANCH or OUT_GEN))

    if OUT_ALL == -1:
        OUT_ALL_LIM = ppopt['OUT_ALL_LIM']
    elif OUT_ALL == 1:
        OUT_ALL_LIM = 2
    else:
        OUT_ALL_LIM = 0

    OUT_ANY         = OUT_ANY or (OUT_ALL_LIM >= 1)
    if OUT_ALL_LIM == -1:
        OUT_V_LIM       = ppopt['OUT_V_LIM']
        OUT_LINE_LIM    = ppopt['OUT_LINE_LIM']
        OUT_PG_LIM      = ppopt['OUT_PG_LIM']
        OUT_QG_LIM      = ppopt['OUT_QG_LIM']
    else:
        OUT_V_LIM       = OUT_ALL_LIM
        OUT_LINE_LIM    = OUT_ALL_LIM
        OUT_PG_LIM      = OUT_ALL_LIM
        OUT_QG_LIM      = OUT_ALL_LIM

    OUT_ANY         = OUT_ANY or ((OUT_ALL_LIM == -1) and (OUT_V_LIM or OUT_LINE_LIM or OUT_PG_LIM or OUT_QG_LIM))
    ptol = 1e-4        ## tolerance for displaying shadow prices

    ## create map of external bus numbers to bus indices
    i2e = bus[:, BUS_I].astype(int)
    e2i = zeros(max(i2e) + 1, int)
    e2i[i2e] = arange(bus.shape[0])

    ## sizes of things
    nb = bus.shape[0]      ## number of buses
    nl = branch.shape[0]   ## number of branches
    ng = gen.shape[0]      ## number of generators

    ## zero out some data to make printout consistent for DC case
    if isDC:
        bus[:, r_[QD, BS]]          = zeros((nb, 2))
        gen[:, r_[QG, QMAX, QMIN]]  = zeros((ng, 3))
        branch[:, r_[BR_R, BR_B]]   = zeros((nl, 2))

    ## parameters
    ties = find(bus[e2i[branch[:, F_BUS].astype(int)], BUS_AREA] !=
                   bus[e2i[branch[:, T_BUS].astype(int)], BUS_AREA])
                            ## area inter-ties
    tap = ones(nl)                           ## default tap ratio = 1 for lines
    xfmr = find(branch[:, TAP])           ## indices of transformers
    tap[xfmr] = branch[xfmr, TAP]            ## include transformer tap ratios
    tap = tap * exp(1j * pi / 180 * branch[:, SHIFT]) ## add phase shifters
    nzld = find((bus[:, PD] != 0.0) | (bus[:, QD] != 0.0))
    sorted_areas = sort(bus[:, BUS_AREA])
    ## area numbers
    s_areas = sorted_areas[r_[1, find(diff(sorted_areas)) + 1]]
    nzsh = find((bus[:, GS] != 0.0) | (bus[:, BS] != 0.0))
    allg = find( ~isload(gen) )
    ong  = find( (gen[:, GEN_STATUS] > 0) & ~isload(gen) )
    onld = find( (gen[:, GEN_STATUS] > 0) &  isload(gen) )
    V = bus[:, VM] * exp(-1j * pi / 180 * bus[:, VA])
    out = find(branch[:, BR_STATUS] == 0)        ## out-of-service branches
    nout = len(out)
    if isDC:
        loss = zeros(nl)
    else:
        loss = baseMVA * abs(V[e2i[ branch[:, F_BUS].astype(int) ]] / tap -
                             V[e2i[ branch[:, T_BUS].astype(int) ]])**2 / \
                    (branch[:, BR_R] - 1j * branch[:, BR_X])

    fchg = abs(V[e2i[ branch[:, F_BUS].astype(int) ]] / tap)**2 * branch[:, BR_B] * baseMVA / 2
    tchg = abs(V[e2i[ branch[:, T_BUS].astype(int) ]]      )**2 * branch[:, BR_B] * baseMVA / 2
    loss[out] = zeros(nout)
    fchg[out] = zeros(nout)
    tchg[out] = zeros(nout)

    ##----- print the stuff -----

    if OUT_SYS_SUM:
        fd.write('\n================================================================================')
        fd.write('\n|     System Summary                                                           |')
        fd.write('\n================================================================================')
        fd.write('\n\nHow many?                How much?              P (MW)            Q (MVAr)')
        fd.write('\n---------------------    -------------------  -------------  -----------------')
        fd.write('\nBuses         %6d     Total Gen Capacity   %7.1f       %7.1f to %.1f' % (nb, sum(gen[allg, PMAX]), sum(gen[allg, QMIN]), sum(gen[allg, QMAX])))
        fd.write('\nGenerators     %5d     On-line Capacity     %7.1f       %7.1f to %.1f' % (len(allg), sum(gen[ong, PMAX]), sum(gen[ong, QMIN]), sum(gen[ong, QMAX])))
        fd.write('\nCommitted Gens %5d     Generation (actual)  %7.1f           %7.1f' % (len(ong), sum(gen[ong, PG]), sum(gen[ong, QG])))
        fd.write('\nLoads          %5d     Load                 %7.1f           %7.1f' % (len(nzld)+len(onld), sum(bus[nzld, PD])-sum(gen[onld, PG]), sum(bus[nzld, QD])-sum(gen[onld, QG])))
        fd.write('\n  Fixed        %5d       Fixed              %7.1f           %7.1f' % (len(nzld), sum(bus[nzld, PD]), sum(bus[nzld, QD])))
        fd.write('\n  Dispatchable %5d       Dispatchable       %7.1f of %-7.1f%7.1f' % (len(onld), -sum(gen[onld, PG]), -sum(gen[onld, PMIN]), -sum(gen[onld, QG])))
        fd.write('\nShunts         %5d     Shunt (inj)          %7.1f           %7.1f' % (len(nzsh),
            -sum(bus[nzsh, VM]**2 * bus[nzsh, GS]), sum(bus[nzsh, VM]**2 * bus[nzsh, BS]) ))
        fd.write('\nBranches       %5d     Losses (I^2 * Z)     %8.2f          %8.2f' % (nl, sum(loss.real), sum(loss.imag) ))
        fd.write('\nTransformers   %5d     Branch Charging (inj)     -            %7.1f' % (len(xfmr), sum(fchg) + sum(tchg) ))
        fd.write('\nInter-ties     %5d     Total Inter-tie Flow %7.1f           %7.1f' % (len(ties), sum(abs(branch[ties, PF]-branch[ties, PT])) / 2, sum(abs(branch[ties, QF]-branch[ties, QT])) / 2))
        fd.write('\nAreas          %5d' % len(s_areas))
        fd.write('\n')
 
    ## bus data
    if OUT_BUS:
        fd.write('\n============================================')
        fd.write('\n|     Bus Data                             |')
        fd.write('\n============================================')
        fd.write('\n Bus      Nom Volt     Three-Phase Fault')
        fd.write('\n  #         kV          MVA         kA')
        fd.write('\n--------------------------------------------')
        for i in range(nb):
            fd.write('\n%3d %11.1f %14.2f %9.2f' % (bus[i, BUS_I], bus[i, BASE_KV], bus[i, FAULT_MVA], bus[i, FAULT_MVA] / sqrt(3) / bus[i, BASE_KV]))
        
        fd.write('\n')


    ## execute userfcn callbacks for 'printpf' stage
    if have_results_struct and 'userfcn' in results:
        if not isOPF:  ## turn off option for all constraints if it isn't an OPF
            ppopt = ppoption(ppopt, 'OUT_ALL_LIM', 0)
        run_userfcn(results["userfcn"], 'printpf', results, fd, ppopt)
