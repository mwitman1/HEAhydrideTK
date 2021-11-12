#! /usr/bin/env python3

from copy import deepcopy

from matminer.featurizers.site import ChemEnvSiteFingerprint
import numpy as np
from pprint import pprint
from .hydrider import BCC_sites, FCC_sites,\
                      BCC_octahedrons, FCC_octahedrons,\
                      BCC_tetrahedrons, FCC_tetrahedrons

from ase import Atom, Atoms
from ase.io import read, write

from pymatgen.core.sites import PeriodicSite
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
import logging
from pymatgen.ext.matproj import MPRester
from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import SimplestChemenvStrategy, MultiWeightsChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments



def reverse_analyze2(ase_struct, latticetype, reps):
    """
    Strict assumptions: struct is either bcc or fcc cubic lattice

    ase_struct : an ase.Atoms() object
    latticetype : a member of {'fcc','bcc'}
    reps : tuple of (a, b, c) replications of the primitive cubic lattice cell
    """

    a0 = 2 #arbitrary box length for visualization, but must be same across dataset

    ase_Hs = deepcopy(ase_struct)
    ase_Hs = ase_Hs[np.where(ase_Hs.get_atomic_numbers()==1)[0]]
    ase_struct = ase_struct[np.where(ase_struct.get_atomic_numbers()!=1)[0]]

    if latticetype == "bcc":
        lattice_sites = np.array(BCC_sites)
        lattice_interst = np.array(BCC_octahedrons + BCC_tetrahedrons)
    elif latticetype == "fcc":
        lattice_sites = np.array(FCC_sites)
        lattice_interst = np.array(FCC_octahedrons + FCC_tetrahedrons)
    else:
        raise ValueError("Only 'fcc' or 'bcc' supported for now")


    num_nonH = len(np.where(ase_struct.get_atomic_numbers() != 1)[0])

    super_lattice_sites = []
    super_lattice_interst = []
    for i in range(reps[0]):
        for j in range(reps[1]):
            for k in range(reps[2]):
                super_lattice_sites.append(lattice_sites+np.array([i,j,k]))
                super_lattice_interst.append(lattice_interst+np.array([i,j,k]))

    super_lattice_sites = np.vstack(super_lattice_sites)/\
                          np.array([reps[0],reps[1],reps[2]])
    super_interst_sites = np.vstack(super_lattice_interst)/\
                          np.array([reps[0],reps[1],reps[2]])


    ###################################################################
    # Step 1: Compute mapping of alloy atoms to ideal sites
    assert len(super_lattice_sites) == len(ase_struct), print("Structure (%d) and idealized alloy lattice (%d) have non matching # of atoms"%(len(ase_struct), len(super_lattice_sites)))

    # add the ideal sites to the alloy structure
    scaled_coords = ase_struct.get_scaled_positions()
    for site in super_lattice_sites:
        ase_struct += Atom('X', position=site)
    ase_struct.set_scaled_positions(np.vstack((scaled_coords,super_lattice_sites)))

    # compute pairwise distances of everything (alloy atoms + ideal sites)
    distM = ase_struct.get_all_distances(mic=True)
    nearest_lattice_site = np.argmin(distM[0:len(scaled_coords),
                                           len(scaled_coords):],axis=0)
    ideal_lattice_site_elems = ase_struct.get_atomic_numbers()[nearest_lattice_site]
    #print(distM[len(super_lattice_sites):,len(super_lattice_sites):])
    #print(nearest_lattice_site)
    #print(ideal_lattice_site_elems)

    ideal_lattice = Atoms()
    ideal_lattice.set_cell(a0*np.array([[reps[0],0,0],[0,reps[1],0],[0,0,reps[2]]]))
    for i, site in enumerate(super_lattice_sites):
        # dummy position b/c interpreted as cartesian
        ideal_lattice += Atom(ideal_lattice_site_elems[i],position=super_lattice_sites[i])
    ideal_lattice.set_scaled_positions(super_lattice_sites)
    ###################################################################


    ###################################################################
    # Step 2: Compute mapping of H atoms to ideal interstitials
    scaled_coords = ase_Hs.get_scaled_positions()
    for site in super_interst_sites:
        ase_Hs += Atom('X',position=site)

    ase_Hs.set_scaled_positions(np.vstack((scaled_coords,super_interst_sites)))

    # compute pairwise distances of everything (alloy atoms + ideal sites)
    distM = ase_Hs.get_all_distances(mic=True)
    nearest_interst_site = np.argmin(distM[0:len(scaled_coords),
                                           len(scaled_coords):],axis=1)
    #print(distM[len(super_interst_sites):,len(super_interst_sites):])
    #print(nearest_interst_site)

    ideal_Hs = Atoms()
    ideal_Hs.set_cell(a0*np.array([[reps[0],0,0],[0,reps[1],0],[0,0,reps[2]]]))
    for i, site in enumerate(nearest_interst_site):
        # dummy position b/c interpreted as cartesian
        ideal_Hs += Atom('H',position=super_interst_sites[i])
    ideal_Hs.set_scaled_positions(super_interst_sites[nearest_interst_site])
    ###################################################################

    return ideal_lattice+ideal_Hs

