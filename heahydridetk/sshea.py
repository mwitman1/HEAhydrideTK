import argparse
import numpy as np
from iteround import saferound

from ase.build import bulk, make_supercell
from ase.io import read, write

import pymatgen as pym

class SSHEA(object):
    """ A class for generating Solid Solution-High Entropy Alloys (SS-HEAs)"""

    def __init__(self, comp, latticetype, numatoms = 100, radius = 0, 
                 scale = 1, seed=None, tag=None, outformats = ['cif']):
        """Prepare the desired lattice/compositional attributes of the SS-HEA

        Args:
            comp (str): A string of the alloy composition
            latticetype (str): either "fcc" or "bcc"
            numatoms (int): create the smallest supercell with at
                            least this many number of atoms
            radius (float): radius of the largest element (default = 0.0
                            and will automatically be computed)
            scale (float): scaling factor for setting the nearest neighbor
                           distance where distance = 2*radius*scale
                           (default = 1.0) 
            seed (int): A seed for randomly permuting the elemental identities
                       (defalut=None means system will randomly set seed)
            tag (str): An additional identifying tag in the output file names
            outformats (list of str): ase compatible output formats


        """
            
        # Store composition information
        self.compstr = comp
        self.comp = pym.core.Composition(comp)
        self.eldict = self.comp.get_el_amt_dict()

        # file types to output
        self.outformats = outformats
   
        # minimum number of atoms in unit cell
        self.numatoms = numatoms 
       
        # Assign the desiree interatomic distance 
        if radius == 0.0:
            # get largest atomic radius in composition
            Rs = [pym.core.Element(el).atomic_radius for el in self.eldict.keys()]
            Reff = np.max(Rs)*scale
        else:
            Reff = radius*scale

        # compute the lattice a vector that gives the desired spacing
        self.latticetype = latticetype.lower()
        if self.latticetype == "fcc":
            self.latticevector = 4*Reff/2**(1/2)
        elif self.latticetype == "bcc":
            self.latticevector = 4*Reff/3**(1/2)
        else:
            raise ValueError("Only fcc or bcc crystal type supported")

        # initial seed for random site ordering
        np.random.seed(seed)

        # For now only deal with building cubic cells
        self.struct = bulk("Cu", self.latticetype, a=self.latticevector,
                           cubic=True)

        # Obtain the required supercell replication
        initnum = len(self.struct)
        numreps = int(np.ceil(self.numatoms/initnum))
        finalrep1 = int(np.ceil(numreps**(1/3)))
        if finalrep1*(finalrep1-1)**2 > numreps:
            finalrep2 = finalrep1-1
            finalrep3 = finalrep1-1
        elif finalrep1**2*(finalrep1-1) > numreps:
            finalrep2 = finalrep1
            finalrep3 = finalrep1-1
        else:
            finalrep2 = finalrep1
            finalrep3 = finalrep1
        self.finalreps = [finalrep1, finalrep2, finalrep3]

        # create supercell
        self.struct = make_supercell(self.struct, np.diagflat(self.finalreps))

    def generate_new_structure(self, seed=0, tag=None, write=True):
        """ Generate the SS-HEA and write corresponding simulation files

        Uses the random seed (for reproducability) to permute the atom indices
        for assignment to their new elemental identities and serves as an 
        identifying tag for this structure

        Args:
            seed (int): overwrite the seed for this permutation (default = 0)
            write (bool): write any requested ase compatible file types

        Returns:
            self.struct (Atoms): the ase atoms object of the new SS-HEA

        """

        # overwrite the seed for this config (more reproducible)
        if seed is not None:
            np.random.seed(seed)
            self.seed = seed

        # can use this to double check that the interatomic distance equals
        # what was requested
        #distances = self.struct.get_all_distances(mic=True)

        # Now randomly shuffle the indices of all atoms
        inds = [i for i in range(len(self.struct))]
        np.random.shuffle(inds)
        
        # Sort the composition from lowest to highest at.%
        data = sorted([(key, val) for key, val in self.eldict.items()], 
                      key=lambda tup: tup[1])
        totalstoich = np.sum([val[1] for val in data])

        # numpts gives the number of atoms for each element type that should
        # exist in the final material, which is a close to the original at.%
        # as possible
        numpts = saferound([val[1]/totalstoich*len(inds) for val in data],
                           places=0)
        numpts = [int(i) for i in numpts]
        if np.sum(numpts) != len(self.struct):  
            raise Exception("I messed up")

        # subarrays contains the indices to assing to each element type
        subarrays = np.split(inds, np.cumsum(numpts)[:-1])

        # reassign the element identities based on subarrays
        newlabels = []
        newtypes = []
        for i in range(len(subarrays)):
            if len(subarrays[i]) == 0:
                print("Warning, supercell was not large enough to incorporate "
                      "%s with at. frac. = %.3f"%\
                      (data[i][0],data[i][1]/totalstoich))
            for ind in subarrays[i]:
                self.struct.symbols[ind] = data[i][0]

        basename=None
        if write:
            basename = self.write_output(self.outformats, self.latticetype, seed, tag)

        return self.struct, basename


    def write_output(self, outformats, latticetype, seed, tag):

        basename = self.compstr+'_%s_rep%dx%dx%d_s%s'%(latticetype, *self.finalreps, str(seed))

        if tag != None:
            basename += '_tag%s'%(tag)

        for form in outformats:
            print("Writing file to: %s.%s"%(basename,form))
            write('%s.%s'%(basename,form),self.struct,format=form)

        return basename
