#! /usr/bin/env python
import numpy as np
import argparse
import itertools
import os

from pymatgen.core.periodic_table import Element
from pprint import pprint
from copy import deepcopy

import ase
from ase import neighborlist
from ase.io import read, write
from ase.build import make_supercell

BCC_sites = [[0, 0, 0],[.50,.50,.50]]
FCC_sites = [[0,0,0],[.50,.50,0],[.50,0,.50],[0,0.5,0.5]]

# https://www.tf.uni-kiel.de/matwis/amat/def_en/kap_1/illustr/t1_3_3.html
# c = up, a/b = right/intopage
BCC_octahedrons = [[  0,.50,  0],[.50,  0,  0],
                   [  0,.50,.50],[.50,  0,.50],
                   [.50,.50,  0],[  0,  0,.50]]
#BCC_octahedrons = [[  0,.50,.50],[.50,  0,.50],[.50,.50,0]]

FCC_octahedrons = [[0,0.5,0],[0.5,0,0],
                   [0.5,0.5,0.5], [0,0,0.5]]

#https://www.tf.uni-kiel.de/matwis/amat/def_en/kap_1/illustr/t1_3_4.html
# c = up, a/b = right/intopage
BCC_tetrahedrons = [[.25,.50,  0],[.50,.25,  0],[.75,.50,  0],[.50,.75,  0],
                    [  0,.25,.50],[  0,.50,.25],[  0,.75,.50],[  0,.50,.75],
                    [.25,  0,.50],[.50,  0,.25],[.75,  0,.50],[.50,  0,.75]]
#BCC_tetrahedrons = [0,0,0]
FCC_tetrahedrons = [[.25,.25,.25],[.25,.75,.25],[.75,.25,.25],[.75,.75,.25],
                    [.25,.25,.75],[.25,.75,.75],[.75,.25,.75],[.75,.75,.75]]


def enumerate_Nth_polyhedra(i, N, indstruct, allcombinedsuper, alldistances, allpos, 
                            debug=False):
    """
    i : index of the interstitial of interest
    N : Number of closest neighbors to find
    indstruct : the indices of structural atoms (as opposed to intersticial indices)

    allcombinedsuper : the full Ase atoms object containing all atoms and interstices
    alldistances : the pairwise distance matrix of all structural atoms and interstices
    allpos : cartesian positions of all atoms and interstices in cell

    debug : print out a test polyhedra
    """

    idealpos = allpos[i]

    closestNatoms = indstruct[np.argpartition(alldistances[i][indstruct],N)][:N]


    continuousObj = np.zeros((N,3))
    continuousObj[0] = allpos[closestNatoms[0]]

    for j in range(N):
        mic = ase.geometry.find_mic(allpos[closestNatoms[j]]-idealpos,\
                                    allcombinedsuper.cell)
        continuousObj[j]=idealpos+mic[0]
        

    centroidObj = np.mean(continuousObj,axis=0)

    if debug:
        testObj = ase.Atoms([ase.Atom('X') for _ in range(N)]+[ase.Atom('H')],
                              cell=allcombinedsuper.get_cell(),pbc=True)
        testObj.set_positions(np.vstack((continuousObj,centroidObj)))
        write('%i_testObj_N%d.cif'%(i,N),testObj)


    constituentsObj = [Element(at.symbol) for at in allcombinedsuper[closestNatoms]]
    meanX = np.mean([el.X for el in constituentsObj])
    stdX = np.std([el.X for el in constituentsObj])
    meanCov = np.mean([el.atomic_radius for el in constituentsObj])
    stdCov = np.std([el.atomic_radius for el in constituentsObj])

    if N==4:
        typ = 't'
        constituentstr = " -  - "+" ".join(["%2s"%el for el in constituentsObj])
    elif N==6:
        typ = 'o'
        constituentstr = " ".join(["%2s"%el for el in constituentsObj])
    else:
        raise ValueError("Only N=4 (tetrahedral) and N=6 (octahedral supported)")

    data=(i, typ, constituentstr,centroidObj[0],centroidObj[1],centroidObj[2],
          meanX,stdX,meanCov,stdCov)

    summarystr= "%3d %3s %2s %3.4f %3.4f %3.4f %3.4f %3.4f %3.4f %3.4f\n"%data

    if debug:
        print(i,closestNatoms,alldistances[i][closestNatoms])
        print(summarystr)

    #if i == 107 or i == 7:
    #    print(continuousObj)
    #    print(mic)
    #    print(centroidObj)
    #    sys.exit()
    

    return summarystr, data


def hydrider(cifname, lattice_type, supercell, writeall=0, writesequential=0,writerandom=(0,0,0,0),
             writeallrandom=0,towrite=True, mindistance=2.1,asestruct=None,path='.',seed=0):

    """
    cifname : str
        path to cif file for hydriding
    lattice_type : str
        for now, bcc or fcc lattice
    supercell : tuple(int, int, int)
        number of replications of the primitive fcc or bcc cubic cell
    witeall : bool
        load every interstitial with hydrgoen
    writesequential : bool
        load octahedral intersitials sequentially subject to H-H distance < mindistance
        then while H/M < 2, load tetrahedral sites sequentially 
    writerandom : tuple(float1, float2, int3, int4)
        float1 : fraction of octahedral holes to randomly fill
        float2 : fraction of tetrahedral holes to randomly fill
        int3 : number of random structures to write
        int4 : seed for random cfgs
    mindistance : float
        mindistance is minimum allowable distance for H-H in the lattice 
    """

    lattice_type = lattice_type.upper()
 
    structname = cifname.split('.cif')[0]
    interstitialsname = structname+'.interstitials'

    if asestruct == None:
        struct = read(cifname) 
        structcell = struct.cell.cellpar()
    else:
        # can directly pass in asestruct 
        struct = asestruct
        structcell = struct.cell.cellpar()

    if seed:
        np.random.seed(seed)
        

    basecell=(structcell[0]/supercell[0],
              structcell[1]/supercell[1],
              structcell[2]/supercell[2],
              structcell[3], structcell[4], structcell[5])

    newcell = np.array([[supercell[0],0,0],[0,supercell[1],0],[0,0,supercell[2]]])

    if lattice_type == 'BCC':

        octosite = ase.Atoms([ase.Atom('X') for _ in BCC_octahedrons],
                             cell=basecell,pbc=True)
        octosite.set_scaled_positions(np.array(BCC_octahedrons))
        octositesuper = make_supercell(octosite, newcell)

        tetrasite = ase.Atoms([ase.Atom('Y') for _ in BCC_tetrahedrons],
                              cell=basecell,pbc=True)
        tetrasite.set_scaled_positions(np.array(BCC_tetrahedrons))
        tetrasitesuper = make_supercell(tetrasite, newcell)

        allsite = octosite + tetrasite
        allsitesuper = make_supercell(allsite, newcell) 

        allcombinedsuper = allsitesuper+struct

    elif lattice_type == 'FCC': 

        octosite = ase.Atoms([ase.Atom('X') for _ in FCC_octahedrons],
                             cell=basecell,pbc=True)
        octosite.set_scaled_positions(np.array(FCC_octahedrons))
        octositesuper = make_supercell(octosite, newcell)

        tetrasite = ase.Atoms([ase.Atom('Y') for _ in FCC_tetrahedrons],
                              cell=basecell,pbc=True)
        tetrasite.set_scaled_positions(np.array(FCC_tetrahedrons))
        tetrasitesuper = make_supercell(tetrasite, newcell)

        allsite = octosite + tetrasite
        allsitesuper = make_supercell(allsite, newcell) 

        allcombinedsuper = allsitesuper+struct

    else:
        raise ValueError("%s not supported, only BCC or FCC"%lattice_type)


    # get the indices of the interstitials in the ase.Atoms array
    indocto = np.array([i for i in range(len(allcombinedsuper))\
                        if allcombinedsuper[i].symbol=='X'])
    indtetra = np.array([i for i in range(len(allcombinedsuper))\
                        if allcombinedsuper[i].symbol=='Y'])
    # get the indices of the alloy atoms in the ase.Atoms array
    indstruct = np.array([i for i in range(len(allcombinedsuper))\
                         if allcombinedsuper[i].symbol!='X' and allcombinedsuper[i].symbol!='Y'])
    indall = np.array([i for i in range(len(allcombinedsuper))])

    
    allcombinedsuperHs = deepcopy(allcombinedsuper)
    newatomicnumbers = allcombinedsuper.get_atomic_numbers() 
    newatomicnumbers[np.concatenate((indocto,indtetra))]=1
    allcombinedsuperHs.set_atomic_numbers(newatomicnumbers)


    # write a hydrogen at each interstitial
    if writeall:
        savename = structname+'_allH.cif'
        finalstruct = allcombinedsuperHs
        print("Saving structure with hydrogens at all interstices to: %s"%savename)
        if towrite:
            write(os.path.join(path,savename),finalstruct) 

    # number of randomn hydrogens to insert agnostic to interstitial type
    if writeallrandom:
        # get random sampling of 
        chosen = np.random.choice(len(allsitesuper), writeallrandom, replace=False)
        finalstruct = struct+allcombinedsuperHs[chosen]
        if towrite:
            write(os.path.join(path,savename),finalstruct) 
    else:
        finalstruct = struct

    # sequentially fill interstitials octahedral, then tetrahedral
    if writesequential:
        #f = open(interstitialsname,"w")
        alldistances = allcombinedsuper.get_all_distances(mic=True)
        allpos = allcombinedsuper.get_positions()
        alldata = []
        #f.write("Isite_ind Isite_type At1 At2 At3 At4 At5 At6 Isite_x Isite_y Isite_z mean_X std_X mean_Rad std_Rad\n") 
        for i in indocto:
            N=6
            summarystr,data = enumerate_Nth_polyhedra(i, N, indstruct, allcombinedsuper, 
                                                 alldistances, allpos, debug=False)
            alldata.append(data)
            allcombinedsuper[i].position = data[3:6]
            allcombinedsuper[i].symbol = 'H'
            #f.write(summarystr)
            
        for i in indtetra:
            N=4
            summarystr,data = enumerate_Nth_polyhedra(i, N, indstruct, allcombinedsuper, 
                                                 alldistances, allpos, debug=False)
            alldata.append(data)
            allcombinedsuper[i].position = data[3:6]
            allcombinedsuper[i].symbol = 'H'
            #f.write(summarystr)
        #f.close() 

        alldata.sort(key=lambda x: x[6])

        numplaced = 0
        placed_ind = []
        placed_ind.append(alldata[0][0])
        placed_sortedind = [] 
        placed_sortedind.append(0)
        numplaced+=1

        # fill tetrahedral holes first, at most up to H/M = 2
        for i in range(1,len(alldata)):
            added = False

            if alldata[i][1]=='t' and numplaced <= 2*len(struct):
                # check if this is within the Westlake empirical rule
                # No hydrogens can be less than 2.1 A apart:
                # Violations: http://folk.uio.no/ravi/pub/2002-3-hydride_prl.pdf
            
                thisind = alldata[i][0]
                dist_to_placed = alldistances[thisind,np.array(placed_ind)]


                if np.min(dist_to_placed) > mindistance:
                    placed_ind.append(thisind)
                    placed_sortedind.append(i)
                    numplaced+=1
                    added=True

            if numplaced%2 == 0 and added:
                savename = structname+'_N%d_mind%.1f.cif'%(numplaced,mindistance)
                print("Saving sequentially added H structure: %s"%savename)
                finalstruct = struct+allcombinedsuper[placed_ind]
                if towrite:
                    write(os.path.join(path,savename),finalstruct)

        print("H/M = %.1f after filling tetrahedral holes"%(numplaced/len(struct)))

        # fill octahedral holes next if still haven't reached H/M = 2
        for i in range(1,len(alldata)):
            added=False

            if alldata[i][1]=='o' and numplaced <= 2*len(struct):

                thisind = alldata[i][0]
                dist_to_placed = alldistances[thisind,np.array(placed_ind)]

                if np.min(dist_to_placed) > mindistance:
                    placed_ind.append(thisind)
                    placed_sortedind.append(i)
                    numplaced+=1
                    added=True

            if numplaced%2 == 0 and added:
                savename = structname+'_N%d_mind%.1f.cif'%(numplaced,mindistance)
                print("Saving sequentially added H structure: %s"%savename)
                finalstruct = struct+allcombinedsuper[placed_ind]
                if towrite:
                    write(os.path.join(path,savename),finalstruct)

        print("H/M = %.1f after filling octahedral holes"%(numplaced/len(struct)))


    # write random hydrogens with specific proportions in octa vs tetrahedral holes
    if np.sum(writerandom) != 0:
        numHocta = int(writerandom[0]*len(indocto))
        numHtetra = int(writerandom[1]*len(indtetra))

        np.random.seed(writerandom[3])

        octositeHs = deepcopy(octositesuper)
        tetrasiteHs = deepcopy(tetrasitesuper)

        octositeHs.set_atomic_numbers([1 for _ in range(len(octositeHs))])
        tetrasiteHs.set_atomic_numbers([1 for _ in range(len(tetrasiteHs))])

        for i in range(writerandom[2]):
            savename = structname+'_N%d-i%d_R%.4f-%.4f-%d.cif'%(numHocta+numHtetra,i,
                                                                    writerandom[0],
                                                                    writerandom[1],
                                                                    writerandom[3])

            finalstruct = struct + \
                          octositeHs[np.random.choice(len(octositeHs),numHocta,replace=False)]+\
                          tetrasiteHs[np.random.choice(len(tetrasiteHs),numHtetra,replace=False)]

            if towrite:
                write(os.path.join(path,savename), finalstruct)

    return finalstruct
