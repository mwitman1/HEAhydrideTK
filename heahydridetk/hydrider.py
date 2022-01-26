#! /usr/bin/env python
import numpy as np
import argparse
import itertools

from pymatgen.core.periodic_table import Element
from pprint import pprint

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
             mindistance=2.1):

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

    struct = read(cifname) 
    structcell = struct.cell.cellpar()

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

    
    alldistances = allcombinedsuper.get_all_distances(mic=True)
    allpos = allcombinedsuper.get_positions()

    # get the indices of the interstitials in the ase.Atoms array
    indocto = np.array([i for i in range(len(allcombinedsuper))\
                        if allcombinedsuper[i].symbol=='X'])
    indtetra = np.array([i for i in range(len(allcombinedsuper))\
                        if allcombinedsuper[i].symbol=='Y'])
    # get the indices of the alloy atoms in the ase.Atoms array
    indstruct = np.array([i for i in range(len(allcombinedsuper))\
                         if allcombinedsuper[i].symbol!='X' and allcombinedsuper[i].symbol!='Y'])
    indall = np.array([i for i in range(len(allcombinedsuper))])


    f = open(interstitialsname,"w")

    alldata = []
    f.write("Isite_ind Isite_type At1 At2 At3 At4 At5 At6 Isite_x Isite_y Isite_z mean_X std_X mean_Rad std_Rad\n") 
    for i in indocto:
        N=6
        summarystr,data = enumerate_Nth_polyhedra(i, N, indstruct, allcombinedsuper, 
                                             alldistances, allpos, debug=False)
        alldata.append(data)
        allcombinedsuper[i].position = data[3:6]
        allcombinedsuper[i].symbol = 'H'
        f.write(summarystr)
        
    for i in indtetra:
        N=4
        summarystr,data = enumerate_Nth_polyhedra(i, N, indstruct, allcombinedsuper, 
                                             alldistances, allpos, debug=False)
        alldata.append(data)
        allcombinedsuper[i].position = data[3:6]
        allcombinedsuper[i].symbol = 'H'
        f.write(summarystr)
        
    f.close() 

    if writeall:
        savename = structname+'_allH.cif'
        print("Saving structure with hydrogens at all interstices to: %s"%savename)
        write(savename,allcombinedsuper) 

    if writesequential:
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
                write(savename,struct+allcombinedsuper[placed_ind])

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
                write(savename,struct+allcombinedsuper[placed_ind])

        print("H/M = %.1f after filling octahedral holes"%(numplaced/len(struct)))

    if np.sum(writerandom) != 0:
        numHocta = np.int(writerandom[0]*len(indocto))
        numHtetra = np.int(writerandom[1]*len(indtetra))

        np.random.seed(writerandom[3])

        octositesuper.set_atomic_numbers([1 for _ in range(len(octositesuper))])
        tetrasitesuper.set_atomic_numbers([1 for _ in range(len(tetrasitesuper))])

        for i in range(writerandom[2]):
            savename = structname+'_N%d-i%d_R%.4f-%.4f-%d-%d.cif'%(numHocta+numHtetra,i,
                                                                    writerandom[0],
                                                                    writerandom[1],
                                                                    writerandom[2],
                                                                    writerandom[3])
            write(savename, struct + \
                            octositesuper[np.random.choice(len(octositesuper),numHocta)]+\
                            tetrasitesuper[np.random.choice(len(tetrasitesuper),numHtetra)])



    

#if __name__=="__main__":
#
#    # must provide:
#    # 1. Structure file name
#    # 2. Lattice type (FCC or BCC, but MUST be cubic representation
#    # 3. Number of replications of the simple cubic cell
#    # 4. Write all possible H sites
#    # 5. Write sequentially H sites (starting with trahedral) filling sites 
#    #    of increasing electronegativity if the new site is more than mindistance
#    #    from an H that's already been placed 
#
#    parser = argparse.ArgumentParser()
#    parser.add_argument("-cifname", type=str, 
#                        help="filename of BCC or FCC lattice to find "+\
#                             "octahedral/tetrahedral sites" )
#    parser.add_argument("-lattice_type",type=str, help="FCC or BCC")
#    parser.add_argument("-supercell",type=int,nargs=3,
#                        help="how many replications of the basic cubic structure")
#    parser.add_argument("--writeall",type=int,default=0, 
#                        help="Number of hydrogens per metal to place")
#    parser.add_argument("--writesequential",type=int,default=0, 
#                        help="Fill structure with H at interstices, 2 at a time, from low electronegativity to high")
#    
#    parser.add_argument("--mindistance",type=float,default=2.1, 
#                        help="The minimum allowable distance between two H atoms")
#    args = parser.parse_args()
#    print(vars(args))
#    hydrider(**vars(args))
