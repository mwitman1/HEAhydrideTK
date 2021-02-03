#! /usr/bin/env python3                                                         
                                                                                
from heahydridetk.sshea import SSHEA                                            
from heahydridetk.hydrider import hydrider                                      

import argparse

def generate_structures(latticetype, comp, numatoms, radius, scale, seed, tag,
                        outformats, cifname, supercell, writeall, 
                        writesequential, mindistance):

    if comp is None and cifname is None:
        raise Exception("If comp is not specified, cifname must be "
                        "provided as the starting lattice for hydriding")
    if comp is not None and cifname is not None:
        raise Exception("Both comp and cifname may not be specified ")
    
   
    if comp is not None:
        print("Generating random  HEA alloy and hydride") 
        # A random SS alloy from composition, crystal class, and other params
        sshea = SSHEA(comp, latticetype, numatoms = numatoms, radius = radius,
                      scale = scale,
                      seed = seed,
                      tag = tag,
                      outformats = outformats)
        struct, name = sshea.generate_new_structure(seed=1,tag=1,write=True)            
  
        # generate the hydride structures 
        for fmt in outformats:
            hydrider(name+"." + fmt, latticetype, sshea.finalreps,
                     writeall, writesequential, mindistance)

    else:
        print("Hydriding pre-existing HEA alloy: %s"%cifname)
        if supercell is None:
            raise Exception("If loading a previous cif structure, "
                            "supercell replication of this simple cubic structure "
                            "must be specified") 

        # generate the hydride structures 
        for fmt in outformats:
            hydrider(cifname, latticetype, supercell,
                     writeall, writesequential, mindistance)


def main():

    parser = argparse.ArgumentParser()                                          
                                                                                
    parser.add_argument("latticetype", type=str, choices=['fcc','bcc'],           
                        help="the lattice type for the crystal")                
                                                                                
    parser.add_argument("--comp", type=str, default=None,
        help="ALLOY cmd: Generate a random HEA alloy with composition: "
             "e.g. Co0.5CrMnFeZnTi0.5 "
             "If None, no new HEA alloy will be generated")

    parser.add_argument("--numatoms", type=int, default=100,                      
        help="ALLOY cmd: create the smallest supercell with at least this many "
             "number of atoms")
                                                                                
    parser.add_argument("--radius", type=None, default=0.0,                       
        help="ALLOY cmd: nearest neighbor (nn) distance is computed based on "
             "this radius (default=0.0 takes the max elemental radius.")

    parser.add_argument("--scale", type=float, default=1.0,                       
        help="ALLOY cmd: scaling factor to compute nn distance s.t. "
             "dist = 2*radius*scale")                                           
                                                                                
    parser.add_argument("--outformats",nargs='*', default=['cif'],        
        help="ALLOY cmd: list of ase compatible output formats, "
             "e.g. ['cif','vasp'] " )
                                                                                
    parser.add_argument("--seed", type=int, default=None,                         
        help="ALLOY cmd: A seed for randomly permuting the elemental identities "
             "(Defalut=None means system will randomly set seed")               
                                                                                
    parser.add_argument("--tag", type=str, default=None,                          
        help="ALLOY cmd: An additional identifying tag in the output file names")

    parser.add_argument("--cifname", type=str, default=None,
        help="HYDRIDE cmd: If specified, use this as the starting point for "
             "lattice hydriding. Must be a CIF file.")

    parser.add_argument("--supercell", type=int, nargs=3,                             
        help="HYDRIDE cmd: how many replications of the basic cubic structure "
             "exist in cifname, if specified, which are needed to replicate "
             "the octahedral/tetrahedral sites in the supercell")

    parser.add_argument("--writeall", action='store_true',                        
        help="HYDRIDE cmd: If True, write a structure that has H at all interstices")

    parser.add_argument("--writesequential", action='store_true',
        help="HYDRIDE cmd: Fill structure with H at interstices, 2 at a time"+\
             "according to type and electronegativity rules, until no more "
             "can be placed without violating mindistance")
                                                                                
    parser.add_argument("--mindistance", type=float, default=2.1,                 
        help="HYDRIDE cmd: The minimum allowable distance between two H atoms")
                                                                                
                                                                                
    args = parser.parse_args()            

    generate_structures(**vars(args))
    
