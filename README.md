# Installation
pip install .

# Example
```python
#! /usr/bin/env python3

from heahydridetk.sshea import SSHEA
from heahydridetk.hydrider import hydrider

# create a random SS alloy by composition, crystall class, and seed
sshea = SSHEA("TiVCrNb","FCC",outformats=['cif'])
struct, name = sshea.generate_new_structure(seed=1,tag=1,write=True)

#Insert hydrogens according to heuristic rules in hydrider function
# need to know how many the abc replications (stored in sshea.finalreps)
hydrider(name+".cif",'FCC',sshea.finalreps,writesequential=True,mindistance=1.8)
```

