
import sys
from objects import D2Q9Lattice
import numpy as np
import matplotlib.pyplot as plt

lattice = D2Q9Lattice('input/chip.png', 0.04, 20, animate=True)
# lattice = D2Q9Lattice('input/CMNLISE_lowres.png', 0.02, 3000, animate=True)
maxiter = 10000

lattice.run(maxiter)
lattice.show_performance()
lattice.make_animation()
