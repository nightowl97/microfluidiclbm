
import sys
from objects import D2Q9Lattice

lattice = D2Q9Lattice('input/chip.bmp', 0.05, 20, animate=True)
# lattice = D2Q9Lattice('input/CMNLISE_lowres.png', 0.02, 3000, animate=True)
maxiter = 200
# sys.stdout.write("\nCurrent step took: \r{} seconds".format(time.time() - start_time))
lattice.run(maxiter)
lattice.show_performance()
lattice.make_animation()
