
import sys
from objects import D2Q9Lattice

# lattice = D2Q9Lattice('input/chip.bmp', 0.04, 200, animate=True)
lattice = D2Q9Lattice('input/CMNLISE_lowres.png', 0.03, 3000, animate=True)
maxiter = 400
# sys.stdout.write("\nCurrent step took: \r{} seconds".format(time.time() - start_time))
lattice.run(maxiter)
lattice.show_performance()
lattice.make_animation()
