import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm


class D2Q9Lattice:

    # Lattice Velocities
    c_list = [(x, y) for x in [0, 1, -1] for y in [0, 1, -1]]
    c = np.array(c_list)
    c_opposite = [c.tolist().index((-c[i]).tolist()) for i in range(9)]

    # Velocity weights
    weights = np.ones(9) * (1 / 36)
    weights[np.array([norm(c[i]) < 1.05 for i in range(9)])] = 1 / 9
    weights[0] = 4 / 9

    # Incoming columns
    incoming_right = np.arange(9)[np.array([vel[0] == -1 for vel in c])]
    incoming_left = np.arange(9)[np.array([vel[0] == 1 for vel in c])]
    center = np.arange(9)[np.array([vel[0] == 0 for vel in c])]

    def __init__(self, Nx, Ny, obstacles=None):
        self.Nx = Nx
        self.Ny = Ny
        self.Fin = np.zeros()
        self.Fout = np.zeros()
        self.obstacles = obstacles

    def stream(self):
        pass

    def collide(self):
        pass
