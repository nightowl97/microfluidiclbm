#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from helpers import *


def sum_pops(f): return np.sum(f, axis=0)


def equilibrium(density, v):
    u_squared = (3 / 2) * (v[:, :, 0] ** 2 + v[:, :, 1] ** 2)
    cu = np.dot(v, c.transpose())
    feq = np.zeros((Nx, Ny, Q))
    for vel in range(Q):
        feq[:, :, vel] = weights[vel] * density * (1 + 3 * cu[:, :, vel] + (9 / 2) * (cu[:, :, vel] ** 2) - u_squared)
    return feq


Nt = 4000
Nx = 400
Ny = 100
rho0 = 1
tau = 0.5
uL = 0.04

# Lattice
D, Q = 2, 9
c = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]])
c_opposite = [0, 2, 1, 4, 3, 6, 5, 8, 7]
w0, w1, w2 = 4 / 9, 1 / 9, 1 / 36
weights = [w0, w1, w1, w1, w1, w2, w2, w2, w2]
incoming_right = [2, 6, 8]  # right wall population indices on c
center = [0, 3, 4]
incoming_left = [1, 5, 7]  # left wall indices

# Obstacle
obstacle = np.zeros((Nx, Ny), dtype=bool)
# obstacle[50:100, 33:66] = True

# Inlet velocity
inlet_vel = np.fromfunction(lambda y, d: (1 - d) * uL * (1 + 0.001 * np.sin(2 * np.pi * (y / Ny))), (Ny, 2))

# Initial conditions
rho = rho0 * np.ones((Nx, Ny))
# Fin = np.zeros((Nx, Ny, Q))
# for i in range(Q):
#     Fin[:, :, i] = weights[i]  * rho
# Initial velocity same as inlet
ini_vel = np.repeat(inlet_vel[np.newaxis, :, :], Nx, axis=0)
Fin = equilibrium(rho, ini_vel)


for t in range(Nt):
    Fin[-1, :, incoming_right] = Fin[-2, :, incoming_right]
    rho = np.sum(Fin, axis=2)
    rho_expanded = np.expand_dims(rho, axis=-1)
    u = (1 / rho_expanded) * np.dot(Fin, c)

    # Zou/He Inflow boundary condition (See J. Latt slides p.71),
    u[0, :, :] = inlet_vel
    rho[0, :] = (1 / (1 - u[0, :, 0])) * (sum_pops(Fin[0, :, center]) + 2 * sum_pops(Fin[0, :, incoming_right]))
    Feq = equilibrium(rho, u)  # Equilibrium
    Fin[0, :, incoming_left] = Feq[0, :, incoming_left] + (Fin[0, :, incoming_right] - Feq[0, :, incoming_right])

    # Collision
    Fout = Fin - tau * (Fin - Feq)

    # Bounce-back sites
    for vel in range(Q):
        Fout[obstacle, vel] = Fin[obstacle, c_opposite[vel]]
        Fout[:, Ny - 1, vel] = Fin[:, Ny - 1, c_opposite[vel]]

    # Streaming
    for vel in range(Q):
        Fin[:, :, vel] = np.roll(np.roll(Fout[:, :, vel], c[vel, 0], axis=0), c[vel, 1], axis=1)

    if t % 2 == 0:
        print(t)
        plt.clf()
        plt.imshow(np.sqrt(u[:, :, 0] ** 2 + u[:, :, 1] ** 2).transpose())
        plt.pause(0.001)

print("Done")
