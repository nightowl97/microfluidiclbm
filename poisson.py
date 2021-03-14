import numpy as np
from objects import D2Q9Lattice
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation

# Get density data at iteration 50
lat2 = D2Q9Lattice('input/chip.png', 0.09, 30, animate=False)
# lat2.run(100)
# # lat2.rho_f.transpose().tofile("rho_f")
# # lat2.rho_g.transpose().tofile("rho_g")
# # lat2.u_prime.transpose().tofile("u_prime")
f = np.squeeze(np.fromfile("rho_f").reshape(lat2.rho_f.transpose().shape))
g = np.squeeze(np.fromfile("rho_g").reshape(lat2.rho_g.transpose().shape))
u = np.fromfile("u_prime").reshape(lat2.u_prime.transpose().shape)
# # print(u.shape)
#
# plt.subplots(3, 1)
# plt.subplot(3, 1, 1)
# plt.imshow(f)
# plt.subplot(3, 1, 2)
# plt.imshow(g)
# plt.subplot(3, 1, 3)
# x, y = np.meshgrid(range(lat2.Nx), range(lat2.Ny))
# plt.imshow(u[0, :, :] ** 2 + u[1, :, :] ** 2, vmax=0.1)
# print(u.shape)
# plt.streamplot(x, y, u[0, :, :], u[1, :, :],
#                density=2.5, arrowsize=.5)
# plt.show()
#
phi = np.zeros_like(f)
# phi_p = np.zeros_like(phi)
n = 100
tol = .001
dx, dy = 4 / lat2.Nx, 1 / lat2.Ny
b = f - g

x, y = np.linspace(0, lat2.Nx, lat2.Nx), np.linspace(0, lat2.Ny, lat2.Ny)
X, Y = np.meshgrid(x, y)

stop = False

while not stop:
    plt.cla()
    phi_p = phi.copy()
    phi[1:-1, 1: -1] = (dx ** 2 * (phi_p[1:-1, 0:-2] + phi_p[1:-1, 2:]) + dy ** 2 * (
                phi_p[0:-2, 1:-1] + phi_p[2:, 1:-1]) \
                        - b[1:-1, 1:-1] * dx ** 2 * dy ** 2) / (2 * (dx ** 2 + dy ** 2))
    phi[0, :], phi[-1, :] = phi[1, :], phi[-2, :]
    phi[:, 0], phi[:, -1] = phi[:, 1], phi[:, -2]
    diff = phi - phi_p
    stop = np.linalg.norm(diff) < tol
    # print(np.linalg.norm(diff))

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
graph = ax.plot_surface(X, Y, phi, cmap=cm.coolwarm_r, linewidth=0, antialiased=False)
plt.show()

phi = phi.transpose()
E = np.zeros((lat2.Nx, lat2.Ny, 2))

E[:-1, :, 0], E[:, :-1, 1] = np.diff(phi, axis=0), np.diff(phi, axis=1)  # does not set last element


# plt.figure()
# print(X.shape)
# print(E[:, :, 0].transpose().shape)
# color = E[:, :, 0] ** 2 + E[:, :, 1] ** 2
# plt.subplot(2,1,1)
plt.quiver(X, Y, E[:, :, 0].transpose(), E[:, :, 1].transpose(), cmap=cm.coolwarm_r)
# plt.streamplot(X, Y, E[:, :, 0].transpose(), E[:, :, 1].transpose(), cmap=cm.coolwarm_r)
# plt.subplot(2,1,2)
# plt.imshow(phi.transpose(), cmap=cm.coolwarm_r)

plt.show()
