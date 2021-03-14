import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt
from numpy.linalg import norm
from PIL import Image
import matplotlib.animation as animation
from datetime import datetime
from progressbar import progressbar
import time
import pyximport
import matplotlib.cm as cm
pyximport.install(language_level=3)
import performance

# np.seterr(all='raise')

# TODO: Colors to define site type (Inlet, Outlet, Bounce-back, Fluid etc.)


class D2Q9Lattice:
    """
    Lattice:
    7 -- 1 -- 4
     \  |   /
      \ | /
    6---0---3
      / | \
     /  |  \
    8---2---5"""
    D, Q = 2, 9
    # Lattice Velocities
    c_list = [(x, y) for x in [0, 1, -1] for y in [0, 1, -1]]
    c = np.array(c_list, dtype=np.intc)

    # Incoming columns
    incoming_right = np.arange(9)[np.array([vel[0] == -1 for vel in c])]
    incoming_left = np.arange(9)[np.array([vel[0] == 1 for vel in c])]
    center = np.arange(9)[np.array([vel[0] == 0 for vel in c])]

    # Interaction strength
    G = 7

    def __init__(self, geometry_image, ini_vel, re, animate=False):
        # Initialize object and get geometry
        self.t = 0
        self.runtime = None
        self.c_opposite = [self.c.tolist().index((-self.c[i]).tolist()) for i in range(9)]
        # Velocity weights
        self.weights = np.ones(9) * (1 / 36)
        self.weights[np.array([norm(self.c[i]) < 1.05 for i in range(9)])] = 1 / 9
        self.weights[0] = 4 / 9

        self.image = np.asarray(Image.open(geometry_image))
        if len(self.image.shape) == 2:
            self.Ny, self.Nx = self.image.shape
            self.geometry = self.image.transpose() == 0
        elif len(self.image.shape) == 3:
            self.Ny, self.Nx, color = self.image.shape
            self.geometry = self.image.transpose()[0] == 0
        else:
            print("Image color profile not supported, exiting..")
            exit(0)

        # Define the different regions of the lattice
        halfheight = self.Ny // 2
        self.fluid = ~self.geometry
        self.inlet_f, self.inlet_g, self.outlet = self.fluid.copy(), self.fluid.copy(), self.fluid.copy()
        self.inlet_f[1:, :], self.inlet_g[1:, :] = False, False
        self.inlet_f[:, :halfheight], self.inlet_g[:, halfheight:] = False, False
        self.outlet[:-1, :] = False
        self.inner_domain = self.fluid & ~self.inlet_f & ~self.inlet_g
        bottom_half, top_half = self.fluid.copy(), self.fluid.copy()
        bottom_half[:, :halfheight], top_half[:, halfheight:] = False, False

        self.Fout = np.zeros((self.Nx, self.Ny, self.Q))
        self.Gout = np.zeros((self.Nx, self.Ny, self.Q))

        self.u_prime = np.zeros((self.Nx, self.Ny, self.D))
        self.rho_f, self.rho_g = np.zeros((self.Nx, self.Ny, 1)), np.zeros((self.Nx, self.Ny, 1))
        self.rho_f[self.fluid] = .5 * np.ones((self.Nx, self.Ny, 1))[self.fluid] + 0.1 * np.random.rand(self.Nx, self.Ny, 1)[self.fluid]
        # self.rho_f[top_half] = .01
        self.rho_g[self.fluid] = .5 * np.ones((self.Nx, self.Ny, 1))[self.fluid] + 0.1 * np.random.rand(self.Nx, self.Ny, 1)[self.fluid]
        # self.rho_g[bottom_half] = .01

        # Electric field
        self.phi = np.zeros((self.Nx, self.Ny))
        self.E = np.zeros_like(self.u_prime)

        self.ini_vel = ini_vel
        self.Re = re  # Higher Re usually needs higher characteristic length
        self.nu = self.ini_vel * self.Ny / self.Re  # U * d / Re
        self.omega_f = 1 / (3 * self.nu + 0.5)
        self.omega_g = self.omega_f + .2
        print("Relaxation times are f: {:.2f} and g: {:.2f}".format(1 / self.omega_f, 1 / self.omega_g))

        # Initialize state
        self.u_prime[self.fluid] = 0 * np.asarray([[self.ini_vel, 0]])
        self.u_prime[:, :halfheight] = 0 * np.asarray([[self.ini_vel, 0]])  # Different velocities
        self.u_prime[self.geometry] = np.asarray([[0, 0]])
        self.Feq = self.equilibrium(self.rho_f, self.u_prime)
        self.Geq = self.equilibrium(self.rho_g, self.u_prime)
        self.Fin, self.Gin = self.Feq.copy(), self.Geq.copy()

        self.animate = animate
        if animate:
            self.rho_data = []
            self.u_data = []
            self.e_data = []

    @staticmethod
    def sum_pops(sites):
        # Sites is an (X,Y,9) array
        if len(sites.shape) == 3:
            s = np.sum(sites, axis=2)
            return s[..., np.newaxis]
        else:
            return np.sum(sites, axis=-1)

    # @profile
    def equilibrium(self, rho, u):
        u_squared = (3 / 2) * (u[:, :, 0] ** 2 + u[:, :, 1] ** 2)
        cu = np.dot(u, self.c.transpose())  # TODO: Replace with explicit formulas
        feq = np.zeros((self.Nx, self.Ny, self.Q))
        for vel in range(self.Q):
            feq[:, :, vel] = self.weights[vel] * rho[:, :, 0] * \
                              (1 + 3 * cu[:, :, vel] + (9 / 2) * (cu[:, :, vel] ** 2) - u_squared)
        if np.any(rho < 0):
            # If we find zero or negative densities, Feq is 0
            print("Negative densities found! Continuing anyway..")
            print(np.where(rho <= 0))
            mask = rho[:, :, 0] <= 0
            feq[mask] = np.zeros(9)
        return feq

    # @profile
    def collide(self):
        # self.Fout = self.Fin - self.omega * (self.Fin - self.Feq)
        fin, w_f, feq = self.Fin, self.omega_f, self.Feq
        gin, w_g, geq = self.Gin, self.omega_g, self.Geq
        self.Fout = ne.evaluate("fin - w_f * (fin - feq)")  # use numexpr
        self.Gout = ne.evaluate("gin - w_g * (gin - geq)")
        # Bounce back
        for vel in range(self.Q):
            self.Fout[self.geometry, vel] = self.Fin[self.geometry, self.c_opposite[vel]]
            self.Gout[self.geometry, vel] = self.Gin[self.geometry, self.c_opposite[vel]]

    # @profile
    def stream(self):
        for vel in range(self.Q):
            self.Fin[:, :, vel] = np.roll(np.roll(self.Fout[:, :, vel], self.c[vel, 0], axis=0), self.c[vel, 1], axis=1)
            self.Gin[:, :, vel] = np.roll(np.roll(self.Gout[:, :, vel], self.c[vel, 0], axis=0), self.c[vel, 1], axis=1)

    def shan_chen_forces(self):
        Fsc, Gsc = np.zeros_like(self.u_prime), np.zeros_like(self.u_prime)
        try:
            psi_f, psi_g = 1 - np.exp(-self.rho_f), 1 - np.exp(-self.rho_g)
        except FloatingPointError:
            divergence = self.rho_f > 1
            print("Density divergence at of order {} at {}".format(np.max(self.rho_f), np.where(self.rho_f > 1)))
            psi_f, psi_g = 1 - np.exp(-self.rho_f), 1 - np.exp(-self.rho_g)
        for x in range(1, self.Nx - 1):
            for y in range(1, self.Ny - 1):
                # if x == self.Nx - 1:
                #     Fsc[x, y, 1] += self.weights[1] * psi_g[x, y + 1] - self.weights[2] * psi_g[x, y - 1]
                #     Gsc[x, y, 1] += self.weights[1] * psi_f[x, y + 1] - self.weights[2] * psi_f[x, y - 1]
                #     Fsc[x, y, 1] += self.weights[7] * psi_g[x - 1, y + 1] - self.weights[8] * psi_g[x - 1, y - 1]
                #     Gsc[x, y, 1] += self.weights[7] * psi_f[x - 1, y + 1] - self.weights[8] * psi_f[x - 1, y - 1]
                #     continue
                # if x == 0:
                #     Fsc[x, y, 1] += self.weights[1] * psi_g[x, y + 1] - self.weights[2] * psi_g[x, y - 1]
                #     Gsc[x, y, 1] += self.weights[1] * psi_f[x, y + 1] - self.weights[2] * psi_f[x, y - 1]
                #     Fsc[x, y, 1] += self.weights[4] * psi_g[x + 1, y + 1] - self.weights[5] * psi_g[x + 1, y - 1]
                #     Gsc[x, y, 1] += self.weights[4] * psi_f[x + 1, y + 1] - self.weights[5] * psi_f[x + 1, y - 1]
                #     continue
                #
                # # Each group represents a particular column of velocities
                # Fsc[x, y, 1] += self.weights[1] * psi_g[x, y + 1] - self.weights[2] * psi_g[x, y - 1]
                # Gsc[x, y, 1] += self.weights[1] * psi_f[x, y + 1] - self.weights[2] * psi_f[x, y - 1]
                #
                # Fsc[x, y, 0] += self.weights[3] * psi_g[x + 1, y] + \
                #                 self.weights[4] * psi_g[x + 1, y + 1] + self.weights[5] * psi_g[x + 1, y - 1]
                # Fsc[x, y, 1] += self.weights[4] * psi_g[x + 1, y + 1] - self.weights[5] * psi_g[x + 1, y - 1]
                # Gsc[x, y, 0] += self.weights[3] * psi_f[x + 1, y] + self.weights[4] * psi_f[x + 1, y + 1] + self.weights[5] * psi_f[x + 1, y - 1]
                # Gsc[x, y, 1] += self.weights[4] * psi_f[x + 1, y + 1] - self.weights[5] * psi_f[x + 1, y - 1]
                #
                # Fsc[x, y, 0] -= self.weights[6] * psi_g[x - 1, y] - \
                #                 self.weights[7] * psi_g[x - 1, y + 1] - self.weights[8] * psi_g[x - 1, y - 1]
                # Fsc[x, y, 1] += self.weights[7] * psi_g[x - 1, y + 1] - self.weights[8] * psi_g[x - 1, y - 1]
                # Gsc[x, y, 0] -= self.weights[6] * psi_f[x - 1, y] - \
                #                 self.weights[7] * psi_f[x - 1, y + 1] - self.weights[8] * psi_f[x - 1, y - 1]
                # Gsc[x, y, 1] += self.weights[7] * psi_f[x - 1, y + 1] - self.weights[8] * psi_f[x - 1, y - 1]
        #
                for vel in range(1, self.Q):
                    Fsc[x, y, 0] += self.weights[vel] * psi_g[x + self.c[vel, 0], y + self.c[vel, 1]] * self.c[vel, 0]
                    Fsc[x, y, 1] += self.weights[vel] * psi_g[x + self.c[vel, 0], y + self.c[vel, 1]] * self.c[vel, 1]
                    Gsc[x, y, 0] += self.weights[vel] * psi_f[x + self.c[vel, 0], y + self.c[vel, 1]] * self.c[vel, 0]
                    Gsc[x, y, 1] += self.weights[vel] * psi_f[x + self.c[vel, 0], y + self.c[vel, 1]] * self.c[vel, 1]

        Fsc *= - self.G * psi_f
        Gsc *= - self.G * psi_g
        return Fsc, Gsc

    # Move Lattice from t to t + 1
    # @profile
    def step(self):
        # TODO: Check outflow and inflow conditions
        # Outflow
        self.Fin[-1, :, self.incoming_left] = self.Fin[-2, :, self.incoming_left]
        self.Gin[-1, :, self.incoming_left] = self.Gin[-2, :, self.incoming_left]

        # Densities and Pseudopotential
        self.rho_f = self.sum_pops(self.Fin)
        self.rho_g = self.sum_pops(self.Gin)

        # Inflow (Zou/He condition) (See J. Latt slides p.71)
        self.u_prime[self.inlet_f] = np.asarray([[self.ini_vel, 0]])
        self.u_prime[self.inlet_g] = np.asarray([[self.ini_vel, 0]])
        self.rho_f[self.inlet_f, 0] = (1 / (1 - self.u_prime[self.inlet_f, 0])) * \
                                        (self.sum_pops(self.Fin[self.inlet_f][:, self.center]) +
                                         2 * self.sum_pops(self.Fin[self.inlet_f][:, self.incoming_right]))
        # self.rho_f[self.inlet_g, 0] = (1 / (1 - np.zeros_like(self.u_prime[self.inlet_f, 0]))) * \
        #                                 (self.sum_pops(self.Fin[self.inlet_g][:, self.center]) +
        #                                  2 * self.sum_pops(self.Fin[self.inlet_g][:, self.incoming_right]))
        self.rho_g[self.inlet_g, 0] = (1 / (1 - self.u_prime[self.inlet_g, 0])) * \
                                        (self.sum_pops(self.Gin[self.inlet_g][:, self.center]) +
                                         2 * self.sum_pops(self.Gin[self.inlet_g][:, self.incoming_right]))
        # self.rho_g[self.inlet_f, 0] = (1 / (1 - np.zeros_like(self.u_prime[self.inlet_f, 0]))) * \
        #                                  (self.sum_pops(self.Gin[self.inlet_f][:, self.center]) +
        #                                   2 * self.sum_pops(self.Gin[self.inlet_f][:, self.incoming_right]))

        # Filter negative values of density
        # TODO: Make velocity = 0 when rho <= 0
        self.rho_g[self.rho_g < 0], self.rho_f[self.rho_f < 0] = 0.001, 0.001
        self.u_prime[self.geometry] = 0

        # Shan Chen Forces using Cython
        psi_f, psi_g = np.squeeze(1 - np.exp(-self.rho_f)), np.squeeze(1 - np.exp(-self.rho_g))
        Fsc, Gsc = performance.cy_shan_chen(psi_f, psi_g, self.c, self.weights, self.Q, self.Nx, self.Ny, self.G)

        # Calculate velocities
        f_dot, g_dot = np.dot(self.Fin, self.c), np.dot(self.Gin, self.c)
        self.u_prime[self.inner_domain] = (self.omega_f * f_dot[self.inner_domain] + self.omega_g * g_dot[self.inner_domain]) / \
                       (self.omega_f * self.rho_f[self.inner_domain] + self.omega_g * self.rho_g[self.inner_domain])

        u_f_eq, u_g_eq = self.u_prime.copy(), self.u_prime.copy()
        u_f_eq[self.inner_domain] = self.u_prime[self.inner_domain] + Fsc[self.inner_domain] / (self.omega_f * self.rho_f[self.inner_domain])
        u_g_eq[self.inner_domain] = self.u_prime[self.inner_domain] + Gsc[self.inner_domain] / (self.omega_g * self.rho_g[self.inner_domain])

        # Equilibrium
        self.Feq = self.equilibrium(self.rho_f, u_f_eq)
        self.Geq = self.equilibrium(self.rho_g, u_g_eq)

        # Finalize Zou/He
        self.Fin[self.inlet_f][:, self.incoming_left] = self.Feq[self.inlet_f][:, self.incoming_left]
        # self.Fin[self.inlet_g][:, self.incoming_left] = self.Fin[self.inlet_g][:, self.incoming_right] \
        #                                                 + self.Feq[self.inlet_g][:, self.incoming_left] \
        #                                                 - self.Fin[self.inlet_g][:, self.incoming_right]
        self.Gin[self.inlet_g][:, self.incoming_left] = self.Geq[self.inlet_g][:, self.incoming_left]
        # self.Gin[self.inlet_f][:, self.incoming_left] = self.Gin[self.inlet_f][:, self.incoming_right] \
        #                                                 + self.Geq[self.inlet_f][:, self.incoming_left] \
        #                                                 - self.Gin[self.inlet_f][:, self.incoming_right]

        u2 = np.sqrt(self.u_prime[:, :, 0] ** 2 + self.u_prime[:, :, 1] ** 2)
        if np.any(u2 > .57):
            print("Warning, velocity divergence immminent")
        self.collide()
        self.t += 1
        self.stream()

        self.poisson()

        if self.animate:
            self.rho_data.append(np.squeeze(self.rho_f).transpose().copy())
            self.u_data.append(u2.transpose())
            self.e_data.append(self.E[:, :, 0].transpose() ** 2 + self.E[:, :, 1].transpose() ** 2)
            if self.t % 100 == 0:
                # print(np.mean(self.rho_f))
                plt.cla()
                plt.imshow(np.squeeze(self.rho_f).transpose())
                # x, y = np.linspace(0, self.Nx, self.Nx), np.linspace(0, self.Ny, self.Ny)
                # X, Y = np.meshgrid(x, y)
                # Ex, Ey = self.E[:, :, 0].transpose(), self.E[:, :, 1].transpose()
                # plt.quiver(X, Y, Ex, Ey)
                plt.axis('off')
                plt.show()

    def poisson(self):
        tol = .001
        dx, dy = .01, .01
        x, y = np.linspace(0, self.Nx, self.Nx), np.linspace(0, self.Ny, self.Ny)
        X, Y = np.meshgrid(x, y)
        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        b = np.squeeze(self.rho_f) - np.squeeze(self.rho_g)  # source term
        phi = np.zeros_like(b)

        # stop = False

        for i in range(100):
            phi_prev = phi.copy()
            phi[1:-1, 1: -1] = (dx ** 2 * (phi_prev[1:-1, 0:-2] + phi_prev[1:-1, 2:]) + dy ** 2 * (
                    phi_prev[0:-2, 1:-1] + phi_prev[2:, 1:-1]) - b[1:-1, 1:-1] * dx ** 2 * dy ** 2) / (2 * (dx ** 2 + dy ** 2))
            phi[0, :], phi[-1, :] = phi[1, :], phi[-2, :]
            phi[:, 0], phi[:, -1] = phi[:, 1], phi[:, -2]

            diff = phi - phi_prev
            stop = np.linalg.norm(diff) < tol

        self.E[:-1, :, 0], self.E[:, :-1, 1] = np.diff(phi, axis=0), np.diff(phi, axis=1)  # does not set last element

    def make_animation(self):
        print("########################################################")
        if self.animate:
            print("# Animation has {} frames.".format(len(self.rho_data)))
        else:
            print("No postprocessing requested. Exiting..")
            return 1

        print("########################################################")
        print("# Making animation, please wait..                      #")
        fig = plt.figure()
        # plt.legend()
        # anim_length, interval = len(self.rho_data), 1 / 30
        anim_length, interval = len(self.u_data), 1 / 30
        min_vel, max_vel = 0, 0.1

        def update(t):
            plt.clf()
            plt.cla()
            plt.subplot(3, 1, 1)
            plt.axis('off')
            im1 = plt.imshow(self.rho_data[t], animated=True)
            plt.title("Rho")
            plt.subplot(3, 1, 2)
            plt.axis('off')
            im2 = plt.imshow(self.u_data[t], animated=True, cmap=cm.coolwarm)
            plt.title("Champs de vitesses")
            plt.subplot(3, 1, 3)
            plt.title("Champs electrique")
            im3 = plt.imshow(self.e_data[t], animated=True, cmap=cm.coolwarm)

            return [im1, im3]

        # anim = animation.ArtistAnimation(fig, ims, interval=interval,     blit=False)E
        anim = animation.FuncAnimation(fig, update, frames=anim_length, interval=interval, blit=True)
        str_name = "output/animation - Re: {} - Visc: {} - {}.mp4".format(self.Re, self.nu, datetime.now())
        print("# Saving video..                                       #")
        writer = animation.FFMpegWriter(fps=30)
        anim.save(str_name, writer=writer)
        print("# Animation saved in: \"{}\"".format(str_name))
        print("########################################################")

    def run(self, maxiter):
        start = time.time()
        for iteration in progressbar(range(maxiter)):
            self.step()
        self.runtime = time.time() - start

    def show_performance(self):
        MLUPS = self.Nx * self.Ny * self.t / (self.runtime * 1e6)
        print("########################################################")
        print("# Average performance is : {:.4f} MLUPS                #".format(MLUPS))
        print("########################################################")
