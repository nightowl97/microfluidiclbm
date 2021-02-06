import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt
from numpy.linalg import norm
from PIL import Image
import matplotlib.animation as animation
from datetime import datetime


# TODO: Colors to define site type (Inlet, Outlet, Bounce-back, Fluid etc.)
# TODO: Optimize dot product in equilibrium function
# TODO: Allow cavity with moving lid example


class D2Q9Lattice:
    D, Q = 2, 9
    # Lattice Velocities
    c_list = [(x, y) for x in [0, 1, -1] for y in [0, 1, -1]]
    c = np.array(c_list)

    # Incoming columns
    incoming_right = np.arange(9)[np.array([vel[0] == -1 for vel in c])]
    incoming_left = np.arange(9)[np.array([vel[0] == 1 for vel in c])]
    center = np.arange(9)[np.array([vel[0] == 0 for vel in c])]

    def __init__(self, geometry_image, ini_vel, re, animate=False):
        # Initialize object and get geometry
        self.t = 0
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

        self.fluid_geometry = ~self.geometry

        # self.Fin = np.zeros((self.Nx, self.Ny, self.Q))
        self.Fout = np.zeros((self.Nx, self.Ny, self.Q))
        # self.Feq = np.zeros((self.Nx, self.Ny, self.Q))

        self.u = np.zeros((self.Nx, self.Ny, self.D))
        self.rho = np.ones((self.Nx, self.Ny))

        self.ini_vel = ini_vel
        self.Re = re
        self.nu = self.ini_vel * self.Ny / self.Re  # U * r / Re
        self.omega = 1 / (3 * self.nu + 0.5)

        # Initialize state
        self.inlet = self.fluid_geometry[0]  # Inlet to the left at x = 0
        self.outlet = self.fluid_geometry[-1]  # Outlet to the right
        # self.rho[self.fluid_geometry] = 1
        self.u[self.fluid_geometry] = np.asarray([[self.ini_vel, 0]])
        self.Feq = self.equilibrium(self.rho, self.u)
        self.Fin = self.Feq.copy()

        self.animate = animate
        if animate:
            self.rho_data = []
            self.u_data = []

    @staticmethod
    def sum_pops(sites):
        # Sites is an (X,Y,9) array
        if len(sites.shape) == 3:
            return np.sum(sites, axis=2)
        else:
            return np.sum(sites, axis=0)

    def equilibrium(self, rho, u):
        u_squared = (3 / 2) * (u[:, :, 0] ** 2 + u[:, :, 1] ** 2)
        cu = np.dot(u, self.c.transpose())
        feq = np.zeros((self.Nx, self.Ny, self.Q))
        for vel in range(self.Q):
            feq[:, :, vel] = self.weights[vel] * rho * \
                             (1 + 3 * cu[:, :, vel] + (9 / 2) * (cu[:, :, vel] ** 2) - u_squared)
        return feq

    def enf_bounds(self):
        # Outflow
        self.Fin[-1, :, self.incoming_right] = self.Fin[-2, :, self.incoming_right]

        # Calculate macroscopic quantities
        self.rho = self.sum_pops(self.Fin)
        rho_expanded = np.expand_dims(self.rho, axis=-1)
        self.u = (1 / rho_expanded) * np.dot(self.Fin, self.c)

        # Inflow (Zou/He condition) (See J. Latt slides p.71)
        self.rho[0, :] = (1 / (1 - self.u[0, :, 0])) * \
                         (self.sum_pops(self.Fin[0, :, self.center]) +
                          2 * self.sum_pops(self.Fin[0, :, self.incoming_right]))
        self.Feq = self.equilibrium(self.rho, self.u)  # TODO: Calculate inlet equilibrium only
        self.Fin[0, :, self.incoming_left] = self.Feq[0, :, self.incoming_left] + \
                                             (self.Fin[0, :, self.incoming_right] - self.Feq[0, :, self.incoming_right])

    def collide(self):
        # self.Fout = self.Fin - self.omega * (self.Fin - self.Feq)
        # use numexpr
        fin, w, feq = self.Fin, self.omega, self.Feq
        self.Fout = ne.evaluate("fin - w * (fin - feq)")
        # self.Fout[self.fluid_geometry] = self.Fin[self.fluid_geometry] - self.omega * \
        #                                  (self.Fin[self.fluid_geometry] - self.Feq[self.fluid_geometry])
        # Bounce back
        for vel in range(self.Q):
            self.Fout[self.geometry, vel] = self.Fin[self.geometry, self.c_opposite[vel]]
            # g, vel, cop = self.geometry, vel, self.c_opposite
            # self.Fout[self.geometry, vel] = ne.evaluate("fin[g, cop[vel]]")

    def stream(self):
        # TODO: change periodic streaming
        for vel in range(self.Q):
            self.Fin[:, :, vel] = np.roll(np.roll(self.Fout[:, :, vel], self.c[vel, 0], axis=0), self.c[vel, 1], axis=1)

    # Move Lattice from t to t + 1
    def step(self):
        self.enf_bounds()

        self.collide()
        self.t += 1
        self.stream()

        if self.animate:
            u2 = np.sqrt(self.u[:, :, 0] ** 2 + self.u[:, :, 1] ** 2)
            self.u_data.append(u2.transpose())
            # self.rho_data.append(self.rho.transpose())

    def make_animation(self):
        if self.animate:
            print("\nAnimation has {} frames.".format(len(self.u_data)))
        else:
            print("No postprocessing requested. Exiting..")
            return 1

        print("Making animation, please wait..")
        fig = plt.figure()
        anim_length, interval = len(self.u_data), 1 / 30
        min_vel, max_vel = 0, 0.3

        def update(t):
            plt.clf()
            im = plt.imshow(self.u_data[t], animated=True, interpolation='none', vmin=min_vel, vmax=max_vel)
            return [im]

        # anim = animation.ArtistAnimation(fig, ims, interval=interval, blit=False)E
        anim = animation.FuncAnimation(fig, update, frames=anim_length, interval=interval, blit=True)
        str_name = "output/animation - {}.mp4".format(datetime.now())
        writer = animation.FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
        anim.save(str_name, writer=writer)
        print("animation saved as: \"{}\"".format(str_name))
