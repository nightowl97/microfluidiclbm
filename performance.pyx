#cython: language_level=3
#cython: boundscheck=False
import cython
import numpy as np


cpdef cy_shan_chen(double [:, ::1] psi_f, double [:, ::1] psi_g, int [:, ::1] c, double [::1] w, int q, int nx, int ny, int g):
    """
    Python exposed function
    :param psi_f: Pseudopotential for the first component
    :param psi_g: Pseudopotential for the second component
    :param c: Lattice velocities
    :param q: Number of lattice velocities
    :param w: Weights
    :param nx: Number of lattice sites in X direction
    :param ny: Number of lattice sites in Y direction
    :param g: Interaction strength
    :return: Tuple of arrays containing the Shan-Chen Forces of each component 
    """
    fsc_res = np.zeros((nx, ny, 2))
    gsc_res = np.zeros((nx, ny, 2))
    cdef:
        double[:, :, ::1] fsc = fsc_res
        double[:, :, ::1] gsc = gsc_res
        int x = 0
        int y = 0
        int vel = 0

    for x in range(nx):
        for y in range(1, ny - 1):
            for vel in range(1, q):
                if x == 0 and c[vel, 0] < 0: continue
                if x == nx - 1 and c[vel, 0] > 0: continue
                fsc[x, y, 0] += - g * psi_f[x, y] * w[vel] * psi_g[x + c[vel, 0], y + c[vel, 1]] * c[vel, 0]
                fsc[x, y, 1] += - g * psi_f[x, y] * w[vel] * psi_g[x + c[vel, 0], y + c[vel, 1]] * c[vel, 1]

                gsc[x, y, 0] += - g * psi_g[x, y] * w[vel] * psi_f[x + c[vel, 0], y + c[vel, 1]] * c[vel, 0]
                gsc[x, y, 1] += - g * psi_g[x, y] * w[vel] * psi_f[x + c[vel, 0], y + c[vel, 1]] * c[vel, 1]

    return fsc_res, gsc_res