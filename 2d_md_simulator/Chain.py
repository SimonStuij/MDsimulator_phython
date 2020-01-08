"""
Created on Fri Oct  12 20:10:00 2018

this is an MD model
this means it should have an advance time function
that specifies based on the elements (particles, force sources (traps))
how the particles will advance if time is being run

the actual time advancing is being done in a Simulation class

"""
import numpy as np
import sys


class Chain(object):
    """
    a thermal chain of bonded particles that
    has identical brownian particles

    inputs:
    N -  #particles
    gamma -  drag coefficient
    kT - temprature
    Larc0 - initial arc length
    chain0type - choose from:
        -straight_stretched
        -triangle_stretched
        -circle
        -random zigzag
    """

    def __init__(self, N, gamma, kT, chain0type, Larc0):
        self.D = 2
        self.N = N

        self.gamma = gamma
        self.kT = kT
        self.diffusion = kT/gamma

        self.sqrt_2diffusion = np.sqrt(2*self.diffusion)
        self.sigma = np.sqrt(2*gamma*kT)
        self.chain0type = chain0type
        self.Larc0 = Larc0
        self.r = 1.0*np.zeros((N, self.D))
        self.f = 1.0*np.zeros((N, self.D))
        self.theta = 1.0*np.zeros(N-2)
        self.d = 1.0*np.zeros(N-1)
        self.phi = 1.0*np.zeros(N-1)

        self.chain0type = chain0type
        self.Larc0 = Larc0
        self.initialize_positions()
        self.update_d_and_theta_from_r()

    def initialize_positions(self):
        """
        initialize the positions r in a chain of length L0 on x-axis or Circle
        depending on chain0type
        """

        if self.chain0type == "straight_stretched":
            self.r[:, 0] = np.linspace(0, self.Larc0, self.N)

        elif self.chain0type == "triangle_stretched":
            sys.exit("triangle_stretched initial conditions not yet implemented")
            """
            if self.N % 2 == 0:
                self.r[:self.N/2, 0] = np.linspace(0, self.Larc0/2, self.N/2)
                # vertical part
                self.r[self.N/2:, 0] = np.linspace(self.Larc0/2, self.N/2)
            self.r[0] = 1.0*np.array([0, 0])
            self.r[1] = 1.0*np.array([1, 0])
            self.r[2] = 1.0*np.array([1, 1])
            """

        elif self.chain0type == "circle":
            """
            this only makes sense if d0=1
            """
            print("circle initial conditions is not correct implemented," +
                  "Larc0 is interpreted as end to end distance and d0=1 is assumed")
            from scipy.optimize import root
            L_max = self.N-1  # beyond this the string can not be circular
            # we assume that this is the arclength of the chain
            if self.Larc0 < L_max:  # otherwise it is in an stretched state
                def root_fun(x): return(x*np.sin(L_max/(x*2))-self.Larc0/2)
                """this is the trancendental function that gives
                the radius of the circle that connects points L_trap apart with
                 an arclength between them of L_max
                """

                R = abs(root(root_fun, x0=L_max)['x'])
                if self.Larc0 > 2*L_max/np.pi:
                    centre = np.array([self.Larc0/2, -np.sqrt(R**2-self.Larc0**2/4)])
                else:
                    centre = np.array([self.Larc0/2, +np.sqrt(R**2-self.Larc0**2/4)])
                theta = L_max/R
                phipart = np.linspace(np.pi/2+theta/2, np.pi/2-theta/2, self.N)
                [xpart, ypart] = np.array(
                    [R*np.cos(phipart)+centre[0], R*np.sin(phipart)+centre[1]])
                self.r[:, 0] = xpart
                self.r[:, 1] = ypart
            else:
                self.r[:, 0] = np.linspace(0, self.Larc0, self.N)

        elif self.chain0type == "random_zigzag":
            sys.exit("random_zigzag initial conditions not yet implemented")

        else:
            sys.exit("chain0type is not in possible chain0type")

    def update_d_and_theta_from_r(self):
        """
        this overcomes the problem of the string orientation by
        making sure that each pair of dr vectors of which you calculate the
        bending angle the pair is rotated such that the first dr is on the x axis
        the calculation becomes much more expensive unfortanately
        be aware that this theta does not remember full rotations and if a too big
        value is reached errors can occur
        one could put a limit the angle can not be bigger than such that the particle touch
        positive theta is curvature to the left
        """
        dr = self.r[1:, :] - self.r[:-1, :]
        drx = dr[:, 0]
        dry = dr[:, 1]
        phi = np.arctan2(dry, drx)
        self.phi = phi

        for i, (phi_i, dr_i) in enumerate(zip(phi[0:-1], dr[1::])):
            c, s = np.cos(-phi_i), np.sin(-phi_i)
            dr_i_rot_x, dr_i_rot_y = c*dr_i[0]-s*dr_i[1], s*dr_i[0]+c*dr_i[1]
            self.theta[i] = np.arctan2(dr_i_rot_y, dr_i_rot_x)

        self.d = np.sqrt(np.einsum('ij,ij->i', dr, dr))
        # equivalent to np.linalg.norm(dr, axis=1)
        # but this doesn't run on obelix numpy version


class ChainElastic(Chain):
    """
    this one has elastic type advance time
    """

    def __init__(self, N, gamma, kT, chain0type, Larc0, k, k_theta, d0, sigma_theta0):
        super(ChainElastic, self).__init__(N, gamma, kT, chain0type, Larc0)

        self.internal_energy = 0
        self.k = k
        self.k_theta = k_theta
        self.d0 = 1.0*d0*np.ones(self.N-1)
        if sigma_theta0 == 0:
            self.theta0 = 1.0*np.zeros(self.N-2)
        elif sigma_theta0 > 0:
            self.theta0 = np.random.normal(0, sigma_theta0, self.N-2)

        self.update_internal_forces()

    def update_internal_forces(self):

        self.f = 1.0*np.zeros((self.N, self.D))

        dr = self.r[1:, :] - self.r[: -1, :]

        # the ,None is to be able to multiply a list of factors
        # with an equally long list of coordinates
        if self.k > 0:
            self.f[: -1] += self.k*dr*(1-(self.d0/self.d)[:, None])
            self.f[1:] -= self.k*dr*(1-(self.d0/self.d)[:, None])
        # calculate internal bending forces
        if self.k_theta > 0:
            drx = dr[:, 0]
            dry = dr[:, 1]
            dr_rot_90 = np.zeros((self.N-1, self.D))
            dr_rot_90[:, 0] = -dry  # rotated 90 degress counterclockwise
            dr_rot_90[:, 1] = drx

            dtheta = self.theta - self.theta0

            d_square = self.d**2
            fleft = -self.k_theta*(
                dtheta[:, None]*dr_rot_90[:-1]/d_square[:-1, None])
            fright = -self.k_theta*(
                dtheta[:, None]*dr_rot_90[1:]/d_square[1:, None])

            self.f[:-2] += fleft
            self.f[2:] += fright
            self.f[1:-1] -= fleft+fright

    def integrate(self, dt, dtsqrt):
        """
        had to think a bit about sequence of update, updata f before or after
        updating of r, initially had after becuae then with saving you the data at
        a certain time you get the force that you would caluclate from the
        positions of the particles at that timeself.
        However now i have it before which is robuster I just need to remember in saving
        that I save the force with the previous times
        So when I save, i first save r and t than move time and then save f at the same t spot
        Also first the trap was moved after updating the particle positions
        but then the trapped particles are not at their rigth position at the time
        when the trap moves
        """
        self.update_d_and_theta_from_r()
        self.update_internal_forces()

        xi = np.random.normal(0, 1, (self.N, self.D))
        self.r += self.f*dt/self.gamma + self.sqrt_2diffusion*xi*dtsqrt


class ChainPlastic(ChainElastic):
    """
    plastic behavior implemented by immediate change of eqilibrium bond angle theta0
    essentially stick slip model with zero dynamic friction
    and  also assuming the slipping time is much faster than the relevant brownian time scales
    """

    def __init__(self, N, gamma, kT, chain0type, Larc0, k, k_theta, d0, sigma_theta0, theta0_crit):

        super(ChainPlastic, self).__init__(N, gamma, kT,
                                           chain0type, Larc0, k, k_theta, d0, sigma_theta0)

        self.theta0_crit = theta0_crit

    def check_for_plastic_slip(self):

        for i, (thetai, theta0i) in enumerate(zip(self.theta, self.theta0)):
            if abs(thetai - theta0i) > self.theta0_crit:
                # slip
                self.theta0[i] = thetai

    def integrate(self, dt, dtsqrt):

        self.update_d_and_theta_from_r()
        self.check_for_plastic_slip()
        self.update_internal_forces()

        xi = np.random.normal(0, 1, (self.N, self.D))
        self.r += self.f*dt/self.gamma + self.sqrt_2diffusion*xi*dtsqrt


class ChainElasticIncompressible(ChainElastic):

    def __init__(self, N, gamma, kT, chain0type, Larc0, k, k_theta, d0, sigma_theta0, theta0_crit):

        super(ChainElasticIncompressible, self).__init__(N, gamma, kT,
                                                         chain0type, Larc0, k, k_theta, d0, sigma_theta0)

        self.theta0_crit = theta0_crit

    def find_lagrangian_multipliers(self):

        dr = self.r[1:, :] - self.r[:-1, :]
        drx = dr[:, 0]
        dry = dr[:, 1]
        phi = np.arctan2(dry, drx)

        dr0 = np.array()
        for i, phi_i in enumerate(self.phi):
            c, s = np.cos(-phi_i), np.sin(-phi_i)
            dr_i_rot_x, dr_i_rot_y = c*dr_i[0]-s*dr_i[1], s*dr_i[0]+c*dr_i[1]
            self.theta[i] = np.arctan2(dr_i_rot_y, dr_i_rot_x)

        self.d = np.sqrt(np.einsum('ij,ij->i', dr, dr))

    def add_lagrangian_bond_forces(self):

        self.f_lagrange = 1.0*np.zeros((self.N-1, self.D))

    def integrate(self, dt, dtsqrt):

        self.update_d_and_theta_from_r()
        self.update_internal_forces()

        xi = np.random.normal(0, 1, (self.N, self.D))
        self.r_hat += self.f*dt/self.gamma + self.sqrt_2diffusion*xi*dtsqrt

        self.r[:-1] += -2/self.gamma*(self.lambda_bond*(self.r[1:]-self.r[:-1]))/self.d0
        self.r[1:] += -2/self.gamma*(self.lambda_bond*(self.r[:-1]-self.r[1:]))/self.d0


class ChainOrientedElastic(ChainElastic):
    """
    orientation (psi)
    this one doesn't calculate it's forces based deviation of theta from theta_0
    but on deviaion of phi_i+1-phi_i from theta_0
    each particle is thus dressed with a direction here phi_i
    the main reason for implementing like this is to simulate lattices

    it assume orientation is always "energy minimized", meaning that the orientation
    of a particle is always exactly half of the two bond directions adjecent to it
    the end particles have their orientation in direction of the bond
    """

    def __init__(self, N, gamma, kT, chain0type, Larc0, k, k_theta, d0, sigma_theta0):

        super(ChainOrientedElastic, self).__init__(N, gamma, kT,
                                                   chain0type, Larc0, k, k_theta, d0, sigma_theta0)

        self.psi = 1.0*np.zeros(N)
        self.update_psi_from_phi_and_theta()

    def update_psi_from_phi_and_theta(self):

        self.psi[0] = self.phi[0]
        self.psi[-1] = self.phi[-1]
        self.psi[1:-1] = self.phi[:-1]+self.theta/2

        # correct to keep angle within -pi pi range
        self.psi[self.psi < -np.pi] += 2*np.pi
        self.psi[self.psi > np.pi] -= 2*np.pi

    def update_psi_d_and_theta_from_r(self):
        """
        also update orientation when calling this one
        """
        super(ChainOrientedElastic, self).update_d_and_theta_from_r()
        self.update_psi_from_phi_and_theta()


class ChainOrientedElasticNonUniform(ChainOrientedElastic):
    """
    this chain does not need to have the same bending rigidities for each bond
    though when initiated it assumes each rigidity is identical
    one needs to explicitlys set specific rigidities if you want them different
    """

    def __init__(self, N, gamma, kT, chain0type, Larc0, k, k_theta, d0, sigma_theta0):

        self.k_theta_array = k_theta*np.ones(N-2)
        super(ChainOrientedElasticNonUniform, self).__init__(N, gamma, kT,
                                                             chain0type, Larc0, k, k_theta, d0, sigma_theta0)

    def update_internal_forces(self):

        self.f = 1.0*np.zeros((self.N, self.D))

        dr = self.r[1:, :] - self.r[: -1, :]

        # the ,None is to be able to multiply a list of factors
        # with an equally long list of coordinates
        if self.k > 0:
            self.f[: -1] += self.k*dr*(1-(self.d0/self.d)[:, None])
            self.f[1:] -= self.k*dr*(1-(self.d0/self.d)[:, None])
        # calculate internal bending forces
        if self.k_theta > 0:
            drx = dr[:, 0]
            dry = dr[:, 1]
            dr_rot_90 = np.zeros((self.N-1, self.D))
            dr_rot_90[:, 0] = -dry  # rotated 90 degress counterclockwise
            dr_rot_90[:, 1] = drx

            dtheta = self.theta - self.theta0

            d_square = self.d**2
            fleft = -self.k_theta_array[:, None]*(
                dtheta[:, None]*dr_rot_90[:-1]/d_square[:-1, None])
            fright = -self.k_theta_array[:, None]*(
                dtheta[:, None]*dr_rot_90[1:]/d_square[1:, None])

            self.f[:-2] += fleft
            self.f[2:] += fright
            self.f[1:-1] -= fleft+fright
