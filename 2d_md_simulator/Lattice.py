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
from Chain import ChainOrientedElasticNonUniform


class SquareLattice(object):
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

    def __init__(self, N_width, N_height, M_width, M_height, gamma, kT, lattice0type="unit_square"):
        """
        consider the lattice as a square lattice of L shapes
        the hanging particles at the top and right are not used to calculate anything
        M is inside unit cell
        N is  number of unit cells

        note that this lattice has a redundancy
        the linking particles are listed two times in self.r and self.f
        this is to ensure easy slicing
        care should be taken though that these follow the exact same dynamics
        forces on the one should be always equal to forces on the other
        """

        self.D = 2

        self.gamma = gamma
        self.kT = kT
        self.diffusion = kT/gamma

        self.sqrt_2diffusion = np.sqrt(2*self.diffusion)
        self.sigma = np.sqrt(2*gamma*kT)
        self.lattice0type = lattice0type

        self.N_width = N_width
        self.N_height = N_height
        self.M_width = M_width
        self.M_height = M_height
        self.M_link1 = int(M_width/2.)
        # thought about this and tested it, looks awkward but is correct
        self.M_link2 = int((M_height-1)/2.)+M_width
        self.r = 1.0*np.zeros((N_width, N_height, M_width+M_height, self.D))
        self.f = 1.0*np.zeros((N_width, N_height, M_width+M_height, self.D))
        self.psi = 1.0*np.zeros((N_width, N_height, M_width+M_height))
        self.initialize_positions()

    def initialize_positions(self):
        """
        initialize the positions r in a chain of length L0 on x-axis or Circle
        depending on chain0type
        """

        if self.lattice0type == "unit_square":

            N_width = self.N_width
            N_height = self.N_height
            M_width = self.M_width
            M_height = self.M_height
            M_link1 = self.M_link1
            M_link2 = self.M_link2
            D = self.D
            N_unit_cell = M_width+M_height
            r = 1.0*np.zeros((N_width, N_height, N_unit_cell, D))

            # there must be a simpler way, but I broke my head to come up with this way to get to linked particles at (0,0)
            unit_cell = 1.0*np.zeros((N_unit_cell, D))
            unit_cell[:M_link1, 0] = np.linspace(-M_link1, -1, M_link1)
            unit_cell[M_link1, 0] = 0
            unit_cell[M_link1+1:M_width, 0] = np.linspace(1, M_width-M_link1-1, M_width-M_link1-1)

            unit_cell[M_width:M_link2, 1] = np.linspace(-(M_link2-M_width), -1, M_link2-M_width)
            unit_cell[M_link2, 1] = 0
            unit_cell[M_link2+1:N_unit_cell,
                      1] = np.linspace(1, M_height-(M_link2-M_width)-1, M_height-(M_link2-M_width)-1)

            for i in range(0, N_width):
                for j in range(0, N_height):
                    shift = (i*M_width, j*M_height)
                    r[i, j, :, 0] = shift[0]+unit_cell[:, 0]
                    r[i, j, :, 1] = shift[1]+unit_cell[:, 1]

            self.r = r


class SquareLatticeElastic(SquareLattice):
    """
    this one has elastic type advance time
    """

    def __init__(self,  N_width, N_height, M_width, M_height, gamma, kT, k, k_theta, k_theta_link, d0, lattice0type, sigma_theta0):

        super(SquareLatticeElastic, self).__init__(N_width, N_height, M_width,
                                                   M_height, gamma, kT, lattice0type)

        self.internal_energy = 0
        self.k = k
        self.k_theta = k_theta
        self.k_theta_link = k_theta_link
        self.sigma_theta0 = sigma_theta0
        self.theta_link_0 = np.pi/2

        self.horizontal_chain_list = [ChainOrientedElasticNonUniform(N=N_width*M_width, gamma=gamma, kT=kT,
                                                                     chain0type="straight_stretched",
                                                                     Larc0=N_width*M_width-1,
                                                                     k=k, k_theta=k_theta, d0=d0, sigma_theta0=sigma_theta0) for i in range(0, N_height)]

        self.vertical_chain_list = [ChainOrientedElasticNonUniform(N=N_height*M_height, gamma=gamma, kT=kT,
                                                                   chain0type="straight_stretched",
                                                                   Larc0=N_height*M_height-1,
                                                                   k=k, k_theta=k_theta, d0=d0, sigma_theta0=sigma_theta0) for i in range(0, N_width)]
        self.set_linker_k_theta_in_chains()
        self.update_chains_state_from_r()
        self.update_psi_from_chains_state()
        self.update_f_from_chains_state()
        self.update_f_linker_bond_from_r()

    def set_linker_k_theta_in_chains(self):

        for i, horizontal_chain_i in enumerate(self.horizontal_chain_list):
            horizontal_chain_i.k_theta_array[int(
                self.M_width/2.)-1::self.M_width] = self.k_theta_link

        for i, vertical_chain_i in enumerate(self.vertical_chain_list):
            vertical_chain_i.k_theta_array[int(
                self.M_height/2.)-1::self.M_height] = self.k_theta_link

    def update_chains_state_from_r(self):
        """
        updates the position, distance, theta and force between particles in a linked subchain of the network
        based on the position of all the particles in the network r

        """
        for i, horizontal_chain_i in enumerate(self.horizontal_chain_list):
            r_chain_i = self.r[:, i, :self.M_width]
            r_chain_i_flatten = np.reshape(r_chain_i, (horizontal_chain_i.N, self.D))
            horizontal_chain_i.r = r_chain_i_flatten
            horizontal_chain_i.update_psi_d_and_theta_from_r()
            horizontal_chain_i.update_internal_forces()

        for i, vertical_chain_i in enumerate(self.vertical_chain_list):
            r_chain_i = self.r[i, :, self.M_width:]
            r_chain_i_flatten = np.reshape(r_chain_i, (vertical_chain_i.N, self.D))
            vertical_chain_i.r = r_chain_i_flatten
            vertical_chain_i.update_psi_d_and_theta_from_r()
            vertical_chain_i.update_internal_forces()

    def update_f_from_chains_state(self):

        #self.f = 1.0*np.zeros((self.N_width, self.N_height, self.M_width+self.M_height, self.D))

        for i, horizontal_chain_i in enumerate(self.horizontal_chain_list):
            f_chain_i = horizontal_chain_i.f
            f_chain_i_shaped = np.reshape(f_chain_i, (self.N_width, self.M_width, self.D))
            self.f[:, i, :self.M_width] = f_chain_i_shaped

        for i, vertical_chain_i in enumerate(self.vertical_chain_list):
            f_chain_i = vertical_chain_i.f
            f_chain_i_shaped = np.reshape(f_chain_i, (self.N_height, self.M_height, self.D))
            self.f[i, :, self.M_width:] = f_chain_i_shaped

        # take care the linking particles experience the same force
        f_linkers_sum = self.f[:, :, self.M_link1] + \
            self.f[:, :, self.M_link2]
        self.f[:, :, self.M_link1] = f_linkers_sum
        self.f[:, :, self.M_link2] = f_linkers_sum

    def update_f_linker_bond_from_r(self):
        """
        always update this on eafter updating f from chain
        make sure the linkers keep a 90deg angle
        this can be done only on one combination of three particle next to the linker
        as the straight chain condition ensure that they will cross 90deg
        we choose the
        """
        theta_link = self.psi[:, :, self.M_link2]-self.psi[:, :, self.M_link1]
        d_theta_link = theta_link - self.theta_link_0

        r_three_chain = self.r[:, :, [self.M_link2+1, self.M_link1, self.M_link1+1]]
        dr = r_three_chain[:, :, 1:]-r_three_chain[:, :, :-1]
        d_square = np.einsum('ijkl,ijkl->ijk', dr, dr)
        drx = dr[:, :, :, 0]
        dry = dr[:, :, :, 1]
        dr_rot_90 = np.zeros((self.N_width, self.N_height, 2, self.D))
        dr_rot_90[:, :, :, 0] = -dry  # rotated 90 degress counterclockwise
        dr_rot_90[:, :, :, 1] = drx

        fleft = self.k_theta_link*(
            d_theta_link[:, :, None]*dr_rot_90[:, :, 0]/d_square[:, :, 0, None])
        fright = self.k_theta_link*(
            d_theta_link[:, :, None]*dr_rot_90[:, :, 1]/d_square[:, :, 1, None])
        self.f[:, :, self.M_link2+1] += fleft
        self.f[:, :, self.M_link1+1] += fright
        self.f[:, :, self.M_link1] += fleft+fright
        self.f[:, :, self.M_link2] += fleft+fright

    def update_psi_from_chains_state(self):

        for i, horizontal_chain_i in enumerate(self.horizontal_chain_list):
            psi_chain_i = horizontal_chain_i.psi
            psi_chain_i_shaped = np.reshape(psi_chain_i, (self.N_width, self.M_width))
            self.psi[:, i, :self.M_width] = psi_chain_i_shaped

        for i, vertical_chain_i in enumerate(self.vertical_chain_list):
            psi_chain_i = vertical_chain_i.psi
            psi_chain_i_shaped = np.reshape(psi_chain_i, (self.N_height, self.M_height))
            self.psi[i, :, self.M_width:] = psi_chain_i_shaped

    def integrate(self, dt, dtsqrt):
        """
        shifted to after to have indeed the same time f and r and psi

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

        xi = np.random.normal(0, 1, (self.N_width, self.N_height,
                                     self.M_width+self.M_height, self.D))

        # ensure the double linking particles experience the same stochastic noice
        xi[:, :, self.M_link1] = xi[:, :, self.M_link2]

        self.r += self.f*dt/self.gamma + self.sqrt_2diffusion*xi*dtsqrt

        self.update_chains_state_from_r()
        self.update_psi_from_chains_state()

        self.update_f_from_chains_state()
        self.update_f_linker_bond_from_r()
