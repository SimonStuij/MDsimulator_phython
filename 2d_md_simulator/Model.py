"""
Created on Fri Oct  12 20:10:00 2018


traps are always on the x axis,
one can set the initial distance between them
L0_trap and the way the mobile trap moves


"""

from Chain import ChainElastic, ChainPlastic
from Lattice import SquareLatticeElastic
from Trap import Trap
import numpy as np


class TrappedChainElastic(object):
    """
    trap is kept on x axis and compressing by default
    though you can try inputting negative amplitudes to stretch
    """

    def __init__(self, chain_elastic_input, L0_trap, mobile_trap_parameters):

        self.name = "trapped_chain_elastic"
        self.time = 0
        self.chain = ChainElastic(**chain_elastic_input)

        self.static_trap = Trap(1.0*np.array([0, 0]), [0])
        self.mobile_trap = Trap(1.0*np.array([L0_trap, 0]), [self.chain.N-1])

        self.mobile_trap.make_mobile(direction=np.array([-1, 0]), **mobile_trap_parameters)
        # self.mobile_trap.movetrap(self.time)

        self.model_parameters = {"model_name": self.name,
                                 "L0_trap": L0_trap}
        self.model_parameters.update(chain_elastic_input)
        self.model_parameters.update(mobile_trap_parameters)

    def advance_time(self, dt, dtsqrt):

        self.time += dt
        self.chain.integrate(dt, dtsqrt)

        # the trapped particles are fixed by the trap positions:
        self.mobile_trap.movetrap(self.time)
        self.chain.r[0] = self.static_trap.position
        self.chain.r[-1] = self.mobile_trap.position


class TrappedChainPlastic(object):

    def __init__(self, chain_elastic_input, theta0_crit, L0_trap, mobile_trap_parameters):

        self.name = "trapped_chain_plastic"
        self.time = 0
        self.chain = ChainPlastic(theta0_crit=theta0_crit, **chain_elastic_input)

        self.static_trap = Trap(1.0*np.array([0, 0]), [0])
        self.mobile_trap = Trap(1.0*np.array([L0_trap, 0]), [self.chain.N-1])

        self.mobile_trap.make_mobile(direction=np.array([-1, 0]), **mobile_trap_parameters)
        # self.mobile_trap.movetrap(self.time)

        self.model_parameters = {"model_name": self.name,
                                 "theta0_crit": L0_trap,
                                 "L0_trap": L0_trap}
        self.model_parameters.update(chain_elastic_input)
        self.model_parameters.update(mobile_trap_parameters)

    def advance_time(self, dt, dtsqrt):

        self.time += dt
        self.chain.integrate(dt, dtsqrt)

        # the trapped particles are fixed by the trap positions:
        self.mobile_trap.movetrap(self.time)
        self.chain.r[0] = self.static_trap.position
        self.chain.r[-1] = self.mobile_trap.position


class HarmonicTrappedChainElastic(object):
    """
    the two traps are harmonic with spring constant k_trap
    """

    def __init__(self, chain_elastic_input, L0_trap, k_trap, mobile_trap_parameters):

        self.name = "harmonic_trapped_chain_elastic"
        self.time = 0
        self.chain = ChainElastic(**chain_elastic_input)

        self.static_trap = Trap(1.0*np.array([0, 0]), [0])
        self.mobile_trap = Trap(1.0*np.array([L0_trap, 0]), [self.chain.N-1])
        self.static_trap.make_harmonic(k_trap)
        self.mobile_trap.make_harmonic(k_trap)

        self.mobile_trap.make_mobile(direction=np.array([-1, 0]), **mobile_trap_parameters)
        # self.mobile_trap.movetrap(self.time)

        # initialise trapping forces
        self.static_trap.f = - self.static_trap.k_trap * \
            (self.chain.r[0]-self.static_trap.position)
        self.mobile_trap.f = - self.static_trap.k_trap*(self.chain.r[-1]-self.mobile_trap.position)

        self.model_parameters = {"model_name": self.name,
                                 "L0_trap": L0_trap}
        self.model_parameters.update(chain_elastic_input)
        self.model_parameters.update(mobile_trap_parameters)

    def advance_time(self, dt, dtsqrt):

        self.time += dt
        self.chain.integrate(dt, dtsqrt)

        # the trapped particles are fixed by the trap positions:
        self.mobile_trap.movetrap(self.time)

        self.static_trap.f = - self.static_trap.k_trap * \
            (self.chain.r[0]-self.static_trap.position)
        self.mobile_trap.f = - self.static_trap.k_trap*(self.chain.r[-1]-self.mobile_trap.position)

        self.chain.r[0] += self.static_trap.f * dt/self.chain.gamma
        self.chain.r[-1] += self.mobile_trap.f*dt/self.chain.gamma


class TrappedSquareLatticeElastic(object):
    """
    with auto trap lattice will start with traps at zero strain distances

    input:
    lattice_elastic_input:
    N_width,
    N_height,
    M_width,
    M_height,
    gamma,
    kT,
    k,
    k_theta,
    lattice0type="unit_square"
    L0_trap_top = "auto"
    L0_trap_right = "auto"
    """

    def __init__(self, lattice_elastic_input, mobile_trap_parameters, L0_trap_top="auto", L0_trap_right="auto"):
        self.name = "trapped_square_lattice_elastic"
        self.time = 0
        self.lattice = SquareLatticeElastic(**lattice_elastic_input)

        # still think about list of trapped particles
        self.trap_outer_particles = False
        self.trap_outer_nodes = True
        if self.trap_outer_particles:
            if L0_trap_top == "auto":
                L0_trap_top = self.lattice.N_height*self.lattice.M_height-1
            if L0_trap_right == "auto":
                L0_trap_right = self.lattice.N_width*self.lattice.M_width-1
            y_bottom = np.min(self.lattice.r[..., 1])
            x_left = np.min(self.lattice.r[..., 0])
            self.static_trap_line_bottom = Trap(1.0*np.array([y_bottom]), [0])
            self.mobile_trap_line_top = Trap(1.0*np.array([y_bottom+L0_trap_top]), [0])
            self.static_trap_line_left = Trap(1.0*np.array([x_left]), [0])
            self.mobile_trap_line_right = Trap(1.0*np.array([x_left+L0_trap_right]), [0])
        elif self.trap_outer_nodes:
            if L0_trap_top == "auto":
                L0_trap_top = (self.lattice.N_height-1)*self.lattice.M_height
            if L0_trap_right == "auto":
                L0_trap_right = (self.lattice.N_width-1)*self.lattice.M_width
            y_bottom = 0
            x_left = 0
            self.static_trap_line_bottom = Trap(1.0*np.array([y_bottom]), [0])
            self.mobile_trap_line_top = Trap(1.0*np.array([y_bottom+L0_trap_top]), [0])
            self.static_trap_line_left = Trap(1.0*np.array([x_left]), [0])
            self.mobile_trap_line_right = Trap(1.0*np.array([x_left+L0_trap_right]), [0])

        self.mobile_trap_line_top.make_mobile(direction=np.array([-1]), **mobile_trap_parameters)
        self.mobile_trap_line_right.make_mobile(direction=np.array([-1]), **mobile_trap_parameters)
        # self.mobile_trap.movetrap(self.time)

        self.model_parameters = {"model_name": self.name,
                                 "L0_trap_top": L0_trap_top,
                                 "L0_trap_right": L0_trap_right}
        self.model_parameters.update(lattice_elastic_input)
        self.model_parameters.update(mobile_trap_parameters)

    def advance_time(self, dt, dtsqrt):

        self.time += dt
        self.lattice.integrate(dt, dtsqrt)

        # the trapped particles are fixed by the trap positions:
        self.mobile_trap_line_top.movetrap(self.time)
        self.mobile_trap_line_right.movetrap(self.time)

        if self.trap_outer_particles:
            self.lattice.r[:, -1, -1, 1] = self.mobile_trap_line_top.position
            self.lattice.r[-1, :, self.lattice.M_width-1, 0] = self.mobile_trap_line_right.position

            self.lattice.r[:, 0, self.lattice.M_width, 1] = self.static_trap_line_bottom.position
            self.lattice.r[0, :, 0, 0] = self.static_trap_line_left.position
        elif self.trap_outer_nodes:
            M_link1 = self.lattice.M_link1
            M_link2 = self.lattice.M_link2
            self.lattice.r[:, -1, M_link1, 1] = self.mobile_trap_line_top.position
            self.lattice.r[:, -1, M_link2, 1] = self.mobile_trap_line_top.position

            self.lattice.r[-1, :, M_link1, 0] = self.mobile_trap_line_right.position
            self.lattice.r[-1, :, M_link2, 0] = self.mobile_trap_line_right.position

            self.lattice.r[:, 0, M_link1, 1] = self.static_trap_line_bottom.position
            self.lattice.r[:, 0, M_link2, 1] = self.static_trap_line_bottom.position

            self.lattice.r[0, :, M_link1, 0] = self.static_trap_line_left.position
            self.lattice.r[0, :, M_link2, 0] = self.static_trap_line_left.position
