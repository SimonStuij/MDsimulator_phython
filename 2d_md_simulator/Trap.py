"""
Created on Fri Oct  12 20:10:00 2018



"""
import numpy as np
from math import floor


class Trap(object):
    """

    this gives you positions and which particle are trapped and how the trap moves in time
    not nature of trap like boundary condition
    """

    def __init__(self, initial_position, list_of_trapped_particles):
        self.initial_position = initial_position
        self.position = initial_position
        self.trapped_particles = list_of_trapped_particles

    def make_harmonic(self, k_trap):
        """
        ftrap is the force exerted on the particle
        """
        self.f = 1.0*np.zeros(2)
        self.k_trap = k_trap

    def make_mobile(self,
                    traptype=None,
                    tstart=None,
                    tend=None,
                    stepduration=None,
                    stepsize=None,
                    period=None,
                    amplitude=None,
                    direction=None):
        """
        sorry for this error testing style,
        I didn't now the try except syntax yet
        """
        self.traptype = traptype
        if self.traptype == "triangle":
            if all(v is not None for v in [tstart, tend, amplitude, direction, period]):
                self.tstart = tstart
                self.tend = tend
                self.amplitude = amplitude
                self.direction = direction
                self.period = period
            else:
                sys.exit("not enough trap parameters for this traptype")
        elif self.traptype == "linear":
            if all(v is not None for v in [tstart, tend, amplitude, direction]):
                self.tstart = tstart
                self.tend = tend
                self.amplitude = amplitude
                self.direction = direction
            else:
                sys.exit("not enough trap parameters for this traptype")
        elif self.traptype == "stepwise":
            if all(v is not None for v in [stepduration, stepsize, direction]):
                self.stepduration = stepduration
                self.stepsize = stepsize
                self.direction = direction
            else:
                sys.exit("not enough trap parameters for this traptype")
        elif self.traptype == "static":
            pass
        else:
            sys.exit("traptype has to be either triangle,linear,"
                     "stepwise or static")

    def linear_trap(self, t):
        """
        moves in a linear line between time tstart to tend
        when reaching tend it stays in its final position
        """
        if t < self.tstart:
            pass
        elif (t >= self.tstart) & (t <= self.tend):
            self.position = np.copy(self.initial_position)
            self.position += (self.amplitude*self.direction *
                              (t-self.tstart)/(self.tend-self.tstart))
        elif t > self.tend:
            pass

    def triangle_trap(self, t):
        self.position = np.copy(self.initial_position)
        if (t >= self.tstart) & (t <= self.tend):
            t_shift = t-self.tstart
            triangle_wave = 2*(abs(2*(t_shift/self.period+0.25 -
                                      floor(t_shift/self.period+0.75)))-0.5)
            self.position += self.direction*self.amplitude*triangle_wave

    def stepwise_trap(self, t):
        whichstep = floor(t/self.stepduration)
        self.position = (self.initial_position +
                         whichstep*self.stepsize*self.direction)

    def critical_test_trap(self, t):
        whichstep = floor(t/self.stepduration)

        self.position = (self.initial_position +
                         whichstep*self.stepsize*self.direction)

    def static_trap(self, t):

        pass

    def movetrap(self, t):
        if self.traptype == "triangle":
            self.triangle_trap(t)
        elif self.traptype == "linear":
            self.linear_trap(t)
        elif self.traptype == "stepwise":
            self.stepwise_trap(t)
        elif self.traptype == "static":
            self.static_trap(t)
