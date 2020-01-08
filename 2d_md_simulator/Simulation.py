# -*- coding: utf-8 -*-

"""
2d MD simulations developed at UvA
for brownian mechanically coupled particles of which some are trapped
-overdamped Dynamics
-few different models plastic/elastic chain/square lattice
-few different trap functionalities: elastic different movement programs
-some analysis functionalities: chains mode analysis
-some visualisation functionalities: ffmpeg videos 

still to do:
    clean up code:
        -comment
        -change exception style to:
            except Exception as e:
            print(e)
            sys.exit("devloper comment of probable mistake")


"""


import os
import time
import numpy as np
import sys
from sys import getsizeof

from Model import TrappedChainElastic, TrappedChainPlastic, HarmonicTrappedChainElastic, TrappedSquareLatticeElastic
from Data import Data
from Analyse import Analyse

__author__ = 'Simon Stuij'
__email__ = '{simonstuij@gmail.com}'


class Simulation(object):

    def __init__(self, directory, name):

        self.path = directory+name
        self.name = name
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        print('>>>>>>>>>> Simulation %s <<<<' % (self.name))

        self.data = Data(self.path)
        self.analyse = Analyse(self)
        if sys.version_info >= (3, 0):
            from Visualise import Visualise
            self.visualise = Visualise(self)

    def define_model(self, model_name, model_input):
        """
        """
        if model_name == "trapped_chain_elastic":
            self.model = TrappedChainElastic(**model_input)
        elif model_name == "trapped_chain_plastic":
            self.model = TrappedChainPlastic(**model_input)
        elif model_name == "harmonic_trapped_chain_elastic":
            self.model = HarmonicTrappedChainElastic(**model_input)
        elif model_name == "trapped_square_lattice_elastic":

            self.model = TrappedSquareLatticeElastic(**model_input)

    def set_parameters(self,
                       time_step=(1/128.), nmb_of_datapoints=100,
                       run_time=100,
                       max_save_size=25):
        """
        this only contains parameters that specify the simulation
        not the geometry or interaction strength and temperature
        in essence think of these as not containing any physics
        changing this should model the same system but better/longer
        max_save_size is a measure of how big the data files get before they're being save
        max_save_size=25 translates to ~ 300mb
        """

        steps = int(run_time/time_step)

        write_interval = int(steps/nmb_of_datapoints)
        # 1+ because we also record the 0th time point
        actual_nmb_of_datapoints = 1+int(steps/write_interval)

        self.simulation_parameters = {"time_step": time_step,
                                      "run_time": run_time,
                                      "nmb_of_datapoints": nmb_of_datapoints,
                                      "steps": steps,
                                      "write_interval": write_interval,
                                      "actual_nmb_of_datapoints": actual_nmb_of_datapoints,
                                      "max_save_size": max_save_size}

    def run_simulation(self):
        """
        maybe should split up this method and put in the model class how you
        should save the data and how you have to run the time advancer
        """
        t0_computer = time.time()

        try:
            simulation_parameters = self.simulation_parameters
            print("starting simulation with parameters: ")
            for key, item in simulation_parameters.items():
                print(key, ':', item)
        except Exception as ex:
            message = type(ex).__name__
            exit(message+" you didn't set the simulation parameters")

        try:
            model_parameters = self.model.model_parameters
            print("and using the following model:")
            for key, item in model_parameters.items():
                print(key, ':', item)
        except:
            exit("you didn't select a model")

        dt = simulation_parameters["time_step"]
        dtsqrt = np.sqrt(dt)

        self.data.setup_data_dump(self.model)
        self.data.safe_parameters(self)
        nmb_of_saves = 0

        for k in range(0, simulation_parameters["steps"]+1):
            if (k % simulation_parameters["write_interval"] == 0):

                self.data.add_to_data_dump(self.model)

                # translate to ~ 300mb
                if getsizeof(self.data.particle_dump)/10**6 > simulation_parameters["max_save_size"]:
                    self.data.dump_to_df(safe=True, nmb_of_saves=nmb_of_saves)
                    self.data.setup_data_dump(self.model)
                    nmb_of_saves += 1

            self.model.advance_time(dt, dtsqrt)

        print("production time: "+str(int(time.time()-t0_computer)))
        self.data.dump_to_df(safe=True, nmb_of_saves=nmb_of_saves)

        print("data saved")
