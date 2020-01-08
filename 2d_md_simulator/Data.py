import pandas as pd
import os
import numpy as np
from shutil import copyfile


class Data(object):

    def __init__(self, simulation_path):
        self.path = (simulation_path + '/data/')

    def safe_parameters(self, simulation, safe=True):

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        safe_path = self.path+"model_parameters.csv"
        self.model_parameters = pd.DataFrame(simulation.model.model_parameters, index=["value"])
        self.model_parameters.to_csv(safe_path)

        safe_path = self.path+"simulation_parameters.csv"
        self.simulation_parameters = pd.DataFrame(simulation.simulation_parameters, index=["value"])
        self.simulation_parameters.to_csv(safe_path)

    def setup_data_dump(self, model):
        """
        model is an instance of a model object
        we don't need to store theta0 and d0 if
        """
        if model.name == "trapped_chain_elastic":
            self.particle_dump_columns = ["x", "y", "fx", "fy", "particle", "time"]
            self.particle_dump = []
            self.theta_dump_columns = ["theta", "theta0", "angle index", "time"]
            self.theta_dump = []
        elif model.name == "trapped_chain_plastic":
            self.particle_dump_columns = ["x", "y", "fx", "fy", "particle", "time"]
            self.particle_dump = []
            self.theta_dump_columns = ["theta", "theta0", "angle index", "time"]
            self.theta_dump = []
        elif model.name == "harmonic_trapped_chain_elastic":
            self.particle_dump_columns = ["x", "y", "fx", "fy", "particle", "time"]
            self.particle_dump = []
            self.theta_dump_columns = ["theta", "theta0", "angle index", "time"]
            self.theta_dump = []
            self.trap_dump_columns = ["fx_stat", "fy_stat", "fx_mob",
                                      "fy_mob", "x_stat", "y_stat", "x_mob", "y_mob", "time"]
            self.trap_dump = []
        elif model.name == "free_chain_break":
            self.particle_dump_columns = ["x", "y", "fx", "fy", "particle", "time"]
            self.particle_dump = []
            self.theta_dump_columns = ["k_theta", "theta", "theta0", "angle index", "time"]
            self.theta_dump = []
            self.d_dump_columns = ["k", "d", "d0", "bond index", "time"]
            self.d_dump = []
        elif model.name == "trapped_square_lattice_elastic":
            self.particle_dump_columns = ["x", "y", "fx", "fy", "psi", "particle", "time"]
            self.particle_dump = []
            self.horizontal_chain_particle_dump_columns = [
                "x", "y", "fx", "fy", "particle", "chain", "time"]
            self.horizontal_chain_particle_dump = []
            self.vertical_chain_particle_dump_columns = [
                "x", "y", "fx", "fy", "particle", "chain", "time"]
            self.vertical_chain_particle_dump = []

    def add_to_data_dump(self, model):
        """
        model is a Chain model instance


        """
        time = model.time

        if model.name == "trapped_chain_elastic":
            chain = model.chain
            for i, (ri, fi) in enumerate(zip(chain.r, chain.f)):
                self.particle_dump.append([ri[0], ri[1], fi[0], fi[1],
                                           i,
                                           time])
            for i, (thetai, theta0i) in enumerate(zip(chain.theta, chain.theta0)):
                self.theta_dump.append([thetai, theta0i, i, time])

        elif model.name == "trapped_chain_plastic":
            chain = model.chain
            for i, (ri, fi) in enumerate(zip(chain.r, chain.f)):
                self.particle_dump.append([ri[0], ri[1], fi[0], fi[1],
                                           i,
                                           time])
            for i, (thetai, theta0i) in enumerate(zip(chain.theta, chain.theta0)):
                self.theta_dump.append([thetai, theta0i, i, time])

        elif model.name == "harmonic_trapped_chain_elastic":
            chain = model.chain
            (stat, mob) = (model.static_trap, model.mobile_trap)
            self.trap_dump.append([stat.f[0], stat.f[1], mob.f[0], mob.f[1],
                                   stat.position[0], stat.position[1], mob.position[0], mob.position[1],
                                   time])
            for i, (ri, fi) in enumerate(zip(chain.r, chain.f)):
                self.particle_dump.append([ri[0], ri[1], fi[0], fi[1],
                                           i,
                                           time])
            for i, (thetai, theta0i) in enumerate(zip(chain.theta, chain.theta0)):
                self.theta_dump.append([thetai, theta0i, i, time])

        elif model.name == "free_chain_break":
            chain = model.chain
            for i, (ri, fi) in enumerate(zip(chain.r, chain.f)):
                self.particle_dump.append([ri[0], ri[1], fi[0], fi[1],
                                           i,
                                           time])
            for i, (k_theta, thetai, theta0i) in enumerate(zip(chain.k_theta, chain.theta, chain.theta0)):
                self.theta_dump.append([k_theta, thetai, theta0i, i, time])
            for i, (k, di, d0i) in enumerate(zip(chain.k, chain.d, chain.d0)):
                self.d_dump.append([k, di, d0i, i, time])

        elif model.name == "trapped_square_lattice_elastic":
            lattice = model.lattice
            (r, f, psi) = (lattice.r, lattice.f, lattice.psi)
            r_shape = np.shape(r)
            r_reshape = (r_shape[0]*r_shape[1]*r_shape[2], r_shape[3])
            r_flatten = np.reshape(r, r_reshape)
            f_flatten = np.reshape(f, r_reshape)
            psi_flatten = np.reshape(psi, r_reshape[0])

            for i, (ri, fi, psi_i) in enumerate(zip(r_flatten, f_flatten, psi_flatten)):
                self.particle_dump.append([ri[0], ri[1], fi[0], fi[1], psi_i,
                                           i,
                                           time])

            # horizontal chains
            for j, horizontal_chain_i in enumerate(lattice.horizontal_chain_list):
                for i, (ri, fi) in enumerate(zip(horizontal_chain_i.r, horizontal_chain_i.f)):
                    self.horizontal_chain_particle_dump.append([ri[0], ri[1], fi[0], fi[1],
                                                                i, j,
                                                                time])
            # vertical chains
            for j, horizontal_chain_i in enumerate(lattice.vertical_chain_list):
                for i, (ri, fi) in enumerate(zip(horizontal_chain_i.r, horizontal_chain_i.f)):
                    self.vertical_chain_particle_dump.append([ri[0], ri[1], fi[0], fi[1],
                                                              i, j,
                                                              time])

    def dump_to_df(self, safe=True, nmb_of_saves=0):

        if safe:
            nmb_of_saves = str(nmb_of_saves).zfill(3)

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.df_particle = pd.DataFrame(self.particle_dump, columns=self.particle_dump_columns)
        if safe:

            self.df_particle.to_csv(self.path+"particle"+nmb_of_saves+".csv")
            print("particle data "+nmb_of_saves+"for this simulation type saved")

        try:
            self.df_theta = pd.DataFrame(self.theta_dump, columns=self.theta_dump_columns)
            if safe:
                self.df_theta.to_csv(self.path+"theta"+nmb_of_saves+".csv")
                print("theta data "+nmb_of_saves+" for this simulation type saved")
        except:
            print("no theta data for this simulation type")
            pass

        try:
            self.df_trap = pd.DataFrame(self.trap_dump, columns=self.trap_dump_columns)
            if safe:
                self.df_trap.to_csv(self.path+"trap"+nmb_of_saves+".csv")
                print("trap data "+nmb_of_saves+" for this simulation type saved")
        except:
            print("no trap data for this simulation type")
            pass

        try:
            self.df_d = pd.DataFrame(self.d_dump, columns=self.d_dump_columns)
            if safe:
                self.df_d.to_csv(self.path+"d"+nmb_of_saves+".csv")
                print("d data "+nmb_of_saves+" for this simulation type saved")
        except:
            print("no d data for this simulation type")
            pass

        try:
            self.df_vertical_chain_particle = pd.DataFrame(
                self.vertical_chain_particle_dump, columns=self.vertical_chain_particle_dump_columns)
            self.df_horizontal_chain_particle = pd.DataFrame(
                self.horizontal_chain_particle_dump, columns=self.horizontal_chain_particle_dump_columns)

            if safe:
                self.df_vertical_chain_particle.to_csv(
                    self.path+"vertical_chain_particle"+nmb_of_saves+".csv")
                self.df_horizontal_chain_particle.to_csv(
                    self.path+"horizontal_chain_particle"+nmb_of_saves+".csv")
                print("vertical and horizontal chain particle " +
                      nmb_of_saves+" data for this simulation type saved")
        except:
            print("no sub chain particle data for this simulation type")
            pass

    def data_load(self,
                  take_every=1,
                  parameters=True,
                  particle=False,
                  theta=False,
                  observables=False,
                  observables_vs_u=False,
                  string_to_array=False,
                  trap=False,
                  subset_particle=False,
                  start_save_nmb=None,
                  untill_save_nmb=None):

        if parameters:
            self.df_parameters = pd.read_csv(self.path+"model_parameters.csv", index_col=0)

        if take_every != 1:
            datapoints = self.parameters.actual_nmb_of_datapoints
            N = self.model.N

        if particle:
            try:
                if take_every != 1:
                    skip_idx = [i for i in range(1, N*datapoints+1)
                                if ((i-1) % (N*take_every)) >= N]
                else:
                    skip_idx = None
                # columns = ["x", "y", "px", "py", "fx", "fy", "particle", "time"]

                if untill_save_nmb == None:
                    self.df_particle = pd.read_csv(self.path+"particle.csv",
                                                   skiprows=skip_idx, index_col=0)
                else:
                    for i in range(start_save_nmb, untill_save_nmb):
                        str_untill_save_nmb = str(i).zfill(3)
                        if i == start_save_nmb:
                            self.df_particle = pd.read_csv(self.path+"particle"+str_untill_save_nmb+".csv",
                                                           skiprows=skip_idx, index_col=0)
                        else:
                            self.df_particle = self.df_particle.append(pd.read_csv(self.path+"particle"+str_untill_save_nmb+".csv",
                                                                                   skiprows=skip_idx, index_col=0), ignore_index=True)

            except Exception as ex:
                message = type(ex).__name__
                exit(message+" while loading particle data")

        if observables:
            try:
                if take_every != 1:
                    skip_idx = [x for x in range(1, datapoints+1)
                                if ((x-1) % take_every) >= 1]
                else:
                    skip_idx = None
                self.df_observables = pd.read_csv(self.path+"observables.csv",
                                                  skiprows=skip_idx, index_col=0)
            except Exception as ex:
                message = type(ex).__name__
                exit(message+" while loading observables data")

        if observables_vs_u:
            try:
                self.df_obs_vs_u = pd.read_csv(self.path+"observables_vs_u.csv",
                                               index_col=0)
            except Exception as ex:
                message = type(ex).__name__
                exit(message+" while loading observables data")

        if theta:
            try:
                if take_every != 1:
                    skip_idx = [x for x in range(1, (N-2)*datapoints+1)
                                if ((x-1) % ((N-2)*take_every)) >= N-2]
                else:
                    skip_idx = None

                if untill_save_nmb == None:
                    self.df_theta = pd.read_csv(self.path+"theta.csv",
                                                skiprows=skip_idx, index_col=0)
                else:
                    for i in range(start_save_nmb, untill_save_nmb):
                        str_untill_save_nmb = str(i).zfill(3)
                        if i == start_save_nmb:
                            self.df_theta = pd.read_csv(self.path+"theta"+str_untill_save_nmb+".csv",
                                                        skiprows=skip_idx, index_col=0)
                        else:
                            self.df_theta = self.df_theta.append(pd.read_csv(self.path+"theta"+str_untill_save_nmb+".csv",
                                                                             skiprows=skip_idx, index_col=0), ignore_index=True)

            except Exception as ex:
                message = type(ex).__name__
                exit(message+" while loading theta data")

        if trap:
            try:
                if take_every != 1:
                    skip_idx = [x for x in range(1, datapoints+1)
                                if ((x-1) % take_every) >= 1]
                else:
                    skip_idx = None
                self.df_trap = pd.read_csv(self.path+"trap.csv",
                                           skiprows=skip_idx, index_col=0)
            except Exception as ex:
                message = type(ex).__name__
                exit(message+" while loading trap data")

        if subset_particle:
            try:
                self.subset_particle = pd.read_csv(self.path+"subset_particle.csv",
                                                   index_col=0)
            except Exception as ex:
                message = type(ex).__name__
                exit(message+" while loading observables data")

    def save_subset_particles(self, taking_every_nth_frames):
        """
        loads the data and saves every nth frame in a separte csv file,
        usefull for movies
        """

        particle_files = [s for s in os.listdir(self.path) if s.startswith("particle")]
        nmb_of_data_files = len(particle_files)

        subset_particle = pd.DataFrame()

        for i in range(0, nmb_of_data_files):
            self.data_load(
                take_every=1,
                particle=True,
                theta=False,
                observables=False,
                observables_vs_u=False,
                string_to_array=False,
                trap=False, start_save_nmb=i, untill_save_nmb=i+1)

            particle_slice = self.df_particle
            subset_times = particle_slice["time"].unique()

            selected_times = subset_times[::taking_every_nth_frames]

            subset_slice = particle_slice[particle_slice["time"].isin(selected_times)]

            subset_particle = pd.concat([subset_particle, subset_slice], ignore_index=True)

        safe_path = self.path+"subset_particle.csv"
        self.subset_particle = subset_particle
        self.subset_particle.to_csv(safe_path)

    def copy_data_to_other_folder(self,
                                  sim_name,
                                  save_path2,
                                  parameters=True,
                                  particle=False,
                                  theta=False,
                                  observables=False,
                                  observables_vs_u=False,
                                  string_to_array=False,
                                  trap=False,
                                  subset_particle=False):
        """
        copies a subset of the simulation data to a seperate folder save_path2
        this is useful for copying files from  obelix
        """
        destination_path = save_path2+sim_name+"/data/"
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)

        if parameters:
            source = self.path+"model_parameters.csv"
            destination = destination_path+"model_parameters.csv"
            copyfile(source, destination)

            source = self.path+"simulation_parameters.csv"
            destination = destination_path+"simulation_parameters.csv"
            copyfile(source, destination)

        if particle:
            source = self.path+"particle.csv"
            destination = destination_path+"particle.csv"
            copyfile(source, destination)

        if observables:
            source = self.path+"observables.csv"
            destination = destination_path+"observables.csv"
            copyfile(source, destination)

        if observables_vs_u:
            source = self.path+"observables_vs_u.csv"
            destination = destination_path+"observables_vs_u.csv"
            copyfile(source, destination)

        if theta:
            source = self.path+"theta.csv"
            destination = destination_path+"theta.csv"
            copyfile(source, destination)

        if trap:
            source = self.path+"trap.csv"
            destination = destination_path+"trap.csv"
            copyfile(source, destination)

        if subset_particle:
            source = self.path+"subset_particle.csv"
            destination = destination_path+"subset_particle.csv"
            copyfile(source, destination)
