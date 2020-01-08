import pandas as pd
from scipy.fftpack import dst, dct
import numpy as np
import os


class Analyse(object):
    """
    an analysis always needs a data object from which it gets data and to which it can store data
    no wait that doesn't work, no maybe it does
    """

    def __init__(self, simulation):
        self.data = simulation.data

    def dst_modes_from_particles(self):

        time_groups = self.data.df_particle.groupby("time")

        def y_sinfft_group(group):
            """
            because end two particles are bounded zero
            we can apply a dst of type 1 on all the particles
            excluding the two end ones
            the inverse of which is:

            """
            dst_y = dst(group[1:-1], type=1)/(N-1)
            mode_names = ["dst_mode_"+str(i) for i in range(0, len(dst_y))]
            return dict(zip(mode_names, dst_y))

        def inv_f(fy, N):
            """
            this is the reverse of the sp.fftpack.dst type=1
            which is the discrete sine transform type1 from wiki:
            https://en.wikipedia.org/wiki/Discrete_sine_transform
            N is the number of particle then number of modes is N-2
            note the normalization factor 1/N-1 that means that for longer chains all the fourier components will become bigger
            they scale with N,
            as it comes to relative values, the same fourier value on a higher mode requires much more energy
            probably this scales whith mode_nmb squared
            to give a good relative presence of the modes it would be more fair to the higher mode to rescale by this energy
            the exact relation can be found in that one paper we looked at at the beginning
            """
            y = [0]
            for i in range(1, N-1):
                yi = 0
                for k in range(1, N-1):
                    yi += (fy[k-1]/(N-1))*np.sin(k*np.pi*i/(N-1))
                y.append(yi)
            y.append(0)

            return(np.array(y))

        df_modes = time_groups["y"].apply(y_sinfft_group).unstack().reset_index(drop=True)

        return df_modes

    def IPR_from_theta(self, apply_abs_before=False, subtract_straight_mean=False, subtract_mean_below_t=False, t_subtract=[0, 0], apply_abs_after=False, umin=-0.25):
        """
        an average over umin till uc is subtracted if subtract_straight_mean =True
        """

        try:
            time_groups = self.data.df_theta.groupby("time")
        except Exception as ex:
            message = type(ex).__name__
            sys.exit(message+" while grouping df_theta")

        if subtract_straight_mean:
            try:
                constants = self.df_constants
            except:
                print("did you load constants?")

            u0 = constants.loc["u0", "value"]
            uc = constants.loc["uc F fit", "value"]

            u = obs["u_um"].values-u0

            cond_below_uc = (u < uc) & (u > umin)

        if subtract_mean_below_t:
            [t_subtract_min, t_subtract_max] = t_subtract
            t = obs["time"].values
            cond_below_t = (t > t_subtract_min) & (t < t_subtract_max)

        def IPR_group(group):
            """
            group is assumed theta column of df_theta grouped in time
            """

            """
            if apply_abs_before:
                theta = abs(theta)
            if subtract_straight_mean:
                exit("subtract_straight_mean not implemented")
            if subtract_mean_below_t:
                exit("subtract_mean_below_t not implemented")
            if appl_abs_after:
                theta = abs(theta)
            """
            theta_sq_mean = np.mean(group**2)
            theta_mean = np.mean(group)
            theta_fourth_mean = np.mean(group**4)
            theta_sq_mean_subtr = np.mean((group-theta_mean)**2)
            theta_fourth_mean_subtr = np.mean((group-theta_mean)**4)

            IPR2 = theta_sq_mean/theta_mean**2
            IPR4 = (theta_fourth_mean)/(theta_sq_mean**2)
            IPR4_mean_subtr = (theta_fourth_mean_subtr)/(theta_sq_mean_subtr**2)
            return {"IPR2": IPR2, "IPR4": IPR4, "IPR4_mean_subtr": IPR4_mean_subtr}

        df_IPR = time_groups["theta"].apply(IPR_group).unstack().reset_index(drop=True)

        return df_IPR

    def arclength_and_dist(self):

        time_groups = self.data.df_particle.groupby("time")

        def arclength_and_d_group(group):
            """
            """
            y = group["y"].values
            x = group["x"].values
            dy_square = (y[1:]-y[:-1])**2
            dx_square = (x[1:]-x[:-1])**2

            d = np.sqrt(dy_square+dx_square)
            d_names = ["d_"+str(i) for i in range(0, len(d))]
            L_arc = np.sum(d)
            return {"L_arc": L_arc}.update(dict(zip(d_names, d)))

        print(time_groups.apply(arclength_and_d_group))
        df_arclength_and_d = time_groups.apply(
            arclength_and_d_group).unstack().reset_index(drop=True)

        return df_arclength_and_d

    def L_and_u(self):

        N = len(self.data.df_particle["particle"].unique())
        part_in_static_trap = self.data.df_particle.query("particle==0")[["x", "y"]].values
        part_in_mobile_trap = self.data.df_particle.query("particle==@N-1")[["x", "y"]].values

        L_vec = part_in_static_trap-part_in_mobile_trap
        L = np.sqrt(np.einsum('ij,ij->i', L_vec, L_vec))
        u = N-1 - L

        return L, u

    def deflection_dst(self):
        N = len(self.data.df_particle["particle"].unique())
        part_in_static_trap = self.data.df_particle.query("particle==0")[["x", "y"]].values
        part_in_mobile_trap = self.data.df_particle.query("particle==@N-1")[["x", "y"]].values
        [dx, dy] = (part_in_mobile_trap-part_in_static_trap).transpose()
        phi_global = np.arctan2(dy, dx)

        psi = -phi_global
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)
        [xstatic, ystatic] = part_in_static_trap.transpose()

        deflection_list = []
        for n in range(1, N):

            [x, y] = self.data.df_particle.query("particle==@n")[["x", "y"]].values.transpose()
            x = x-xstatic
            y = y-ystatic
            # dx_rot = x_shift*cos_psi-y_shift*sin_psi
            deflection = x*sin_psi+y*cos_psi
            deflection_list.append(deflection)

        deflection_array = np.array(deflection_list).transpose()
        dst_array = dst(deflection_array, type=1)/(N-1)
        dst_list = dst_array.transpose()
        df_deflection = pd.DataFrame()
        df_deflection["phi_global"] = phi_global
        for i in range(0, N-2):
            df_deflection["def_"+str(i)] = deflection_list[i]
            df_deflection["dst_mode_"+str(i)] = dst_list[i]

        return df_deflection

    def particle_dct(self):
        """for some reason the other dct is giving strange results
        this one is similar to the experimental oe
                    # dct
            # use the same normalisation as: https://doi.org/10.1529/biophysj.106.096966
            # and: https://pubs.rsc.org/en/content/articlepdf/2010/sm/c0sm00159g
            # use type2 this some how takes the phi to be the tangent angle at the bond point, so d_0/2
            # doing a dct somehow only is truly correct if all particles are equal sized
            # we could also calculate the matrix of cos(n pi s_i/L) coefficients and invert it
            # but not sure how much better that is
            # ockhams razer says do dct
            # for normalization use "ortho" then exactly phi(s) =sqrt(2/L) sum aq cos qs
            # not that only the resulting dct_modes assume d_0=1 in, to get same units as references:
            # multiply by sqrt(d_0)
        """
        df_particle = self.data.df_particle
        N = len(df_particle["particle"].unique())

        phi_list = []
        for i in range(1, N):

            ri_min1 = df_particle.query("particle==@i-1")[["x", "y"]].values
            ri = df_particle.query("particle==@i")[["x", "y"]].values
            dr = ri-ri_min1
            dx = dr[:, 0]
            dy = dr[:, 1]
            phi = np.arctan2(dy, dx)

            phi_list.append(phi)

        phi_array = np.array(phi_list).transpose()
        dct_array = dct(phi_array, type=2, norm="ortho")
        dct_list = dct_array.transpose()
        df_dct = pd.DataFrame()
        for i in range(0, N-1):
            df_dct["dct_mode_"+str(i)] = dct_list[i]

        return df_dct

    def f_from_trap(self):

        self.data.df_trap

    def dct_from_theta(self):
        """this I guess is wrong dct you have to calculate from phi not theta
        but i think i can fix this by just adding a zero, so assuming the first
        phi is zero, if it is not it will only change the dct mode0 which we are not interested in
        """
        time_groups = self.data.df_theta.groupby("time")

        def dct_group(group):
            """
            """
            theta = group["theta"].values
            phi = np.insert(theta, 0, 0)
            dct_theta = dct(phi, type=1)
            mode_names = ["dct_mode_"+str(i) for i in range(0, len(dct_theta))]
            return dict(zip(mode_names, dct_theta))

        df_dct_modes = time_groups["theta"].apply(dct_group).unstack().reset_index(drop=True)
        return df_dct_modes

    def calculate_observables(self, dst_based_on_deflection=True, IPR=False, add_theta=True, calc_dst=True, calc_dct=False, dct_from_particle=True, safe=False):

        df_observables = pd.DataFrame()

        try:
            time_groups = self.data.df_particle.groupby("time")
        except Exception as ex:
            message = type(ex).__name__
            exit(message+" while grouping df_particle")

        s_meany = time_groups["y"].mean()

        df_observables["time"] = s_meany.index
        df_observables["mean y"] = s_meany.values

        L, u = self.L_and_u()
        df_observables["L"] = L
        df_observables["u"] = u

        Fx_static = -self.data.df_particle.query("particle==0")["fx"].values
        df_observables["Fx_static"] = Fx_static

        # df_arclength_and_d = self.arclength_and_dist()
        # df_observables = df_observables.join(df_arclength_and_d)

        if calc_dst:
            if dst_based_on_deflection:
                df_deflection = self.deflection_dst()
                df_observables = df_observables.join(df_deflection)
            else:
                df_modes = self.dst_modes_from_particles()
                df_observables = df_observables.join(df_modes)

        if IPR:
            df_IPR = self.IPR_from_theta()
            df_observables = df_observables.join(df_IPR)

        if calc_dct:
            if dct_from_particle:
                df_dct = self.particle_dct()
                df_observables = df_observables.join(df_dct)
            else:
                df_dct = self.dct_from_theta()
                df_observables = df_observables.join(df_dct)

        try:
            df_observables = df_observables.join(self.data.df_trap.drop(["time"], axis=1))
        except:
            print("no trap data for this simulation type")
            pass

        if add_theta:
            df_theta = self.data.df_theta
            for i, angle_index in enumerate(df_theta["angle index"].unique()):
                theta_i = df_theta[df_theta["angle index"] == angle_index]["theta"].values
                df_observables["theta_"+str(i)] = theta_i

        self.data.df_observables = df_observables

        if safe:
            safepath = self.data.path+"observables.csv"
            self.data.df_observables.to_csv(safepath)

    def calculate_observables_lattice(self, safe=False):
        """
        assuming we're tweezing the nodes
        """
        df_observables = pd.DataFrame()
        df_part = self.data.df_particle

        df_observables["time"] = df_part["time"].unique()

        [N_width, N_height, M_width, M_height] = self.data.df_parameters.loc["value",
                                                                             ["N_width", "N_height", "M_width", "M_height"]].values

        N_unit_cell = M_width+M_height

        top_link_cond = (df_part["particle"] % (N_width*N_unit_cell)
                         == ((N_width-1)*N_unit_cell)+int(M_width/2))
        bottom_link_cond = (df_part["particle"] % (N_width*N_unit_cell) == int(M_width/2))
        df_part_top = df_part[top_link_cond]
        df_part_bottom = df_part[bottom_link_cond]

        df_observables["fy_top"] = df_part_top.groupby("time")["fy"].mean().values
        df_observables["fy_bottom"] = df_part_bottom.groupby("time")["fy"].mean().values

        right_unit_cell_cond = df_part["particle"] > N_height*(N_width-1)*N_unit_cell
        left_unit_cell_cond = df_part["particle"] < N_height*N_unit_cell
        link_in_unit_cell_cond = (df_part["particle"] % (N_unit_cell) == int(M_width/2))
        df_part_right = df_part[right_unit_cell_cond & link_in_unit_cell_cond]
        df_part_left = df_part[left_unit_cell_cond & link_in_unit_cell_cond]

        df_observables["fx_right"] = df_part_right.groupby("time")["fx"].mean().values
        df_observables["fx_left"] = df_part_left.groupby("time")["fx"].mean().values

        df_observables["f_avg"] = (df_observables["fx_right"] +
                                   -1*df_observables["fx_left"]+-1*df_observables["fy_bottom"]+df_observables["fy_top"])/4
        self.data.df_observables = df_observables

        if safe:
            safepath = self.data.path+"observables.csv"
            self.data.df_observables.to_csv(safepath)

    def calculate_observables_big_data(self, nmb_of_data_files=1, calc_dst=False, safe=False):
        """
        nmb_of_data_files cna be a number or "all"
        """

        if nmb_of_data_files == "all":
            particle_files = [s for s in os.listdir(self.data.path) if "particle" in s]
            nmb_of_data_files = len(particle_files)

        print("loading: "+str(nmb_of_data_files)+" files")

        df_observables = pd.DataFrame()
        for i in range(0, nmb_of_data_files):
            print(i)
            self.data.data_load(
                take_every=1,
                particle=True,
                theta=False,
                observables=False,
                observables_vs_u=False,
                string_to_array=False,
                trap=False, start_save_nmb=i, untill_save_nmb=i+1)

            self.calculate_observables(IPR=False, add_theta=False,
                                       calc_dst=calc_dst, calc_dct=False, safe=False)
            df_observables = df_observables.append(self.data.df_observables, ignore_index=True)

        self.data.df_observables = df_observables
        if safe:
            safepath = self.data.path+"observables.csv"
            self.data.df_observables.to_csv(safepath)

    def mode_of_distribution(self, dst1_values, bins="auto"):
        hist, bin_edges = np.histogram(dst1_values, bins=bins, normed=True)
        bin_centers = (bin_edges[1:]+bin_edges[:-1])/2
        bin_width = bin_edges[1]-bin_edges[0]

        try:
            max_hist_pos = max(hist[bin_centers >= 0])
            cond = np.logical_and(bin_centers >= 0, hist == max_hist_pos)
            mode_pos = bin_centers[cond][0]
        except Exception as ex:
            message = type(ex).__name__
            print("error" + message+" finding mode of positive OP values ")
            mode_pos = np.nan
        try:
            max_hist_neg = max(hist[bin_centers <= 0])
            cond = np.logical_and(bin_centers <= 0, hist == max_hist_neg)
            mode_neg = bin_centers[cond][0]
        except Exception as ex:
            message = type(ex).__name__
            print("error" + message+" finding mode of negative OP values ")
            mode_neg = np.nan
        try:
            max_hist = max(hist)
            mode = bin_centers[hist == max_hist][0]
            mode = abs(mode)
        except Exception as ex:
            message = type(ex).__name__
            print("error" + message+" finding mode of all ")
            mode = np.nan

        return mode

    def obs_vs_u(self, safe=False):

        N = self.data.df_parameters["N"].value
        L0 = N-1
        kth_0 = self.data.df_parameters["k_theta"].value
        k_0 = self.data.df_parameters["k"].value
        Fc_expect = kth_0*np.pi**2/(L0)**2
        S_0 = k_0/(L0)
        uc_expect = Fc_expect/S_0

        steps = int(np.round(self.data.df_observables["time"].max(
        )/self.data.df_parameters["stepduration"].value))

        observables = self.data.df_observables
        extra = int(len(observables["Fx_static"].values) % steps)

        u_raw = observables["u"].values[:-extra]
        dst1_raw = observables["dst_mode_0"].values[:-extra]
        Fx = np.mean(np.split(observables["Fx_static"].values[:-extra], steps), axis=1)
        var_Fx = np.var(np.split(observables["Fx_static"].values[:-extra], steps), axis=1)
        u = np.mean(np.split(u_raw, steps), axis=1)
        dst1 = np.mean(np.split(dst1_raw, steps), axis=1)

        var_dst1 = np.var(np.split(dst1_raw, steps), axis=1)

        dst1_mode = np.array([self.mode_of_distribution(dst1_part, bins=1000)
                              for dst1_part in np.split(dst1_raw, steps)])

        idx_F0 = (np.abs(u)).argmin()
        Fx0 = Fx[idx_F0]
        # idx_u0 = (np.abs(Fx)).argmin() don't use this one prefer F route
        # u0 = u[idx_u0]
        u0_expect = -1.0*Fx0/S_0
        print("expected u0: "+str(u0_expect))
        uc_expect_shift = uc_expect+u0_expect

        u_norm = (u-u0_expect)/uc_expect
        F_norm = Fx/Fc_expect

        def M1_elastica_vs_u(u, uc):
            return (np.piecewise(u, [u < uc, u >= uc], [lambda u: 0, lambda u:(2/np.pi)*np.sqrt(L0*(u-uc))]))

        def F_elastica_vs_u(u, uc):
            return (np.piecewise(u, [u < uc, u >= uc], [lambda u: S_0*u, lambda u: Fc_expect+Fc_expect*0.5*(u-uc)/L0]))
        dst1_elastica = M1_elastica_vs_u(u_raw, uc_expect_shift)
        F_elastica = F_elastica_vs_u(u, uc_expect)

        dst1_above_elastica = np.split(dst1_raw-dst1_elastica, steps)
        dst1_below_elastica = np.split(dst1_raw+dst1_elastica, steps)
        dst1_beyond_elastica = [np.concatenate((above[above > 0], below[below < 0])) for above, below in zip(
            dst1_above_elastica, dst1_below_elastica)]
        var_above_elastica = [np.sum(beyond**2) for beyond in dst1_beyond_elastica]

        dst1_mode_raw = np.reshape(
            np.ones(shape=(steps, int(len(dst1_raw)/steps)))*dst1_mode[:, None], len(dst1_raw))
        dst1_above_mode = np.split(dst1_raw-dst1_mode_raw, steps)
        dst1_below_mode = np.split(dst1_raw+dst1_mode_raw, steps)
        dst1_beyond_mode = [np.concatenate((above[above > 0], below[below < 0])) for above, below in zip(
            dst1_above_mode, dst1_below_mode)]
        var_above_mode = [np.mean(beyond**2) for beyond in dst1_beyond_mode]

        df_obs_vs_u = pd.DataFrame()
        df_obs_vs_u["u"] = u
        df_obs_vs_u["u_shift"] = u-uc_expect
        df_obs_vs_u["Fx"] = Fx
        df_obs_vs_u["Fent"] = Fx-F_elastica
        df_obs_vs_u["dst1"] = dst1
        df_obs_vs_u["dst1_elastica"] = M1_elastica_vs_u(u, uc_expect_shift)
        df_obs_vs_u["var_dst1"] = var_dst1
        df_obs_vs_u["dst1_mode"] = dst1_mode

        df_obs_vs_u["u_norm"] = u_norm
        df_obs_vs_u["F_norm"] = F_norm
        df_obs_vs_u["var_above_elastica"] = var_above_elastica
        df_obs_vs_u["var_above_mode"] = var_above_mode
        df_obs_vs_u["var_Fx"] = var_Fx

        dst1_split = np.split(dst1_raw, steps)
        recordings_per_step = int(len(dst1_raw)/steps)
        for var_cutoff_steps in range(50, int(recordings_per_step/50.), 50):
            var_dst1_cutoff = [np.mean([np.var(a) for a in np.array_split(
                dst1_splitted, var_cutoff_steps)])
                for dst1_splitted in dst1_split]
            df_obs_vs_u["var_dst1_cutoff"+str(var_cutoff_steps)] = var_dst1_cutoff

        # some more paramters:
        dst1_squared = np.mean(np.split(dst1_raw**2, steps), axis=1)
        df_obs_vs_u["dst1_squared"] = dst1_squared
        dst1_fourth = np.mean(np.split(dst1_raw**4, steps), axis=1)
        df_obs_vs_u["dst1_fourth"] = dst1_fourth
        df_obs_vs_u["binder"] = 1-dst1_fourth/(3*dst1_squared)
        df_obs_vs_u["simon"] = dst1_squared - dst1**2
        var_dst1_squared = np.var(np.split(dst1_raw**2, steps), axis=1)
        df_obs_vs_u["simon2"] = var_dst1_squared
        self.data.df_obs_vs_u = df_obs_vs_u

        if safe:
            safepath = self.data.path+"observables_vs_u.csv"
            self.data.df_obs_vs_u.to_csv(safepath)

    def calculate_observables_lattice_big_data(self, nmb_of_data_files, safe=False):

        if nmb_of_data_files == "all":
            particle_files = [s for s in os.listdir(self.data.path) if s.startswith("particle")]
            nmb_of_data_files = len(particle_files)
        print("loading: "+str(nmb_of_data_files)+" files")
        df_observables = pd.DataFrame()
        for i in range(0, nmb_of_data_files):
            print(i)
            self.data.data_load(
                take_every=1,
                particle=True,
                theta=False,
                observables=False,
                observables_vs_u=False,
                string_to_array=False,
                trap=False, start_save_nmb=i, untill_save_nmb=i+1)

            self.calculate_observables_lattice(safe=False)
            df_observables = df_observables.append(self.data.df_observables, ignore_index=True)

        self.data.df_observables = df_observables
        if safe:
            safepath = self.data.path+"observables.csv"
            self.data.df_observables.to_csv(safepath)

        print("observables calculated")

    def dimensionalise_obs_vs_u(self, d0_um=2.744, kT_pNum=0.00413021, D_um2_per_s=0.138):

        self.data.df_obs_vs_u_original = self.data.df_obs_vs_u.copy()
        tD_s = d0_um*2/D_um2_per_s
        self.data.df_obs_vs_u.loc[:, ["u", "u_shift",
                                      "dst1", "dst1_elastica", "dst1_mode"]] *= d0_um
        self.data.df_obs_vs_u.loc[:, ["Fx", "Fent"]] *= kT_pNum/d0_um
        self.data.df_obs_vs_u.loc[:, "var_Fx"] *= kT_pNum**2/d0_um**2
        self.data.df_obs_vs_u.loc[:, ["var_dst1",
                                      "var_above_elastica", "var_above_mode"]] *= d0_um**2

        cutoff_names = [i for i in self.data.df_obs_vs_u.columns if i.startswith('var_dst1_cutoff')]
        for cutoff_name in cutoff_names:
            self.data.df_obs_vs_u.loc[:, cutoff_name] *= d0_um**2

        self.data.df_obs_vs_u_dimensionalised = self.data.df_obs_vs_u
        self.data.df_obs_vs_u = self.data.df_obs_vs_u_original

    def dimensionalise_observables(self, d0_um, kT_pNum, D_um2_per_s):

        self.data.df_observables_original = self.data.df_observables.copy()

        tD_s = d0_um*2/D_um2_per_s

        self.data.df_observables.loc[:, "time"] *= tD_s

        self.data.df_observables.loc[:, "Fx_static"] *= kT_pNum/d0_um

        self.data.df_observables.loc[:, ["mean y", "L", "u"]] *= d0_um

        try:
            self.data.df_observables.loc[:, ["fx_stat",
                                             "fy_stat", "fx_mob", "fy_mob"]] *= kT_pNum/d0_um
        except:
            print("no trap data for this simulation type")
            pass

        dst_modes = self.data.df_observables.columns.str.startswith('dst_mode_')
        self.data.df_observables.loc[:, dst_modes] *= d0_um

        self.data.df_observables_dimensionalised = self.data.df_observables
        self.data.df_observables = self.data.df_observables_original
