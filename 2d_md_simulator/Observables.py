import pandas as pd
from scipy.fftpack import dst, dct
import numpy as np
import os
import re


def calculate_observables_direct(features, fix_N=False):
    """
    input:
    -------
    features: pd.DataFrame with time,x,y,particle columns
        NOTE that chain needs to be oriented with it's long axis along x
        otherwise there might be some artifacts with the angle calculations
        (can be fixed but slows down program a lot)

    output:
    observables Dataframe: wit columns
        phi: tangent angles between particles
        theta: nnn bond angles
        dct_mode: a dct fourier on the tangent angles
        def: the deflection out of the line connecting the two end particles
            (useful for trapped strings)
        dst: a dst fourier transform on the deflection
            (useful for trapped strings)
        true_mode: a linear transform on the dct modes to get the "true" uncoupled dynamical modes


    """
    obs = Observables(features, fix_N=fix_N)
    obs.calculate_observables()

    return obs.df_observables


class Observables(object):

    def __init__(self, features, fix_N=False):
        self.features = features
        if fix_N:
            self.features_fix_N()

    def features_fix_N(self):
        """
        removes frames where not all the particles are present
        """
        new_f = self.features.copy()
        N = len(new_f["particle"].unique())
        for i in range(0, N):
            i_present = new_f.query("particle==@i")["frame"].values

            new_f = new_f[new_f["frame"].isin(i_present)]

        self.features = new_f

    def calculate_observables(self):

        df_observables = pd.DataFrame()

        try:
            time_groups = self.features.groupby("time")
        except Exception as ex:
            message = type(ex).__name__
            exit(message+" while grouping df_particle")

        s_meany = time_groups["y"].mean()
        df_observables["time"] = s_meany.index

        L, u = self.L_and_u()
        df_observables["L"] = L
        df_observables["u"] = u

        df_deflection = self.deflection_dst()
        df_observables = df_observables.join(df_deflection)

        df_dist = self.arclength_and_dist()
        df_observables = df_observables.join(df_dist)

        df_dct = self.particle_dct()
        df_observables = df_observables.join(df_dct)

        self.df_observables = df_observables
        self.calculate_true_dynamic_modes()

        df_diffusion = self.diffusion()
        self.df_observables = self.df_observables.join(df_diffusion)

    def diffusion(self):
        """
        diffusion forward and transverse
        """
        observables = self.df_observables

        observables = observables.drop(columns="phi_global")
        obs_columns = observables.columns

        phi_names = [i for i in obs_columns if i.startswith('phi_')]

        phi_mean = observables[phi_names].mean(axis=1).values

        [com_x, com_y] = self.features.groupby("time")[["x", "y"]].mean().values.transpose()

        cos_phi = np.cos(-phi_mean[:-1])
        sin_phi = np.sin(-phi_mean[:-1])
        dx = com_x[1:]-com_x[:-1]
        dy = com_y[1:]-com_y[:-1]

        d_forward = cos_phi*dx-sin_phi*dy
        d_transverse = sin_phi*dx-cos_phi*dy

        df_diffusion = pd.DataFrame()

        df_diffusion["phi_mean"] = phi_mean
        df_diffusion["com_x"] = com_x
        df_diffusion["com_y"] = com_y

        d_transverse = np.append(0, d_transverse)
        d_forward = np.append(0, d_forward)
        df_diffusion["com_transverse"] = np.cumsum(d_transverse)
        df_diffusion["com_forward"] = np.cumsum(d_forward)

        return df_diffusion

    def arclength_and_dist(self):

        N = len(self.features["particle"].unique())

        df_arclength_and_d = pd.DataFrame()
        for n in range(0, N-1):

            [x, y] = self.features.query("particle==@n")[["x", "y"]].values.transpose()
            [xnext, ynext] = self.features.query("particle==(@n+1)")[["x", "y"]].values.transpose()
            dx = x-xnext
            dy = y-ynext
            # dx_rot = x_shift*cos_psi-y_shift*sin_psi
            dr = np.sqrt(dx**2+dy**2)
            df_arclength_and_d["d_"+str(n)] = dr
            if n == 0:
                df_arclength_and_d["L_arc"] = dr
            else:
                df_arclength_and_d["L_arc"] += dr

        return df_arclength_and_d

    def L_and_u(self):

        N = len(self.features["particle"].unique())
        part_in_static_trap = self.features.query("particle==0")[["x", "y"]].values
        part_in_mobile_trap = self.features.query("particle==@N-1")[["x", "y"]].values

        L_vec = part_in_static_trap-part_in_mobile_trap
        L = np.sqrt(np.einsum('ij,ij->i', L_vec, L_vec))
        u = N-1 - L

        return L, u

    def natural_keys(self, text):
        '''
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        '''
        def atoi(text):
            return int(text) if text.isdigit() else text
        return [atoi(c) for c in re.split(r'(\d+)', text)]

    def deflection_dst(self):
        """
        this first calculates deflection out of the line connecting the two
        end particles
        """
        N = len(self.features["particle"].unique())
        part_in_static_trap = self.features.query("particle==0")[["x", "y"]].values
        part_in_mobile_trap = self.features.query("particle==@N-1")[["x", "y"]].values
        [dx, dy] = (part_in_mobile_trap-part_in_static_trap).transpose()
        phi_global = np.arctan2(dy, dx)

        psi = -phi_global
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)
        [xstatic, ystatic] = part_in_static_trap.transpose()

        deflection_list = []
        for n in range(1, N):

            [x, y] = self.features.query("particle==@n")[["x", "y"]].values.transpose()
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
            df_deflection["dst_"+str(i)] = dst_list[i]

        return df_deflection

    def particle_dct(self):
        """
        use the same normalisation as: https://doi.org/10.1529/biophysj.106.096966
        and: https://pubs.rsc.org/en/content/articlepdf/2010/sm/c0sm00159g
        use type2 this some how takes the phi to be the tangent angle at the bond point, so d_0/2
        doing a dct somehow only is truly correct if all particles are equal sized
        we could also calculate the matrix of cos(n pi s_i/L) coefficients and invert it
        but not sure how much better that is
        ockhams razer says do dct
        for normalization use "ortho" then exactly phi(s) =sqrt(2/L) sum aq cos qs
        not that only the resulting dct_modes assume d_0=1 in, to get same units as references:
        multiply by sqrt(d_0)
        """
        df_particle = self.features
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
            df_dct["dct_"+str(i)] = dct_list[i]
            df_dct["phi_"+str(i)] = phi_list[i]
        for i in range(0, N-2):
            df_dct["theta_"+str(i)] = phi_list[i+1]-phi_list[i]
        for i in range(0, N-3):
            df_dct["psi_"+str(i)] = phi_list[i+2]-phi_list[i]
        return df_dct

    def calculate_true_dynamic_modes(self):
        """
        based on: https://www.sciencedirect.com/science/article/pii/S0006349507712870#bib16
        and: https://pubs.acs.org/doi/pdf/10.1021/ma00152a014

        adds true modes to observables
        """

        from numpy.linalg import inv
        from scipy.optimize import newton
        from scipy.integrate import quad

        alphan_list = []
        for n in range(0, 50):
            alphan0 = (n+1/2)*np.pi  # approximation
            alphan = newton(lambda x: np.cos(x)*np.cosh(x)-1, x0=alphan0)
            alphan_list.append(alphan)

        def ortho_basis(s, L, n):
            alphan = (n+1/2)*np.pi  # approximation
            alphan = alphan_list[int(n)]  # exact
            if n % 2 == 1:  # uneven
                return 1/np.sqrt(L)*(np.cos(alphan*s/L)/np.cos(alphan/2)+np.cosh(alphan*s/L)/np.cosh(alphan/2))
            if n % 2 == 0:  # even
                return 1/np.sqrt(L)*(np.sin(alphan*s/L)/np.sin(alphan/2)+np.sinh(alphan*s/L)/np.sinh(alphan/2))

        def ortho_basis_derivative(s, L, n):
            alphan = (n+1/2)*np.pi  # approximation
            alphan = alphan_list[int(n)]

            if n % 2 == 1:  # uneven
                return -alphan/L**(3/2)*(-np.sin(alphan*s/L)/np.cos(alphan/2)+np.sinh(alphan*s/L)/np.cosh(alphan/2))
            if n % 2 == 0:  # even
                return alphan/L**(3/2)*(np.cos(alphan*s/L)/np.sin(alphan/2)+np.cosh(alphan*s/L)/np.sinh(alphan/2))

        def dst_basis(s, L, n):
            return np.sin(n*np.pi*(s+L/2)/L)

        def dct_basis(s, L, n):
            # according to weitz
            return np.sqrt(2/L)*np.cos(n*np.pi*(s+L/2)/L)

        def M_normal_to_dct_ij(i, j, L):

            def f(s): return dct_basis(s, L, i)*ortho_basis_derivative(s, L, j)
            return np.round(quad(f, -L/2, L/2)[0], 6)

        def M_normal_to_dct(L, nmax):

            M = np.zeros((nmax, nmax))
            for i in range(1, nmax+1):
                for j in range(1, nmax+1):
                    M[i-1, j-1] = M_normal_to_dct_ij(i, j, L)

            return M

        def M_dct_to_normal(L, nmax):
            M = M_normal_to_dct(L, nmax)
            return inv(M)

        obs = self.df_observables
        obs_columns = obs.columns

        dct_names = [i for i in obs_columns if i.startswith('dct_')]
        dct_names.sort(key=self.natural_keys)

        rows = obs[dct_names[1:]].values

        Minv = M_dct_to_normal(10, len(rows[0]))

        normal = np.array([np.dot(Minv, row) for row in rows]).transpose()

        for i in range(0, len(normal)):
            self.df_observables["true_mode_"+str(i+1)] = normal[i]
