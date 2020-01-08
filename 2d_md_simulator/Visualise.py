import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arrow
import subprocess


class Visualise(object):

    def __init__(self, simulation):
        self.simulation_path = simulation.path
        self.data = simulation.data

    def draw_state(self, t, xmin, xmax, ymin, figsize=(16, 9), show_psi=False, show_trap=False):
        """
        16 x 9 inches is good figsize
        as this with 120dpi translates to 1920x1080 a standard video hd format
        for golden ratio
        ratio_height_to_width=0.618
        this is slightly less broad then standard dia presentation style 9:16
        which has ratio_height_to_width=0.5625
        """
        x = self.data.df_particle.query("time==@t")["x"].values
        y = self.data.df_particle.query("time==@t")["y"].values

        try:
            x_stat = self.data.df_trap.query("time==@t")["x_stat"].values
            y_stat = self.data.df_trap.query("time==@t")["y_stat"].values
            x_mob = self.data.df_trap.query("time==@t")["x_mob"].values
            y_mob = self.data.df_trap.query("time==@t")["y_mob"].values
            s_trap = (x_stat, y_stat)
            m_trap = (x_mob, y_mob)
        except:
            s_trap = (x[0], y[0])
            m_trap = (x[-1], y[-1])
            pass

        fig = plt.figure(figsize=figsize)
        ratio_height_to_width = figsize[1]/figsize[0]
        # initialize axis, important: set the aspect ratio to equal
        # axis is designed such that it fills whole fig
        ax = fig.add_axes([0, 0, 1, 1], aspect='equal')
        ax.axis('off')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymin+(xmax-xmin)*ratio_height_to_width)
        # loop through all triplets of x-,y-coordinates and radius and
        # plot a circle for each:

        for xi, yi in zip(x, y):
            ax.add_artist(Circle(xy=(xi, yi),
                                 radius=0.5, alpha=1, color="b"))
        if show_trap:
            ax.add_artist(Circle(xy=s_trap,
                                 radius=0.1, alpha=1, color="r"))
            ax.add_artist(Circle(xy=m_trap,
                                 radius=0.1, alpha=1, color="r"))

        if show_psi:
            psi = self.data.df_particle.query("time==@t")["psi"].values
            for xi, yi, psi_i in zip(x, y, psi):
                (xtail, ytail) = (xi-0.5*np.cos(psi_i), yi-0.5*np.sin(psi_i))
                (dx, dy) = (np.cos(psi_i), np.sin(psi_i))
                ax.add_artist(Arrow(x=xtail, y=ytail, dx=dx, dy=dy, width=0.25, color="r"))

        show_linkers = False
        if show_linkers:
            [N_width, N_height, M_width, M_height] = self.data.df_parameters.loc["value",
                                                                                 ["N_width", "N_height", "M_width", "M_height"]].values

        return fig

    def movie(self,
              xmin, xmax, ymin,
              temp_save_folder=None,
              frames=100,
              draw_figures=True,
              animate=True,
              remove_images=True,
              fps=24,
              frame_shape="square",
              show_psi=False,
              show_trap=False
              ):
        """
        frame_shape can be square or FHD
        if square images are 9 by 9 inch and 1080x1080 pixels
        if FHD (full HD) images are 16 by 9 inch and 1920x1080 pixels
        """

        all_times = self.data.df_particle["time"].unique()
        nmb_of_times = len(all_times)
        dt_between_frames = int(nmb_of_times/frames)
        selected_times = all_times[::dt_between_frames]

        if frame_shape == "square":
            figsize = (9, 9)
            dpi = 120
            videosize = "1080x1080"
        elif frame_shape == "FHD":
            figsize = (16, 9)
            dpi = 120
            videosize = "1920x1080"

        if temp_save_folder is not None:
            path_series = temp_save_folder + '/images/'
            if not os.path.exists(path_series):
                os.makedirs(path_series)

        else:
            path_series = self.simulation_path + '/images/'
            if not os.path.exists(path_series):
                os.makedirs(path_series)

        if draw_figures == True:
            plt.ioff()

            for i, t in enumerate(selected_times):
                # print(t)

                fig = self.draw_state(t, xmin, xmax, ymin, figsize=figsize,
                                      show_psi=show_psi, show_trap=show_trap)
                fig.savefig(path_series+"\\"+str(i), dpi=dpi)
                plt.close()

        if animate == True:
            os.chdir(path_series)
            videoname = self.simulation_path+"/video.mp4"
            if os.path.exists(videoname):
                os.remove(videoname)
            print(videoname)
            try:
                subprocess.call(["ffmpeg.exe", "-r", str(fps), "-f", "image2", "-i", "%d.png",
                                 "-s", videosize, "-vcodec", "libx264", "-crf", "25", "-pix_fmt", "yuv420p",
                                 videoname])

            except OSError as e:
                if e.errno == os.errno.ENOENT:
                    print("did you install ffmpeg?")
                    # handle file not found error.
                else:
                    # Something else went wrong while trying to run `wget`
                    print("something went wrong running ffmpeg")
                    raise

        if remove_images == True:
            import shutil
            os.chdir(self.simulation_path)
            shutil.rmtree(path_series)

    def movie_big_data(self,
                       nmb_of_data_files,
                       temp_save_folder=None,
                       taking_every_nth_frames=1,
                       draw_figures=True,
                       animate=True,
                       remove_images=True,
                       fps=24,
                       frame_shape="square",
                       show_psi=False,
                       show_trap=False):

        if nmb_of_data_files == "all":
            particle_files = [s for s in os.listdir(self.data.path) if s.startswith("particle")]
            nmb_of_data_files = len(particle_files)

        if temp_save_folder is not None:
            path_series = temp_save_folder + '/images/'
            if not os.path.exists(path_series):
                os.makedirs(path_series)

        else:
            path_series = self.simulation_path + '/images/'
            if not os.path.exists(path_series):
                os.makedirs(path_series)

        image_iterator = 0
        for i in range(0, nmb_of_data_files):
            self.data.data_load(
                take_every=1,
                particle=True,
                theta=False,
                observables=False,
                observables_vs_u=False,
                string_to_array=False,
                trap=False, start_save_nmb=i, untill_save_nmb=i+1)

            subset_times = self.data.df_particle["time"].unique()

            selected_times = subset_times[::taking_every_nth_frames]
            print("now starting times:" + str(selected_times))

            if frame_shape == "square":
                figsize = (9, 9)
                dpi = 120
                videosize = "1080x1080"
            elif frame_shape == "FHD":
                figsize = (16, 9)
                dpi = 120
                videosize = "1920x1080"

            [N_width, N_height, M_width, M_height] = self.data.df_parameters.loc["value",
                                                                                 ["N_width", "N_height", "M_width", "M_height"]].values
            width = N_width*M_width
            height = N_height*M_height
            max_dimension = max(width, height)
            pad = 1
            xmin = -M_width/2
            ymin = -M_height/2

            if draw_figures == True:
                plt.ioff()

                for j, t in enumerate(selected_times):
                    # print(t)

                    fig = self.draw_state(t, xmin=xmin-pad, xmax=xmin+max_dimension+pad, ymin=ymin-pad,
                                          figsize=figsize, show_psi=show_psi, show_trap=show_trap)
                    fig.savefig(path_series+"\\"+str(image_iterator), dpi=dpi)
                    plt.close()
                    image_iterator += 1

        if animate == True:
            os.chdir(path_series)
            videoname = self.simulation_path+"/video.mp4"
            if os.path.exists(videoname):
                os.remove(videoname)
            print(videoname)
            try:
                subprocess.call(["ffmpeg.exe", "-r", str(fps), "-f", "image2", "-i", "%d.png",
                                 "-s", videosize, "-vcodec", "libx264", "-crf", "25", "-pix_fmt", "yuv420p",
                                 videoname])

            except OSError as e:
                if e.errno == os.errno.ENOENT:
                    print("did you install ffmpeg?")
                    # handle file not found error.
                else:
                    # Something else went wrong while trying to run `wget`
                    print("something went wrong running ffmpeg")
                    raise

        if remove_images == True:
            import shutil
            os.chdir(self.simulation_path)
            shutil.rmtree(path_series)
