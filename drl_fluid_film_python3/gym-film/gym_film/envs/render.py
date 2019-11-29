import param
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# from gym_film.utils.convert_reward import to_single_reward
# from matplotlib.patches import Patch

# Uses the following methods/attributes from env:
#   -  O, R (observation and reward)
#   - jets_power
#   - system_state
#   - reward, t
matplotlib.rcParams.update({'font.size': 15})

control = param.show_control
NO_LEGEND = True
show = 1
save = 0


class FilmRender():
    def __init__(self, env, PLOT_JETS=True):
        self.PLOT_JETS = PLOT_JETS
        self.env = env
        self.Ob = self.env.Ob
        self.R = self.env.R
        self.blit = False
        self.setup_plot()

    def setup_h_plot(self, plot_regions=False):
        # Plot h
        self.hlines, = self.hax.plot(np.linspace(
            0, param.L-param.dx, param.NUM), self.env.system_state[0],
            label="y = h(x)", linewidth=2.5)
        self.qlines, = self.hax.plot(np.linspace(
            0, param.L, param.NUM), self.env.system_state[1], alpha=0.5,
            label="y = q(x)", linestyle='--', linewidth=2.5)
        # Add lims on axes
        self.hax.set_xlim(param.start_h_plot, param.L)
        self.hax.set_ylim(param.hq_base_value-param.max_h,
                          param.hq_base_value+param.max_h)
        # self.hax.grid()

        if self.PLOT_JETS:
            # Plot jets
            self.setup_jet(self.hax)

            # # legend
            # legend = self.hax.legend(["y = h(x)", "y = q(x)", "jets position",
            #                 "observation space", "reward space", "jets power"],
            #                   loc='lower left', ncol=2)
            handles, labels = self.hax.get_legend_handles_labels()
            # sort both labels and handles by labels
            order = [0, 1, 4, 2, 3, 5]
            self.hax.legend([handles[idx] for idx in order], [labels[idx]
                                                              for idx in order],
                            loc="lower left", ncol=2)
            # ax = legend.axes
            # handles, labels = ax.get_legend_handles_labels()
            # # obs label
            # handles.append(Patch(facecolor='orange', edgecolor='r'))
            # labels.append("observation domain")
            # # reward label
            # handles.append(Patch(facecolor='orange', edgecolor='r'))
            # labels.append("reward domain")

            # legend._legend_box = None
            # legend._init_legend_box(handles, labels)
            # legend._set_loc(legend._loc)
            # legend.set_title(legend.get_title().get_text())
        else:
            # legend
            self.hax.legend(["y = h(x)", "y = q(x)"], loc='lower left')

        if NO_LEGEND:
            self.hax.get_legend().remove()
        # self.text = self.hax.text(1.1, 0.1, 't = '+str(int(round(float(self.env.t)))))
        self.text = self.hax.text(1.1, 0.1, 't = '+str(int(round(float(self.env.current_step*param.dt)))))

        if plot_regions:
            self.plot_regions(self.hax)

        # adding x and y labels
        no_x_label = False
        if not no_x_label:
            self.hax.set_xlabel('x')
        no_y_label = False
        if not no_y_label:
            self.hax_ylabel = self.hax.set_ylabel('h, q', labelpad=5)

        # changing color of ticks
        self.hax.tick_params(colors='black')
        no_ticks_x = False
        if no_ticks_x:
            # removing ticks
            plt.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False)  # labels along the bottom edge are off

        no_ticks_y = False
        if no_ticks_y:
            # removing ticks
            plt.tick_params(
                axis='y',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                left=False,
                labelbottom=False)  # labels along the bottom edge are off
            self.hax.set_yticklabels([])

    def plot_regions(self, ax):
        x1 = 160
        ax.axvspan(0, x1, facecolor='blue', alpha=0.1)
        ax.axvline(x=x1, ymin=0.0, ymax=1.0, color='k',
                   linestyle='--', alpha=0.3)
        x2 = 270
        ax.axvspan(x1, x2, facecolor='green', alpha=0.1)
        ax.axvline(x=x2, ymin=0.0, ymax=1.0, color='k',
                   linestyle='--', alpha=0.3)
        ax.axvspan(x2, 340, facecolor='red', alpha=0.1)

        textstr1 = "Exponential instability growth region"
        textstr2 = "Pseudo-periodic region"
        textstr3 = "Fully-developped\nchaotic region"

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        ax.text(0.09, 0.95, textstr1, transform=ax.transAxes,
                verticalalignment='top', bbox=props)
        ax.text(0.54, 0.95, textstr2, transform=ax.transAxes,
                verticalalignment='top', bbox=props)
        ax.text(0.83, 0.95, textstr3, transform=ax.transAxes,
                verticalalignment='top', bbox=props)

    def setup_jet(self, ax):
        # self.jet_plot = ax.scatter(param.jets_position*param.dx, np.zeros(len(param.jets_position)), s=100*self.env.jets_power)
        self.jet_spots = ax.scatter(
            param.jets_position*param.dx,
            [(0.9 - 0.095*(k)) for k in range(param.n_jets)],
            label='jets position', s=30)

        plt.plot([], [], label="observation domain", color="green")
        plt.plot([], [], label="reward domain", color="red")

        self.jet_rect = ax.bar(param.jets_position*param.dx,
                               self.env.jets_power, param.JET_WIDTH*param.dx, label="jets power", bottom=1)

        # show the zone where the control is done as well
        x_control_spots = np.array([self.Ob.obs_points+param.jets_position[i]
                                    for i in range(param.n_jets)]).flatten()*param.dx
        y_control_spots = np.concatenate(np.array(
            [(np.zeros(len(self.Ob.obs_points)) + 0.9 - 0.095*(k)) for k in range(param.n_jets)]))

        self.control_spots = ax.scatter(
            x_control_spots, y_control_spots, s=1)
        # shoz the zone where the reward is calculated
        x_reward_spots = np.array([self.R.obs_points_to_reward+param.jets_position[i]
                                   for i in range(param.n_jets)]).flatten()*param.dx
        y_reward_spots = np.concatenate(np.array([(np.zeros(len(
            self.R.obs_points_to_reward)) + 0.89 - 0.095*(k)) for k in range(param.n_jets)]))
        self.reward_spots = ax.scatter(
            x_reward_spots, y_reward_spots, s=1)

    def update_h_plot(self):
        self.hlines.set_ydata(self.env.system_state[0])
        self.qlines.set_ydata(self.env.system_state[1])

    def update_plot_jet(self, ax):
        # self.jet_plot.remove()
        # self.jet_plot = ax.scatter(param.jets_position * param.dx, np.zeros(len(param.jets_position)), s=100 * self.env.jets_power)
        if self.render_plot:
            for i in range(param.n_jets):
                self.jet_rect[i].set_height(self.env.jets_power[i])

    def setup_control_plot(self):
        # Plot control/h as a function of time
        self.control_ax.set_ylim(-1.5, 1.5)
        self.control_ax.set_xlim(0, param.MAX_TIMEFRAME_CONTROL_PLOT-1)

        self.x_t = np.arange(0, param.MAX_TIMEFRAME_CONTROL_PLOT)
        self.y_sensor = [0 for i in range(param.MAX_TIMEFRAME_CONTROL_PLOT)]
        self.y_control = [0 for i in range(param.MAX_TIMEFRAME_CONTROL_PLOT)]
        self.y_reward = [0 for i in range(param.MAX_TIMEFRAME_CONTROL_PLOT)]

        self.sensor_lines, = self.control_ax.plot(self.x_t, self.y_sensor)
        self.control_lines, = self.control_ax.plot(self.x_t, self.y_control)
        self.reward_lines, = self.control_ax.plot(self.x_t, self.y_reward)

        # legend
        self.control_ax.set_title(
            "Some values at x=jets_position[0] as a function of time")
        self.control_ax.legend(["y(t) = {}*h(x_jet, t)".format(param.obs_at_jet_render_param),
                                "jet power (proportion of max jet power)", "{} * reward".format(param.reward_multiplier_render)], loc='lower left')

    def update_control_plot(self):
        self.y_sensor.append(param.obs_at_jet_render_param*(
            self.env.system_state[0, param.jets_position[0]]-param.hq_base_value))
        self.y_sensor.pop(0)

        self.y_control.append(self.env.jets_power[0])
        self.y_control.pop(0)

        self.y_reward.append(param.reward_multiplier_render *
                             self.reward_process(self.env.reward))
        self.y_reward.pop(0)

        if self.render_plot:
            self.control_lines.set_ydata(self.y_control)
            self.sensor_lines.set_ydata(self.y_sensor)
            self.reward_lines.set_ydata(self.y_reward)

        # self.control_ax.set_xlim(max(0, self.env.current_step-max_timeframe), self.env.current_step)

    # setup everything - calls setup_jets and everything
    def reward_process(self, reward):
        if type(reward) is dict:
            return np.mean([reward.get(
                jet_position) for jet_position in sorted(reward.keys())])
        return reward

    def setup_plot(self):
        standard_size = {'width': 10, 'height': 3}
        divide_by = 1
        self.figure = plt.figure(figsize=(standard_size.get(
            'width')/divide_by, standard_size.get('height')/divide_by))
        self.hax = self.figure.add_subplot(1, 1, 1)
        # self.figure.subplots_adjust(wspace=0.2)
        if control:
            self.control_ax = self.figure.add_subplot(2, 1, 2)
        self.figure.subplots_adjust(hspace=1)

        # Plot h
        self.setup_h_plot()

        # Plot control
        # self.setup_control_plot()

        if self.blit:
            # cache the background
            self.haxbackground = self.figure.canvas.copy_from_bbox(
                self.hax.bbox)
            self.control_axbackground = self.figure.canvas.copy_from_bbox(
                self.control_ax.bbox)

        self.figure.canvas.draw()
        if show:
            plt.show(block=False)
        self.counter = 0
        # self.hax.set_title('t = {} \n global_reward = {}'.format(
        #     self.env.t, (to_single_reward(list(self.env.reward.values())) if param.method == '1env_1jet' else self.env.reward)))
        self.save = save
        if self.save:
            self.save_fig()

    def save_fig(self):
        if self.counter == 0 or self.counter % param.SAVE_PERIOD == 0:
            plt.tight_layout()
            plt.savefig('fig'+str(self.counter)+str(id(self.env))[:2]+'.png')
        self.counter += 1

    # update everything
    def update_plot(self):
        # self.render_plot = self.render_clock == param.RENDER_PERIOD
        self.render_plot = True

        # # Update time value
        # self.figure.suptitle('t = {} \n Reward : {}'.format(
        #     self.env.t, self.env.reward), fontsize=16)
        # self.hax.set_title('t = {} \n global_reward = {}'.format(
        #     int(round(self.env.t)), (to_single_reward(list(self.env.reward.values())) if param.method == '1env_1jet' else self.env.reward)))
        if self.save:
            self.save_fig()
        # Update data h
        self.update_h_plot()

        if self.PLOT_JETS:
            # Update jet
            self.update_plot_jet(self.hax)

        # Update control as a function of time
        if control:
            self.update_control_plot()

        self.text.set_text('t = '+str(int(round(float(self.env.current_step*param.simulation_step_time)))))

        if self.render_plot:
            if self.blit:
                # restore background
                self.figure.canvas.restore_region(self.haxbackground)
                self.figure.canvas.restore_region(self.control_axbackground)
                # redraw just the points
                self.hax.draw_artist(self.hlines)
                self.hax.draw_artist(self.qlines)
                self.control_ax.draw_artist(self.sensor_lines)
                self.control_ax.draw_artist(self.control_lines)
                self.control_ax.draw_artist(self.reward_lines)

                # fill in the axes rectangle
                self.figure.canvas.blit(self.hax.bbox)
                self.figure.canvas.blit(self.control_ax.bbox)
            else:
                # We draw here
                self.figure.canvas.draw()
                self.figure.canvas.flush_events()

            self.render_clock = 0
        self.render_clock += 1
