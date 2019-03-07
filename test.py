import multiprocessing as mp
import itertools as it
import pickle
import gzip

import emoji
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import re
import cvxopt

import model_specific_functions
import config

color_set = config.Color()


class Result(object):
    def __init__(self, result_dict: dict, obj_value: float, success: bool, minimal_obj_value: float):
        self.result_dict = result_dict
        self.obj_value = obj_value
        self.success = success
        self.minimal_obj_value = minimal_obj_value

    def __repr__(self):
        return "Result: {}\nObjective value: {}\nSuccess: {}\nMinimal objective value: {}".format(
            self.result_dict, self.obj_value, self.success, self.minimal_obj_value)


def emoji_test():
    emoji_count = 0
    line = "hello 👩🏾‍🎓 emoji hello 👨‍👩‍👦‍👦 how are 😊 you today🙅🏽🙅🏽"
    print(line.encode())
    for word in line:
        if word in emoji.UNICODE_EMOJI:
            emoji_count += 1

    print(emoji_count)


def surrounding_circle():
    def make_circle(p_xloc, q_xloc, s_yloc):
        if p_xloc > q_xloc:
            swap = q_xloc
            q_xloc = p_xloc
            p_xloc = swap
        pq_dist = q_xloc - p_xloc
        an = np.linspace(0, 2 * np.pi, 100)
        circle_x_loc = np.cos(an)
        circle_y_loc = np.sin(an)
        p_circle = [circle_x_loc * pq_dist + p_xloc, circle_y_loc * pq_dist]
        q_circle = [circle_x_loc * pq_dist + q_xloc, circle_y_loc * pq_dist]
        s_xloc = np.arctan(s_yloc / pq_dist)

        fig, ax = plt.subplots()
        ax.plot(p_circle[0], p_circle[1])
        ax.plot(q_circle[0], q_circle[1])


def scarlett_function():
    def exp_sample(_lamb):
        return -1 * np.log(1 - np.random.random()) / _lamb

    def time_sample(_lamb):
        return exp_sample(_lamb) + exp_sample(_lamb)

    class Event(object):
        def __init__(self, customer_id, event_type, time_point):
            self.id = customer_id
            self.event_type = event_type
            self.time_point = time_point

    lamb_entry = 1
    lamb_service = 2.5
    import queue
    q = queue.Queue()
    event_list = []
    total_num_list = []
    wait_time_list = []
    service_time_for_each_custom = []
    current_arrival = exp_sample(lamb_entry)
    next_arrival_custom = 1
    current_custom = 0
    current_service_start = current_arrival
    next_arrival = current_arrival + exp_sample(lamb_entry)
    event_list.append(Event(current_custom, "arrival", current_arrival))
    while current_custom < 10 or not q.empty():
        service_time_for_each_custom.append(time_sample(lamb_service))
        current_service_end = current_service_start + service_time_for_each_custom[current_custom]
        if next_arrival < current_service_end:
            event_list.append(Event(next_arrival_custom, "arrival", next_arrival))
        while next_arrival < current_service_end:
            q.put((next_arrival_custom, next_arrival))
            next_arrival += exp_sample(lamb_entry)
            next_arrival_custom += 1
            if next_arrival < current_service_end:
                event_list.append(Event(next_arrival_custom, "arrival", next_arrival))

        total_num_list.append(q.qsize())
        event_list.append(Event(current_custom, "departure", current_service_end))

        if q.empty():
            current_arrival = next_arrival
            current_service_start = next_arrival
            current_custom = next_arrival_custom
            event_list.append(Event(next_arrival_custom, "arrival", next_arrival))
            next_arrival += exp_sample(lamb_entry)
            next_arrival_custom += 1
        else:
            current_custom, current_arrival = q.get(block=False)
            current_service_start = current_service_end

        wait_time_list.append(current_service_start - current_arrival)

    service_time_for_each_custom.append(time_sample(lamb_service))
    for event in event_list:
        print("Custom: {event.id}\t\tEvent type: {event.event_type}\t\tEvent time: {event.time_point}".format(
            event=event))

    x_t_num = [0]
    y_num = [0]
    total_num = 0
    total_accumulative_time = 0
    last_event_time = 0
    for event in event_list:
        x_t_num.append(event.time_point)
        x_t_num.append(event.time_point)
        if event.event_type == "arrival":
            y_num.append(total_num)
            y_num.append(total_num + 1)
            total_num += 1
        else:
            y_num.append(total_num)
            y_num.append(total_num - 1)
            total_num -= 1

    x_t_total_time = [0]
    y_total_time = [0]
    for event in event_list:
        if event.event_type == "arrival":
            if total_accumulative_time > 1e-10:
                total_accumulative_time -= event.time_point - last_event_time
            x_t_total_time.append(event.time_point)
            y_total_time.append(total_accumulative_time)
            total_accumulative_time += service_time_for_each_custom[event.id]
            x_t_total_time.append(event.time_point)
            y_total_time.append(total_accumulative_time)
        else:
            total_accumulative_time -= event.time_point - last_event_time
            x_t_total_time.append(event.time_point)
            y_total_time.append(total_accumulative_time)
        last_event_time = event.time_point

    fig, ax = plt.subplots()
    x = np.arange(len(total_num_list))
    ax.scatter(x, total_num_list)
    fig, ax = plt.subplots()
    ax.plot(x_t_num, y_num)
    fig, ax = plt.subplots()
    ax.scatter(x, wait_time_list)
    fig, ax = plt.subplots()
    ax.plot(x_t_total_time, y_total_time)
    plt.show()


def dynamic_range_function_record():
    model_name = "model1"
    output_direct = "{}/{}".format(constant_set.new_output_direct, model_name)

    data_collection_kwargs = {
        'label_list': ["glucose"], 'mouse_id_list': ['M1'],
        'source_tissue_marker': constant_set.liver_marker, 'sink_tissue_marker': constant_set.heart_marker}
    model_mid_data_dict = data_loader_rabinowitz(mid_data_loader_model1234, data_collection_kwargs)

    hook_in_each_iteration = result_processing_each_iteration_model12
    hook_after_all_iterations = final_result_processing_and_plotting_model12
    model_construction_func = model1_construction
    parameter_construction_func = dynamic_range_model12

    complete_flux_list = ['F{}'.format(i + 1) for i in range(10)] + ['G{}'.format(i + 1) for i in range(9)] + \
                         ['Fcirc_glc', 'Fcirc_lac']
    complete_flux_dict = {var: i for i, var in enumerate(complete_flux_list)}
    constant_flux_dict = {'Fcirc_glc': 150.9, 'Fcirc_lac': 374.4, 'F10': 100}

    min_flux_value = 1
    max_flux_value = 5000
    optimization_repeat_time = 10
    obj_tolerance = 0.1
    f1_range = [1, 150]
    g2_range = [1, 150]
    if test_running:
        f1_num = 51
        f1_display_interv = 50
        g2_num = 51
        g2_display_interv = 50
    else:
        f1_num = 1500
        f1_display_interv = 250
        g2_num = 1500
        g2_display_interv = 250

    # model12
    output_parameter_dict = {
        'model_name': model_name,
        'output_direct': output_direct,
        'model_mid_data_dict': model_mid_data_dict,
        'constant_flux_dict': constant_flux_dict,
        'complete_flux_dict': complete_flux_dict,
        'optimization_repeat_time': optimization_repeat_time,
        'min_flux_value': min_flux_value,
        'max_flux_value': max_flux_value,
        'obj_tolerance': obj_tolerance,

        'f1_num': f1_num,
        'f1_range': f1_range,
        'f1_display_interv': f1_display_interv,
        'g2_num': g2_num,
        'g2_range': g2_range,
        'g2_display_interv': g2_display_interv,

        'parameter_construction_func': parameter_construction_func,
        'model_construction_func': model_construction_func,
        'hook_in_each_iteration': hook_in_each_iteration,
        'hook_after_all_iterations': hook_after_all_iterations
    }

    # model34
    output_parameter_dict = {
        'model_name': model_name,
        'output_direct': output_direct,
        'model_mid_data_dict': model_mid_data_dict,
        'constant_flux_dict': constant_flux_dict,
        'complete_flux_dict': complete_flux_dict,
        'optimization_repeat_time': optimization_repeat_time,
        'min_flux_value': min_flux_value,
        'max_flux_value': max_flux_value,
        'obj_tolerance': obj_tolerance,

        'total_point_num': total_point_num,
        'free_fluxes_name_list': free_fluxes_name_list,
        'free_fluxes_range_list': free_fluxes_range_list,
        'ternary_sigma': ternary_sigma,
        'ternary_resolution': ternary_resolution,

        'parameter_construction_func': parameter_construction_func,
        'model_construction_func': model_construction_func,
        'hook_in_each_iteration': hook_in_each_iteration,
        'hook_after_all_iterations': hook_after_all_iterations
    }
    return output_parameter_dict


def plot_test():
    brain_marker = 'Br'
    heart_marker = 'Ht'
    muscle_marker = 'SkM'
    liver_marker = 'Lv'
    tissue_marker = heart_marker

    with open("./Figures/data/valid_matrix_{}_{}".format(liver_marker, tissue_marker), 'rb') as f_in:
        valid_matrix = pickle.load(f_in)
    with open("./Figures/data/glucose_contri_matrix_{}_{}".format(liver_marker, tissue_marker), 'rb') as f_in:
        glucose_contri_matrix = pickle.load(f_in)

    x1_num = 300
    x1_limit = [22, 30]
    x1_range = np.linspace(*x1_limit, x1_num + 1)
    x2_num = 6000
    x2_limit = [0, 155]
    x2_range = np.linspace(*x2_limit, x2_num + 1)

    plot_size = (20, 7)
    dpi = 150
    fig, ax = plt.subplots(figsize=plot_size, dpi=dpi)
    im = ax.imshow(valid_matrix)
    ax.set_xlim([0, x2_num])
    ax.set_ylim([0, x1_num])
    x_tick = ax.get_xticks()
    y_tick = ax.get_yticks()
    x_tick_in_range = x_tick[x_tick <= x2_num]
    y_tick_in_range = y_tick[y_tick <= x1_num]
    ax.set_xticks(x_tick_in_range)
    ax.set_yticks(y_tick_in_range)
    ax.set_xticklabels(np.around(x2_range[np.array(x_tick_in_range, dtype='int')]))
    ax.set_yticklabels(np.around(x1_range[np.array(y_tick_in_range, dtype='int')]))
    fig.savefig("./Figures/model3/dynamic_range_{}_{}_100_hf.png".format(liver_marker, tissue_marker), dpi=fig.dpi)

    fig, ax = plt.subplots()
    bin_num = 200
    sample_for_hist = glucose_contri_matrix.reshape([-1])
    sample_for_hist = sample_for_hist[~np.isnan(sample_for_hist)]
    im = ax.hist(sample_for_hist, bins=bin_num)
    ax.set_xlim([0, 1])
    # ax.set_xticks(x_tick)
    # ax.set_yticks(y_tick)
    # x_tick_label = np.around(x2_range[np.array(x_tick, dtype='int')])
    # y_tick_label = np.around(x1_range[np.array(y_tick, dtype='int')])
    # ax.set_xticklabels(x_tick_label)
    # ax.set_yticklabels(y_tick_label)

    fig, ax = plt.subplots(figsize=plot_size, dpi=dpi)
    im = ax.imshow(glucose_contri_matrix, cmap='cool')
    ax.set_xlim([0, x2_num])
    ax.set_ylim([0, x1_num])
    x_tick = ax.get_xticks()
    y_tick = ax.get_yticks()
    ax.set_xticks(x_tick)
    ax.set_yticks(y_tick)
    x_tick_in_range = x_tick[x_tick < x2_num]
    y_tick_in_range = y_tick[y_tick < x1_num]
    ax.set_xticks(x_tick_in_range)
    ax.set_yticks(y_tick_in_range)
    x_tick_label = np.around(x2_range[np.array(x_tick_in_range, dtype='int')])
    y_tick_label = np.around(x1_range[np.array(y_tick_in_range, dtype='int')])
    ax.set_xticklabels(x_tick_label)
    ax.set_yticklabels(y_tick_label)
    fig.savefig("./Figures/model3/glucose_ratio_{}_{}_100_hf.png".format(liver_marker, tissue_marker), dpi=fig.dpi)
    fig, ax = plt.subplots()
    cbar = plt.colorbar(im, ax=ax)
    ax.remove()
    cbar.ax.set_ylabel('Glucose Contribution', rotation=-90, va="bottom")
    fig.savefig("./Figures/model3/glucose_ratio_cbar.png", dpi=fig.dpi)

    plt.show()


def hmm_generator(transition_matrix, emission_matrix, initial_state, time_length):
    hidden_state_num = transition_matrix.shape[0]
    observed_state_num = emission_matrix.shape[1]
    observation_sequence = np.zeros(time_length, dtype=int)
    hidden_state_sequence = np.zeros(time_length, dtype=int)
    hidden_state_sequence[0] = initial_state
    observation_sequence[0] = np.random.choice(
        np.arange(observed_state_num), p=emission_matrix[hidden_state_sequence[0], :])
    for i in range(1, time_length):
        hidden_state_sequence[i] = np.random.choice(
            np.arange(hidden_state_num), p=transition_matrix[hidden_state_sequence[i - 1], :])
        observation_sequence[i] = np.random.choice(
            np.arange(observed_state_num), p=emission_matrix[hidden_state_sequence[i], :])
    return observation_sequence


def hmm_test():
    transition_matrix = np.array([[0.3, 0.7], [0.8, 0.2]])
    emission_matrix = np.array([[0.1, 0.9], [0.8, 0.2]])
    initial_state = 1
    observation_sequence = hmm_generator(transition_matrix, emission_matrix, initial_state, 10)
    print(observation_sequence)


def violin_plot(data_dict, color_dict=None):
    fig, ax = plt.subplots()
    data_list_for_violin = data_dict.values()
    tissue_label_list = data_dict.keys()
    x_axis_position = np.arange(1, len(tissue_label_list) + 1)

    parts = ax.violinplot(data_list_for_violin, showmedians=True, showextrema=True)
    if color_dict is not None:
        color_list = [color_dict[key] for key in tissue_label_list]
        parts['cmaxes'].set_edgecolor(color_list)
        parts['cmins'].set_edgecolor(color_list)
        parts['cbars'].set_edgecolor(color_list)
        parts['cmedians'].set_edgecolor(color_list)
        for pc, color in zip(parts['bodies'], color_list):
            pc.set_facecolor(color)
            pc.set_alpha(0.3)
    dash_color = np.array([255, 126, 22]) / 255
    ax.axhline(0.5, linestyle='--', color=dash_color)
    ax.set_ylim([-0.1, 1.1])
    ax.set_xticks(x_axis_position)
    ax.set_xticklabels(tissue_label_list)
    return fig, ax


def violin_test():
    data_dict = {
        'a': [0, 1, 0.2, 0.1, 0.3, 0.2, 0.4],
        'b': [0, 1, 0.4, 0.5, 0.5, 0.5, 0.5],
        'c': [0, 1, 0.7, 0.8, 0.9, 0.8, 0.7]
    }
    color_dict = {
        'a': np.array([112, 48, 160]) / 255,
        'b': np.array([21, 113, 177]) / 255,
        'c': np.array([251, 138, 68]) / 255
    }
    # violin_plot(data_dict, color_dict)
    violin_plot({'c': data_dict['c']}, {'c': color_set.purple})
    plt.show()


# min (Ax - b) ^ 2, while sum(x) = 1, x >= 0
# object function: (Ax - b) ^ 2 = x^T (A^T @ A) x - 2 (A^T @ b)^T @ x + b^T @ b
# standard form: min 1/2 x^T @ P @ x + q^T x, st G @ x <= h, A @ x = b
# P = A^T @ A, q = A^T @ b, G = -1 * I, h = 0, A = [1, 1, 1,... 1], b = 1
def solve_multi_ratios(source_vector_list, target_vector, ratio_lb, ratio_ub):
    var_num = len(source_vector_list)
    cvx_matrix = cvxopt.matrix
    raw_matrix_a = np.array(source_vector_list, dtype='float64').transpose()
    raw_vector_b = target_vector.reshape([-1, 1])
    matrix_p = cvx_matrix(raw_matrix_a.T @ raw_matrix_a)
    vector_q = -cvx_matrix(raw_matrix_a.T @ raw_vector_b)

    matrix_g = cvx_matrix(np.vstack([-1 * np.identity(var_num), np.identity(var_num)]))
    matrix_h = cvx_matrix(np.vstack([np.ones([var_num, 1]) * ratio_lb, np.ones([var_num, 1]) * ratio_ub]))
    matrix_a = cvx_matrix(np.ones([1, var_num]))
    matrix_b = cvx_matrix(np.ones([1, 1]))

    result = cvxopt.solvers.qp(matrix_p, vector_q, matrix_g, matrix_h, matrix_a, matrix_b)
    result_array = np.array(result['x'])
    return result_array.reshape([-1])


def cvxopt_test():
    test_source_vector_list = [np.array([0.9, 0.1, 0]), np.array([0.1, 0.9, 0])]
    test_target_vector = np.array([1, 0, 0])
    ratio_lb = 0.1
    ratio_ub = 0.9
    result = solve_multi_ratios(test_source_vector_list, test_target_vector, ratio_lb, ratio_ub)
    print(result)


def convex_test():
    def func(_a_vector, _b_vector, _alpha_vector, _f1_value, _f2_value):
        _target_mid = (_a_vector * _f1_value + _b_vector * _f2_value) / (_f1_value + _f2_value)
        return -np.sum(_alpha_vector * np.log(_target_mid))

    a_vector = np.array([0.9, 0.05, 0.05])
    b_vector = np.array([0.05, 0.05, 0.9])
    alpha_vector = np.array([0.225, 0.05, 0.725])
    f1 = np.arange(0.1, 5, 0.01)
    f2 = np.arange(0.1, 5, 0.01)
    f1_mesh, f2_mesh = np.meshgrid(f1, f2)
    z = np.zeros_like(f1_mesh)
    for x in range(len(f1)):
        for y in range(len(f2)):
            z[x, y] = func(a_vector, b_vector, alpha_vector, f1[x], f2[y])

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(
        f1_mesh, f2_mesh, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def test_single_core(var_parameter_list, index, result_queue, other_parameter_dict):
    current_result_list = []
    current_hook_result_list = []
    count = 0
    for var_parameter in var_parameter_list:
        new_result = other_parameter_dict['add'] + var_parameter
        current_result_list.append(new_result)
        current_hook_result_list.append(new_result)
        count += 1
        if count % 100 == 0:
            print(count)
    result_queue.put((index, current_result_list, current_hook_result_list))


def multiprocess_test():
    other_parameter_dict = {'add': 5}
    parallel_num = 5
    var_parameter_list = list(range(100))

    q = mp.Queue()
    total_iter_num = len(var_parameter_list)
    sub_list_length = int(np.ceil(total_iter_num / parallel_num))
    process_list = []
    for i in range(parallel_num):
        start = sub_list_length * i
        end = min(sub_list_length * (i + 1), total_iter_num)
        this_var_parameter_list = var_parameter_list[start:end]
        p = mp.Process(target=test_single_core, args=(
            this_var_parameter_list, i, q, other_parameter_dict))
        p.start()
        process_list.append(p)

    tmp_result_list = [0] * parallel_num
    tmp_hook_result_list = [0] * parallel_num
    complete_process_count = 0
    while complete_process_count < parallel_num:
        index, current_result_list, current_hook_result_list = q.get()
        tmp_result_list[index] = current_result_list
        tmp_hook_result_list[index] = current_hook_result_list
        process_list[index].join()
        complete_process_count += 1
    result_list = []
    hook_result_list = []
    for current_result_list, current_hook_result_list in zip(tmp_result_list, tmp_hook_result_list):
        result_list.extend(current_result_list)
        hook_result_list.extend(current_hook_result_list)
    print(result_list)
    print(hook_result_list)


count_q = mp.Queue()

# from new_model_main import plot_ternary_scatter
import scipy.interpolate
import ternary


def ternary_function_test():
    data_matrix = np.random.random([100, 3])
    norm_data_matrix = data_matrix / np.sum(data_matrix, 1).reshape([-1, 1])
    plot_ternary_scatter(norm_data_matrix)
    plt.show()


def ternary_test():
    # scale = 1
    # figure, tax = ternary.figure(scale=scale)

    # Draw Boundary and Gridlines
    # tax.boundary(linewidth=2.0)
    # tax.gridlines(color="blue", multiple=0.1)

    # Set Axis labels and Title
    # fontsize = 20
    # tax.set_title("Various Lines", fontsize=20)
    # tax.left_axis_label("Left label $\\alpha^2$", fontsize=fontsize)
    # tax.right_axis_label("Right label $\\beta^2$", fontsize=fontsize)
    # tax.bottom_axis_label("Bottom label $\\Gamma - \\Omega$", fontsize=fontsize)

    # Draw lines parallel to the axes
    # tax.horizontal_line(16)
    # tax.left_parallel_line(10, linewidth=2., color='red', linestyle="--")
    # tax.right_parallel_line(20, linewidth=3., color='blue')
    # Draw an arbitrary line, ternary will project the points for you
    # p1 = (12, 8, 10)
    # p2 = (2, 26, 2)
    # tax.line(p1, p2, linewidth=3., marker='s', color='green', linestyle=":")
    # tax.scatter([(0.13, 0.33, 0.54)])

    # tax.ticks(axis='lbr', multiple=0.1, linewidth=1, tick_formats="%.1f")

    # tax.show()

    tri_data_matrix = np.array([[0.8, 0.1, 0.1]])
    save_path = "new_models/model3/ternary_figure.png"
    count_density_dist(tri_data_matrix, save_path=save_path, sigma=0.1, bin_num=2 ** 7)

    # import random
    #
    # def generate_random_heatmap_data(scale=5):
    #     from ternary.helpers import simplex_iterator
    #     d = dict()
    #     for (i, j, k) in simplex_iterator(scale):
    #         d[(i, j)] = random.random()
    #     return d
    #
    # scale = 20
    # d = generate_random_heatmap_data(scale)
    # figure, tax = ternary.figure(scale=scale)
    # tax.heatmap(d, style="h")
    # tax.boundary()
    # tax.set_title("Heatmap Test: Hexagonal")
    # tax.show()


from ternary.helpers import simplex_iterator


# Each row of data matrix is a point in triple tuple
# In cartesian cor, the left bottom corner of triangle is the origin.
# The scale of all triangle points is 1.
# Order of ternary cor: x1: bottom (to right) x2: right (to left) x3: left (to bottom)
def count_density_dist(tri_data_matrix, sigma: float = 1, bin_num: int = 2 ** 8, save_path=None):
    sqrt_3 = np.sqrt(3)

    def standard_2dnormal(x, y, _sigma):
        return np.exp(-0.5 / _sigma ** 2 * (x ** 2 + y ** 2)) / (2 * np.pi * _sigma ** 2)

    # Each row is the cartesian cor.
    def tri_to_car(input_data_matrix):
        y_value = input_data_matrix[:, 1] * sqrt_3 / 2
        x_value = input_data_matrix[:, 0] + y_value / sqrt_3
        return np.vstack([x_value, y_value]).T

    def car_to_tri(input_data_matrix):
        y_value = input_data_matrix[:, 1]
        x2_value = y_value / (sqrt_3 / 2)
        x1_value = input_data_matrix[:, 0] - y_value / sqrt_3
        # x3_value = 1 - x1_value - x2_value
        # x3_value[x3_value < 0] = 0
        return np.vstack([x1_value, x2_value]).T

    def gaussian_kernel_generator(_bin_num, _sigma):
        x = np.linspace(0, 1, _bin_num) - 0.5
        y = np.linspace(0, 1, _bin_num) - 0.5
        X, Y = np.meshgrid(x, y)
        gaussian_kernel = standard_2dnormal(X, Y, _sigma)
        return np.rot90(gaussian_kernel)

    def bin_car_data_points(_car_data_matrix, _bin_num):
        histogram, _, _ = np.histogram2d(
            _car_data_matrix[:, 0], _car_data_matrix[:, 1], bins=np.linspace(0, 1, _bin_num + 1))
        return histogram

    def complete_tri_set_interpolation(_location_list, _value_list, _scale):
        result_tri_array = np.array(list(simplex_iterator(_scale))) / _scale
        result_car_array = tri_to_car(result_tri_array)
        result_value_array = scipy.interpolate.griddata(
            np.array(location_list), np.array(value_list), result_car_array, method='cubic')
        target_dict = {}
        for (i, j, k), result_value in zip(simplex_iterator(bin_num), result_value_array):
            target_dict[(i, j)] = result_value
        return target_dict

    car_data_matrix = tri_to_car(tri_data_matrix)
    data_bin_matrix = bin_car_data_points(car_data_matrix, bin_num)
    gaussian_kernel_matrix = gaussian_kernel_generator(bin_num, sigma)
    car_blurred_matrix = scipy.signal.convolve2d(data_bin_matrix, gaussian_kernel_matrix, mode='same')
    x_axis = y_axis = np.linspace(0, 1, bin_num)
    location_list = []
    value_list = []
    for x_index, x_value in enumerate(x_axis):
        for y_index, y_value in enumerate(y_axis):
            location_list.append([x_value, y_value])
            value_list.append(car_blurred_matrix[x_index, y_index])
    complete_density_dict = complete_tri_set_interpolation(location_list, value_list, bin_num)
    fig, tax = ternary.figure(scale=bin_num)
    tax.heatmap(complete_density_dict, cmap='Blues', style="h")
    tax.boundary(linewidth=1.0)
    tick_labels = list(np.linspace(0, bin_num, 11) / bin_num)
    tax.ticks(axis='lbr', ticks=tick_labels, linewidth=1, tick_formats="")

    tax.clear_matplotlib_ticks()
    plt.tight_layout()
    if save_path:
        print(save_path)
        fig.savefig(save_path, dpi=fig.dpi)
    tax.show()


def shuffle_test():
    total_point_num = 100
    free_fluxes_range_list = [[0, 1], [0, 10], [0, 100]]
    point_num_each_axis = np.round(np.power(total_point_num, 1 / len(free_fluxes_range_list))).astype('int')

    free_flux_raw_list = [
        np.linspace(*free_fluxes_range, total_point_num) for free_fluxes_range in free_fluxes_range_list]
    for row_index in range(len(free_fluxes_range_list)):
        np.random.shuffle(free_flux_raw_list[row_index])
    free_flux_value_list_random_sample = np.array(free_flux_raw_list).T

    free_fluxes_sequence_list = [
        np.linspace(*flux_range, point_num_each_axis) for flux_range in free_fluxes_range_list]
    free_flux_value_list_iteration = np.array(list(it.product(*free_fluxes_sequence_list)))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plot = ax.scatter(
        free_flux_value_list_random_sample[:, 0], free_flux_value_list_random_sample[:, 1],
        free_flux_value_list_random_sample[:, 2])
    plot = ax.scatter(
        free_flux_value_list_iteration[:, 0], free_flux_value_list_iteration[:, 1],
        free_flux_value_list_iteration[:, 2])
    plt.show()


def read_test():
    data_dict_file_path = "C:/Data/PhD/LocasaleLab/Scripts/lactate_exchange/new_models/model7"
    data_dict_file_path = "{}/output_data_dict.gz".format(data_dict_file_path)
    with gzip.open(data_dict_file_path, 'rb') as f_in:
        data_dict = pickle.load(f_in)
    contribution_matrix = data_dict['contribution_matrix']
    print(np.count_nonzero(contribution_matrix[:, 0] > 0.5))
    print(np.count_nonzero(contribution_matrix[:, 1] > 0.5))
    print(np.count_nonzero(contribution_matrix[:, 2] > 0.5))
    # valid_matrix = data_dict['valid_matrix']
    # print(np.count_nonzero(valid_matrix == 0))
    # print(valid_matrix.size)


def read_test2():
    data_dict_file_path = "C:/Data/PhD/LocasaleLab/Scripts/lactate_exchange/new_models/model1_parameter_sensitivity"
    data_dict_file_path = "{}/output_data_dict.gz".format(data_dict_file_path)
    with gzip.open(data_dict_file_path, 'rb') as f_in:
        data_dict = pickle.load(f_in)
    well_fit_glucose_contri_dict = data_dict['well_fit_glucose_contri_dict']
    print(len(well_fit_glucose_contri_dict['mid']))
    sample_type_list = ['mid', 'Fcirc_glc', 'Fcirc_lac', 'F10']
    well_fit_glucose_contri_array_dict = {
        sample_type: [] for sample_type in sample_type_list}
    well_fit_median_contri_dict = {
        sample_type: [] for sample_type in sample_type_list}
    for sample_type, sample_contri_list in well_fit_glucose_contri_dict.items():
        for sample_index, contri_list in enumerate(sample_contri_list):
            new_array = np.array(contri_list)
            if len(new_array) == 0:
                continue
            well_fit_glucose_contri_array_dict[sample_type].append(new_array)
            well_fit_median_contri_dict[sample_type].append(np.median(new_array))
        print(len(well_fit_median_contri_dict[sample_type]))
    model_specific_functions.common_functions.plot_violin_distribution(
        {'mid': well_fit_median_contri_dict['mid']},
        {'mid': color_set.purple},
        # save_path="{}/glucose_contribution_violin_parameter_sensitivity_{}.png".format(
        #     output_direct, sample_type)
    )
    plt.show()


def main():
    # emoji_test()
    # surrounding_circle()
    # scarlett_function()
    # plot_test()
    # hmm_test()
    violin_test()
    # cvxopt_test()
    # convex_test()
    # multiprocess_test()
    # ternary_function_test()
    # ternary_test()
    # shuffle_test()
    # read_test2()


if __name__ == '__main__':
    main()
