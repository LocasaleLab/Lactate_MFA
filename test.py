import multiprocessing as mp

import emoji
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import re
import pickle
import cvxopt


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
    violin_plot(data_dict, color_dict)
    plt.show()


# min (Ax - b) ^ 2, while sum(x) = 1, x >= 0
# object function: (Ax - b) ^ 2 = x^T (A^T @ A) x - 2 (A^T @ b)^T @ x + b^T @ b
# standard form: min 1/2 x^T @ P @ x + q^T x, st G @ x <= h, A @ x = b
# P = A^T @ A, q = A^T @ b, G = -1 * I, h = 0, A = [1, 1, 1,... 1], b = 1
def solve_multi_ratios(source_vector_list, target_vector):
    var_num = len(source_vector_list)
    cvx_matrix = cvxopt.matrix
    raw_matrix_a = np.array(source_vector_list, dtype='float64').transpose()
    raw_vector_b = target_vector.reshape([-1, 1])
    matrix_p = cvx_matrix(raw_matrix_a.T @ raw_matrix_a)
    vector_q = cvx_matrix(raw_matrix_a.T @ raw_vector_b)

    matrix_g = cvx_matrix(-1 * np.identity(var_num))
    matrix_h = cvx_matrix(np.zeros([var_num, 1]))
    matrix_a = cvx_matrix(np.ones([1, var_num]))
    matrix_b = cvx_matrix(np.ones([1, 1]))

    result = cvxopt.solvers.qp(matrix_p, vector_q, matrix_g, matrix_h, matrix_a, matrix_b)
    result_array = np.array(result['x'])
    return result_array.reshape([-1])


def cvxopt_test():
    test_source_vector_list = [np.array([1, 0, 0]), np.array([0, 1, 0])]
    test_target_vector = np.array([0.7, 0.3, 0])
    result = solve_multi_ratios(test_source_vector_list, test_target_vector)
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


def multiprocess_test2():
    pool = mp.Pool(processes=parallel_num)
    chunk_size = 100
    with pool:
        async_result = pool.map_async(
            partial(
                model_solver_single, const_parameter_dict=const_parameter_dict,
                other_parameter_dict=other_parameter_dict, hook_in_each_iteration=hook_in_each_iteration,
                hook_in_each_iteration_kwargs=hook_in_each_iteration_kwargs),
            var_parameter_list, chunk_size, callback)

        count = 0
        total_count = len(var_parameter_list)
        while count < total_count:
            b = count_q.get()
            print(b)
            count += 1
            if count % 50 == 0:
                print("Main process: {} ({:.3f}) completed".format(count, count / total_count))


def main():
    # emoji_test()
    # surrounding_circle()
    # scarlett_function()
    # plot_test()
    # hmm_test()
    # violin_test()
    # cvxopt_test()
    # convex_test()
    multiprocess_test()


if __name__ == '__main__':
    main()
