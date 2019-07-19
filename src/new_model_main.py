import pickle
import multiprocessing as mp
from functools import partial
import gzip
import os

import numpy as np
from scipy.misc import comb as scipy_comb
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.interpolate
import scipy.signal
import tqdm
import ternary
from ternary.helpers import simplex_iterator
import cvxopt

from src import model_specific_functions, config

constant_set = config.Constants()
color_set = config.Color()
test_running = config.test_running


def natural_dist(_c13_ratio, carbon_num):
    c12_ratio = 1 - _c13_ratio
    total_num = carbon_num + 1
    output = []
    for index in range(total_num):
        output.append(
            scipy_comb(carbon_num, index) * _c13_ratio ** index * c12_ratio ** (carbon_num - index))
    return np.array(output)


def split_equal_dist(source_mid, target_carbon_num):
    carbon_num = len(source_mid) - 1
    if carbon_num % 2 != 0:
        raise ValueError("Length is not multiply of 2 !!!")
    new_carbon_num = target_carbon_num
    final_output_vector = np.zeros(new_carbon_num + 1)
    final_output_vector[0] = source_mid[0]
    final_output_vector[-1] = source_mid[-1]
    average_ratio = (1 - final_output_vector[0] - final_output_vector[-1]) / (new_carbon_num - 1)
    for i in range(1, new_carbon_num):
        final_output_vector[i] = average_ratio

    # _c12_ratio = np.power(source_mid[0], (1 / carbon_num))
    # _c13_ratio = 1 - _c12_ratio
    #
    # final_output_vector = natural_dist(_c13_ratio, target_carbon_num)
    return final_output_vector


def collect_all_data(
        data_dict, _metabolite_name, _label_list, _tissue, _mouse_id_list=None, convolve=False,
        split=0, mean=True):
    matrix = []
    for label in _label_list:
        if _mouse_id_list is None:
            _mouse_id_list = data_dict[label].keys()
        for mouse_label in _mouse_id_list:
            data_for_mouse = data_dict[label][mouse_label]
            data_vector = data_for_mouse[_tissue][_metabolite_name]
            if convolve:
                data_vector = np.convolve(data_vector, data_vector)
            elif split != 0:
                data_vector = split_equal_dist(data_vector, split)
            matrix.append(data_vector)
    result_matrix = np.array(matrix).transpose()
    if mean:
        return result_matrix.mean(axis=1)
    else:
        return result_matrix.transpose().reshape([-1])


def solve_two_ratios(source_vector_list, target_vector, ratio_lb, ratio_ub):
    source_vector1, source_vector2 = source_vector_list
    if not (len(source_vector1) == len(source_vector2) == len(target_vector)):
        raise ValueError("Length of 3 vectors are not equal !!!")
    a = (source_vector1 - source_vector2).reshape([-1, 1])
    b = target_vector - source_vector2
    result = np.linalg.lstsq(a, b)
    coeff = result[0][0]
    modified_coeff = min(max(ratio_lb, coeff), ratio_ub)
    return [modified_coeff, 1 - modified_coeff]


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


def flux_ratio_constraint_generator_linear_model(mid_constraint_list, complete_flux_dict, ratio_lb, ratio_ub):
    ratio_matrix_list = []
    constant_vector_list = []
    complete_var_num = len(complete_flux_dict)
    for mid_constraint_dict in mid_constraint_list:
        target_mid_vector = mid_constraint_dict[constant_set.target_label]
        source_mid_vector_list = []
        source_flux_index_list = []
        for flux_name, vector in mid_constraint_dict.items():
            if flux_name == constant_set.target_label:
                continue
            source_flux_index_list.append(complete_flux_dict[flux_name])
            source_mid_vector_list.append(vector)
        if len(source_mid_vector_list) == 2:
            solver = solve_two_ratios
        else:
            solver = solve_multi_ratios
        coeff_list = solver(source_mid_vector_list, target_mid_vector, ratio_lb, ratio_ub)
        basic_ratio = coeff_list[0]
        basic_flux_index = source_flux_index_list[0]
        for index in range(1, len(coeff_list)):
            current_ratio = coeff_list[index]
            current_flux_index = source_flux_index_list[index]
            new_ratio_constraint_vector = np.zeros(complete_var_num)
            new_ratio_constraint_vector[basic_flux_index] = current_ratio
            new_ratio_constraint_vector[current_flux_index] = -basic_ratio
            ratio_matrix_list.append(new_ratio_constraint_vector)
            constant_vector_list.append(0)
    ratio_matrix = np.array(ratio_matrix_list)
    constant_vector = np.array(constant_vector_list)
    return ratio_matrix, constant_vector


def gradient_validation(function_value_func, jacobian_func, test_vector: np.ndarray):
    derivative_from_jacobian_func = jacobian_func(test_vector)
    variation_rate = 1e-3
    derivative_from_function = np.zeros_like(test_vector)
    for index, value in enumerate(test_vector):
        variation_value = max(abs(value) * variation_rate, 1e-5)
        high_test_vector = test_vector.copy()
        high_test_vector[index] += variation_value
        high_function_value = function_value_func(high_test_vector)
        low_test_vector = test_vector.copy()
        low_test_vector[index] -= variation_value
        low_function_value = function_value_func(low_test_vector)
        current_gradient = (high_function_value - low_function_value) / (2 * variation_value)
        derivative_from_function[index] = current_gradient
    print("Derivation from Jacobian function: {}".format(derivative_from_jacobian_func))
    print("Derivation from original function: {}".format(derivative_from_function))


def flux_balance_constraint_constructor(balance_list, complete_flux_dict):
    flux_balance_multiply_array_list = []
    flux_balance_constant_vector_list = []
    for balance_dict in balance_list:
        new_balance_array = np.zeros(len(complete_flux_dict))
        flux_name_list = balance_dict['input'] + balance_dict['output']
        value_list = [-1 for _ in balance_dict['input']] + [1 for _ in balance_dict['output']]
        for flux_name, value in zip(flux_name_list, value_list):
            flux_index = complete_flux_dict[flux_name]
            new_balance_array[flux_index] = value
        flux_balance_multiply_array_list.append(new_balance_array)
        flux_balance_constant_vector_list.append(0)
    flux_balance_matrix = np.array(flux_balance_multiply_array_list)
    flux_balance_constant_vector = np.array(flux_balance_constant_vector_list)
    return flux_balance_matrix, flux_balance_constant_vector


def mid_constraint_constructor(mid_constraint_list, complete_flux_dict):
    complete_var_num = len(complete_flux_dict)
    substrate_mid_matrix_list = []
    flux_sum_matrix_list = []
    target_mid_vector_list = []
    for mid_constraint_dict in mid_constraint_list:
        target_mid_vector = mid_constraint_dict[constant_set.target_label]
        vector_dim = len(target_mid_vector)
        new_substrate_mid_matrix_list = [np.zeros(complete_var_num) for _ in range(vector_dim)]
        new_flux_sum_matrix_list = [np.zeros(complete_var_num) for _ in range(vector_dim)]
        target_mid_vector_list.append(target_mid_vector)
        for flux_name, vector in mid_constraint_dict.items():
            if flux_name == constant_set.target_label:
                continue
            flux_index = complete_flux_dict[flux_name]
            for index, vector_value in enumerate(vector):
                new_substrate_mid_matrix_list[index][flux_index] = vector_value
                new_flux_sum_matrix_list[index][flux_index] = 1
        substrate_mid_matrix_list.extend(new_substrate_mid_matrix_list)
        flux_sum_matrix_list.extend(new_flux_sum_matrix_list)
    substrate_mid_matrix = np.array(substrate_mid_matrix_list)
    flux_sum_matrix = np.array(flux_sum_matrix_list)
    target_mid_vector = np.hstack(target_mid_vector_list) + constant_set.eps_for_log
    optimal_obj_value = -np.sum(target_mid_vector * np.log(target_mid_vector))
    return substrate_mid_matrix, flux_sum_matrix, target_mid_vector, optimal_obj_value


def constant_flux_constraint_constructor(constant_flux_dict, complete_flux_dict):
    constant_flux_multiply_array_list = []
    constant_flux_constant_vector_list = []
    for constant_flux, value in constant_flux_dict.items():
        new_balance_array = np.zeros(len(complete_flux_dict))
        flux_index = complete_flux_dict[constant_flux]
        new_balance_array[flux_index] = 1
        constant_flux_multiply_array_list.append(new_balance_array)
        constant_flux_constant_vector_list.append(-value)
    constant_flux_matrix = np.array(constant_flux_multiply_array_list)
    constant_constant_vector = np.array(constant_flux_constant_vector_list)
    return constant_flux_matrix, constant_constant_vector


def cross_entropy_obj_func_constructor(substrate_mid_matrix, flux_sum_matrix, target_mid_vector):
    def cross_entropy_objective_func(complete_vector):
        # complete_vector = np.hstack([f_vector, constant_flux_array]).reshape([-1, 1])
        complete_vector = complete_vector.reshape([-1, 1])
        predicted_mid_vector = (
                substrate_mid_matrix @ complete_vector / (flux_sum_matrix @ complete_vector) + constant_set.eps_for_log)
        cross_entropy = -target_mid_vector.reshape([1, -1]) @ np.log(predicted_mid_vector)
        return cross_entropy

    return cross_entropy_objective_func


def cross_entropy_jacobi_func_constructor(substrate_mid_matrix, flux_sum_matrix, target_mid_vector):
    def cross_entropy_jacobi_func(complete_vector):
        complete_vector = complete_vector.reshape([-1, 1])
        substrate_mid_part = substrate_mid_matrix / (substrate_mid_matrix @ complete_vector)
        flux_sum_part = flux_sum_matrix / (flux_sum_matrix @ complete_vector)
        jacobian_vector = target_mid_vector.reshape([1, -1]) @ (flux_sum_part - substrate_mid_part)
        return jacobian_vector.reshape([-1])

    return cross_entropy_jacobi_func


def eq_func_constructor(complete_balance_matrix, complete_balance_vector):
    def eq_func(complete_vector):
        result = complete_balance_matrix @ complete_vector.reshape([-1, 1]) + complete_balance_vector.reshape([-1, 1])
        return result.reshape([-1])

    return eq_func


def eq_func_jacob_constructor(complete_balance_matrix, complete_balance_vector):
    def eq_func_jacob(complete_vector):
        return complete_balance_matrix

    return eq_func_jacob


def start_point_generator(
        complete_balance_matrix, complete_balance_vector, min_flux_value, max_flux_value, maximal_failed_time=5):
    a_eq = complete_balance_matrix
    b_eq = -complete_balance_vector
    lp_lb = min_flux_value
    lp_ub = max_flux_value
    result = None
    failed_time = 0
    while failed_time < maximal_failed_time:
        random_obj = np.random.random(a_eq.shape[1]) - 0.4
        try:
            res = scipy.optimize.linprog(
                random_obj, A_eq=a_eq, b_eq=b_eq, bounds=(lp_lb, lp_ub), options={'tol': 1e-10})  # "disp": True
        except ValueError:
            failed_time += 1
            continue
        else:
            if res.success:
                result = np.array(res.x)
                break
            failed_time += 1
    return result


def one_case_solver_linear(
        flux_balance_and_mid_ratio_matrix, flux_balance_and_mid_ratio_constant_vector,
        complete_flux_dict, constant_flux_dict, min_flux_value, max_flux_value, label=None,
        **other_parameters):
    def is_valid_solution(solution_vector, min_value, max_value):
        return np.all(solution_vector > min_value) and np.all(solution_vector < max_value)

    constant_flux_matrix, constant_constant_vector = constant_flux_constraint_constructor(
        constant_flux_dict, complete_flux_dict)
    complete_balance_and_mid_matrix = np.vstack(
        [flux_balance_and_mid_ratio_matrix, constant_flux_matrix])
    complete_balance_and_mid_vector = np.hstack(
        [flux_balance_and_mid_ratio_constant_vector, constant_constant_vector])

    result_dict = {}
    success = False
    try:
        current_result = np.linalg.solve(complete_balance_and_mid_matrix, complete_balance_and_mid_vector)
    except np.linalg.LinAlgError:
        pass
    else:
        if is_valid_solution(current_result, min_flux_value, max_flux_value):
            result_dict = {
                flux_name: flux_value for flux_name, flux_value
                in zip(complete_flux_dict.keys(), current_result.x)}
            success = True
    return config.Result(result_dict, 0, success, 0, label)


def one_case_solver_slsqp(
        flux_balance_matrix, flux_balance_constant_vector, substrate_mid_matrix, flux_sum_matrix, target_mid_vector,
        optimal_obj_value, complete_flux_dict, constant_flux_dict, min_flux_value, max_flux_value,
        optimization_repeat_time, label=None, **other_parameters):
    constant_flux_matrix, constant_constant_vector = constant_flux_constraint_constructor(
        constant_flux_dict, complete_flux_dict)
    complete_balance_matrix = np.vstack(
        [flux_balance_matrix, constant_flux_matrix])
    complete_balance_vector = np.hstack(
        [flux_balance_constant_vector, constant_constant_vector])
    cross_entropy_objective_func = cross_entropy_obj_func_constructor(
        substrate_mid_matrix, flux_sum_matrix, target_mid_vector)
    cross_entropy_jacobi_func = cross_entropy_jacobi_func_constructor(
        substrate_mid_matrix, flux_sum_matrix, target_mid_vector)
    eq_func = eq_func_constructor(complete_balance_matrix, complete_balance_vector)
    eq_func_jacob = eq_func_jacob_constructor(complete_balance_matrix, complete_balance_vector)

    eq_cons = {'type': 'eq', 'fun': eq_func, 'jac': eq_func_jacob}
    bounds = scipy.optimize.Bounds(min_flux_value, max_flux_value)
    start_vector = start_point_generator(
        complete_balance_matrix, complete_balance_vector, min_flux_value, max_flux_value)
    # gradient_validation(cross_entropy_objective_func, cross_entropy_jacobi, start_vector)
    if start_vector is None:
        result_dict = {}
        obj_value = -1
        success = False
    else:
        result_dict = {}
        obj_value = 999999
        success = False
        # print("Find 1 feasible solution")
        for _ in range(optimization_repeat_time):
            start_vector = start_point_generator(
                complete_balance_matrix, complete_balance_vector, min_flux_value, max_flux_value)
            current_result = scipy.optimize.minimize(
                cross_entropy_objective_func, start_vector, method='SLSQP', jac=cross_entropy_jacobi_func,
                constraints=[eq_cons], options={'ftol': 1e-9, 'maxiter': 500}, bounds=bounds)  # 'disp': True,
            if current_result.success and current_result.fun < obj_value:
                result_dict = {
                    flux_name: flux_value for flux_name, flux_value
                    in zip(complete_flux_dict.keys(), current_result.x)}
                obj_value = current_result.fun
                success = current_result.success
    return config.Result(result_dict, obj_value, success, optimal_obj_value, label)


def calculate_one_tissue_tca_contribution(input_net_flux_list):
    real_flux_list = []
    total_input_flux = 0
    total_output_flux = 0
    for net_flux in input_net_flux_list:
        if net_flux > 0:
            total_input_flux += net_flux
        else:
            total_output_flux -= net_flux
    for net_flux in input_net_flux_list:
        current_real_flux = 0
        if net_flux > 0:
            current_real_flux = net_flux - net_flux / total_input_flux * total_output_flux
        real_flux_list.append(current_real_flux)
    return real_flux_list


def result_evaluation(
        result_dict, constant_dict, mid_constraint_list, target_diff, output_direct, common_name):
    flux_value_dict = dict(result_dict)
    flux_value_dict.update(constant_dict)
    for mid_constraint_dict in mid_constraint_list:
        target_vector = mid_constraint_dict[constant_set.target_label]
        calculate_vector = np.zeros_like(target_vector)
        total_flux_value = 1e-5
        for flux_name, mid_vector in mid_constraint_dict.items():
            if flux_name == constant_set.target_label:
                continue
            else:
                flux_value = flux_value_dict[flux_name]
                total_flux_value += flux_value
                calculate_vector += flux_value * mid_vector
        calculate_vector /= total_flux_value
        # print("MID constraint: {}\nCalculated MID: {}\nTarget MID: {}\n".format(
        #     mid_constraint_dict, calculate_vector, target_vector))
        name = "_".join([name for name in mid_constraint_dict.keys() if name != 'target'])
        experimental_label = 'Experimental MID'
        predicted_label = 'Calculated MID'
        plot_data_dict = {experimental_label: target_vector, predicted_label: calculate_vector}
        plot_color_dict = {experimental_label: color_set.blue, predicted_label: color_set.orange}
        save_path = "{}/{}_{}.png".format(output_direct, common_name, name)
        title = "{}_diff_{:.2f}".format(name, target_diff)
        plot_raw_mid_bar(plot_data_dict, plot_color_dict, title, save_path)


def plot_raw_mid_bar(data_dict, color_dict=None, title=None, save_path=None):
    edge = 0.2
    bar_total_width = 0.7
    group_num = len(data_dict)
    bar_unit_width = bar_total_width / group_num
    array_len = 0
    for data_name, np_array in data_dict.items():
        if array_len == 0:
            array_len = len(np_array)
        elif len(np_array) != array_len:
            raise ValueError("Length of array not equal: {}".format(data_name))
    fig_size = (array_len + edge * 2, 4)
    fig, ax = plt.subplots(figsize=fig_size)
    x_mid_loc = np.arange(array_len) + 0.5
    x_left_loc = x_mid_loc - bar_total_width / 2
    for index, (data_name, mid_array) in enumerate(data_dict.items()):
        if color_dict is not None:
            current_color = color_dict[data_name]
        else:
            current_color = None
        x_loc = x_left_loc + index * bar_unit_width + bar_unit_width / 2
        ax.bar(
            x_loc, mid_array, width=bar_unit_width, color=current_color,
            alpha=color_set.alpha_for_bar_plot, label=data_name)
    # ax.set_xlabel(data_dict.keys())
    ax.set_ylim([0, 1])
    ax.set_xlim([-edge, array_len + edge])
    ax.set_xticks(x_mid_loc)
    ax.set_xticklabels([])
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_yticklabels([])
    # ax.legend()
    if title:
        ax.set_title(title)
    if save_path:
        fig.savefig(save_path, dpi=fig.dpi)


# data_matrix: show the location of heatmap
def plot_heat_map(data_matrix, x_free_variable, y_free_variable, cmap=None, cbar_name=None, save_path=None):
    fig, ax = plt.subplots()
    im = ax.imshow(data_matrix, cmap=cmap)
    ax.set_xlim([0, x_free_variable.total_num])
    ax.set_ylim([0, y_free_variable.total_num])
    ax.set_xticks(x_free_variable.tick_in_range)
    ax.set_yticks(y_free_variable.tick_in_range)
    ax.set_xticklabels(x_free_variable.tick_labels)
    ax.set_yticklabels(y_free_variable.tick_labels)
    if cbar_name:
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(cbar_name, rotation=-90, va="bottom")
    if save_path:
        # print(save_path)
        fig.savefig(save_path, dpi=fig.dpi)


def plot_violin_distribution(data_dict, color_dict=None, cutoff=0.5, save_path=None):
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
            pc.set_alpha(color_set.alpha_value)
    if cutoff is not None:
        ax.axhline(cutoff, linestyle='--', color=color_set.orange)
    ax.set_ylim([-0.1, 1.1])
    ax.set_xticks(x_axis_position)
    ax.set_xticklabels(tissue_label_list)
    if save_path:
        # print(save_path)
        fig.savefig(save_path, dpi=fig.dpi)


# Plot a scatter in triangle based on data_matrix
# data_matrix: N-3 matrix. Each row is a point with 3 coordinate
def plot_ternary_scatter(data_matrix):
    ### Scatter Plot
    scale = 1
    figure, tax = ternary.figure(scale=scale)
    tax.set_title("Scatter Plot", fontsize=20)
    tax.boundary(linewidth=2.0)
    tax.gridlines(multiple=0.1, color="blue")
    # Plot a few different styles with a legend
    # points = [data_matrix]
    # tax.heatmap()
    tax.scatter(data_matrix, marker='s', color='red', label="Red Squares")
    tax.legend()
    tax.ticks(axis='lbr', linewidth=1, multiple=0.1)


# Each row of data matrix is a point in triple tuple
# In cartesian cor, the left bottom corner of triangle is the origin.
# The scale of all triangle points is 1.
# Order of ternary cor: x1: bottom (to right) x2: right (to left) x3: left (to bottom)
def plot_ternary_density(tri_data_matrix, sigma: float = 1, bin_num: int = 2 ** 8, mean=False, save_path=None):
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
    tick_labels = list(np.linspace(0, bin_num, 6) / bin_num)
    tax.ticks(axis='lbr', ticks=tick_labels, linewidth=1, tick_formats="")
    tax.clear_matplotlib_ticks()
    plt.tight_layout()
    if mean:
        mean_value = tri_data_matrix.mean(axis=0).reshape([1, -1]) * bin_num
        tax.scatter(mean_value, marker='o', color=color_set.orange, zorder=100)
    if save_path:
        print(save_path)
        fig.savefig(save_path, dpi=fig.dpi)
    # tax.show()


def parallel_solver_single(
        var_parameter_dict, const_parameter_dict, one_case_solver_func, hook_in_each_iteration):
    # var_parameter_dict, q = complete_parameter_tuple
    # result = one_case_solver_slsqp(**const_parameter_dict, **var_parameter_dict)
    result = one_case_solver_func(**const_parameter_dict, **var_parameter_dict)
    hook_result = hook_in_each_iteration(result, **const_parameter_dict, **var_parameter_dict)
    return result, hook_result


def parallel_solver(
        parameter_construction_func, one_case_solver_func, hook_in_each_iteration, model_name,
        hook_after_all_iterations, **other_parameters):
    # manager = multiprocessing.Manager()
    # q = manager.Queue()
    # result = pool.map_async(task, [(x, q) for x in range(10)])

    if test_running:
        chunk_size = 10
        parallel_num = 7
    else:
        chunk_size = 100
        parallel_num = min(os.cpu_count(), 20)

    const_parameter_dict, var_parameter_list = parameter_construction_func(
        parallel_num=parallel_num, model_name=model_name, **other_parameters)
    try:
        total_length = len(var_parameter_list)
    except TypeError:
        total_length = const_parameter_dict['iter_length']

    with mp.Pool(processes=parallel_num) as pool:
        raw_result_iter = pool.imap(
            partial(
                parallel_solver_single, const_parameter_dict=const_parameter_dict,
                one_case_solver_func=one_case_solver_func,
                hook_in_each_iteration=hook_in_each_iteration),
            var_parameter_list, chunk_size)
        raw_result_list = list(tqdm.tqdm(
            raw_result_iter, total=total_length, smoothing=0, maxinterval=5,
            desc="Computation progress of {}".format(model_name)))

    result_iter, hook_result_iter = zip(*raw_result_list)
    result_list = list(result_iter)
    hook_result_list = list(hook_result_iter)
    if not isinstance(var_parameter_list, list):
        const_parameter_dict, var_parameter_list = parameter_construction_func(
            parallel_num=parallel_num, model_name=model_name, **other_parameters)
    hook_after_all_iterations(result_list, hook_result_list, const_parameter_dict, var_parameter_list)


#     output_data_dict = {
#         'result_list': result_list,
#         'processed_result_list': processed_result_list,
#         ...
#     }
#     self.result_dict = result_dict
#     self.obj_value = obj_value
#     self.success = success
#     self.minimal_obj_value = minimal_obj_value
def fitting_result_display(
        model_mid_data_dict, model_name, model_construction_func, obj_tolerance,
        **other_parameters):
    server_data = False
    total_output_direct = "new_models"

    balance_list, mid_constraint_list = model_construction_func(model_mid_data_dict)

    if server_data:
        output_direct = "{}/{}_server".format(total_output_direct, model_name)
    else:
        output_direct = "{}/{}".format(total_output_direct, model_name)
    input_data_dict_gz_file = "{}/output_data_dict.gz".format(output_direct)
    raw_data_dict_gz_file = "{}/raw_output_data_dict.gz".format(output_direct)
    with gzip.open(input_data_dict_gz_file, 'rb') as f_in:
        input_data_dict = pickle.load(f_in)
    with gzip.open(raw_data_dict_gz_file, 'rb') as f_in:
        raw_input_data_dict = pickle.load(f_in)
    result_list: list = raw_input_data_dict['result_list']
    target_result_dict = {}
    target_obj_diff = 9999
    # common_name = "compare_bar_min"
    common_name = "compare_bar_cutoff"
    for result_object in result_list:
        obj_diff = result_object.obj_value - result_object.minimal_obj_value
        # if 0 < obj_diff < target_obj_diff:
        if abs(obj_diff - obj_tolerance) < abs(target_obj_diff - obj_tolerance):
            target_result_dict = result_object.result_dict
            target_obj_diff = obj_diff
    if len(target_result_dict) != 0:
        print("\n".join(["{}: {:.4f}".format(flux_name, value) for flux_name, value in target_result_dict.items()]))
        result_evaluation(target_result_dict, {}, mid_constraint_list, target_obj_diff, output_direct, common_name)
        plt.show()


def linear_main():
    model_parameter_dict = model_specific_functions.linear_model1_parameters()
    parallel_solver(**model_parameter_dict, one_case_solver_func=one_case_solver_linear)


def non_linear_main():
    # model_parameter_dict = model_specific_functions.model1_parameters()
    # model_parameter_dict = model_specific_functions.model2_parameters()
    # model_parameter_dict = model_specific_functions.model3_parameters()
    # model_parameter_dict = model_specific_functions.model4_parameters()
    # model_parameter_dict = model_specific_functions.model5_parameters()
    # model_parameter_dict = model_specific_functions.model6_parameters()
    # model_parameter_dict = model_specific_functions.model7_parameters()
    # model_parameter_dict = model_specific_functions.model1_all_tissue()
    # model_parameter_dict = model_specific_functions.model1_parameter_sensitivity()
    # model_parameter_dict = model_specific_functions.model1_m5_parameters()
    model_parameter_dict = model_specific_functions.model3_all_tissue()
    parallel_solver(**model_parameter_dict, one_case_solver_func=one_case_solver_slsqp)
    # model_parameter_dict = model_specific_functions.model7_parameters()
    # parallel_solver(**model_parameter_dict, one_case_solver_func=one_case_solver_slsqp)
    # fitting_result_display(**model_parameter_dict)


if __name__ == '__main__':
    # linear_main()
    non_linear_main()
