import itertools as it
import pickle
import multiprocessing as mp
import platform
from functools import partial
import time
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

import data_parser as data_parser
import main as main_functions

plasma_marker = 'Sr'
brain_marker = 'Br'
heart_marker = 'Ht'
muscle_marker = 'SkM'
kidney_marker = 'Kd'
lung_marker = 'Lg'
pancreas_marker = 'Pc'
intestine_marker = 'SI'
spleen_marker = 'Sp'
liver_marker = 'Lv'
target_label = 'target'
c13_ratio = 0.01109
eps_for_log = 1e-10


class FreeVariable(object):
    def __init__(self, name, total_num, var_range, display_interv):
        self.flux_name = name
        self.total_num = total_num + 1
        self.range = var_range
        self.display_interv = display_interv
        self.value_array = np.linspace(*self.range, self.total_num)
        self.tick_in_range = np.arange(0, self.total_num, self.display_interv, dtype='int')
        self.tick_labels = np.around(self.value_array[self.tick_in_range])

    def __iter__(self):
        return self.value_array.__iter__()


class Result(object):
    def __init__(self, result_dict: dict, obj_value: float, success: bool, minimal_obj_value: float):
        self.result_dict = result_dict
        self.obj_value = obj_value
        self.success = success
        self.minimal_obj_value = minimal_obj_value

    def __repr__(self):
        return "Result: {}\nObjective value: {}\nSuccess: {}\nMinimal objective value: {}".format(
            self.result_dict, self.obj_value, self.success, self.minimal_obj_value)


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
        data_dict, _metabolite_name, _label_list, _tissue, _mouse_id_list, convolve=False,
        split=0, mean=True):
    matrix = []
    for label in _label_list:
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
        return result_matrix


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
        target_mid_vector = mid_constraint_dict[target_label]
        vector_dim = len(target_mid_vector)
        new_substrate_mid_matrix_list = [np.zeros(complete_var_num) for _ in range(vector_dim)]
        new_flux_sum_matrix_list = [np.zeros(complete_var_num) for _ in range(vector_dim)]
        target_mid_vector_list.append(target_mid_vector)
        for flux_name, vector in mid_constraint_dict.items():
            if flux_name == target_label:
                continue
            flux_index = complete_flux_dict[flux_name]
            for index, vector_value in enumerate(vector):
                new_substrate_mid_matrix_list[index][flux_index] = vector_value
                new_flux_sum_matrix_list[index][flux_index] = 1
        substrate_mid_matrix_list.extend(new_substrate_mid_matrix_list)
        flux_sum_matrix_list.extend(new_flux_sum_matrix_list)
    substrate_mid_matrix = np.array(substrate_mid_matrix_list)
    flux_sum_matrix = np.array(flux_sum_matrix_list)
    target_mid_vector = np.hstack(target_mid_vector_list) + eps_for_log
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
                substrate_mid_matrix @ complete_vector / (flux_sum_matrix @ complete_vector) + eps_for_log)
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
    lb_ub = max_flux_value
    result = None
    failed_time = 0
    while failed_time < maximal_failed_time:
        random_obj = np.random.random(a_eq.shape[1]) - 0.4
        res = scipy.optimize.linprog(
            random_obj, A_eq=a_eq, b_eq=b_eq, bounds=(lp_lb, lb_ub), options={})  # "disp": True
        if res.success:
            result = np.array(res.x)
            break
        failed_time += 1
    return result


def sample_and_one_case_solver_slsqp(
        flux_balance_matrix, flux_balance_constant_vector, substrate_mid_matrix, flux_sum_matrix, target_mid_vector,
        optimal_obj_value, complete_flux_dict: dict, constant_flux_dict: dict, min_flux_value, max_flux_value,
        optimization_repeat_time, **other_parameter_dict):
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
    return Result(result_dict, obj_value, success, optimal_obj_value)


def solve_glucose_contribution_model12(result_dict: dict):
    glucose_flux = 0
    lactate_flux = 0
    f56 = result_dict['F5'] - result_dict['F6']
    f78 = result_dict['F7'] - result_dict['F8']
    g56 = result_dict['G5'] - result_dict['G6']
    g78 = result_dict['G7'] - result_dict['G8']
    if f56 > 0:
        glucose_flux += f56
    else:
        lactate_flux += f56
    if f78 > 0:
        lactate_flux += f78
    else:
        glucose_flux += f78
    if g56 > 0:
        glucose_flux += g56
    else:
        lactate_flux += g56
    if g78 > 0:
        lactate_flux += g78
    else:
        glucose_flux += g78
    glucose_ratio = glucose_flux / (glucose_flux + lactate_flux)
    return glucose_ratio


def solve_glucose_contribution_model34(result_dict: dict):
    def calculate_one_tissue_contribution(input_net_flux_list):
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

    f56 = result_dict['F5'] - result_dict['F6']
    f78 = result_dict['F7'] - result_dict['F8']
    f910 = result_dict['F9'] - result_dict['F10']
    g56 = result_dict['G5'] - result_dict['G6']
    g78 = result_dict['G7'] - result_dict['G8']
    g910 = result_dict['G9'] - result_dict['G10']

    source_tissue_flux_list = calculate_one_tissue_contribution([f56, f78, f910])
    sink_tissue_flux_list = calculate_one_tissue_contribution([g56, g78, g910])
    glucose_flux, lactate_flux, pyruvate_flux = [
        source_flux + sink_flux for source_flux, sink_flux in zip(source_tissue_flux_list, sink_tissue_flux_list)]
    total_flux = glucose_flux + lactate_flux + pyruvate_flux
    glucose_ratio = glucose_flux / total_flux
    lactate_ratio = lactate_flux / total_flux
    pyruvate_ratio = pyruvate_flux / total_flux
    return glucose_ratio, lactate_ratio, pyruvate_ratio


def result_evaluation(result_dict, constant_dict, mid_constraint_list):
    flux_value_dict = dict(result_dict)
    flux_value_dict.update(constant_dict)
    for mid_constraint_dict in mid_constraint_list:
        target_vector = mid_constraint_dict[target_label]
        calculate_vector = np.zeros_like(target_vector)
        total_flux_value = 1e-5
        for flux_name, mid_vector in mid_constraint_dict.items():
            if flux_name == target_label:
                continue
            else:
                flux_value = flux_value_dict[flux_name]
                total_flux_value += flux_value
                calculate_vector += flux_value * mid_vector
        calculate_vector /= total_flux_value
        print("MID constraint: {}\nCalculated MID: {}\nTarget MID: {}\n".format(
            mid_constraint_dict, calculate_vector, target_vector))


def dynamic_range_model1(model_mid_data_dict: dict, total_output_direct, **other_parameters):
    output_direct = "{}/model1".format(total_output_direct)
    if not os.path.isdir(output_direct):
        os.mkdir(output_direct)
    complete_flux_list = ['F{}'.format(i + 1) for i in range(10)] + ['G{}'.format(i + 1) for i in range(9)] + \
                         ['Fcirc_glc', 'Fcirc_lac']
    complete_flux_dict = {var: i for i, var in enumerate(complete_flux_list)}
    constant_flux_dict = {'Fcirc_glc': 150.9, 'Fcirc_lac': 374.4, 'F10': 100}

    f1_range = [1, 150]
    g2_range = [1, 150]
    if platform.node() == 'BaranLiu-PC':
        f1_num = 51
        f1_display_interv = 50
        g2_num = 51
        g2_display_interv = 50
    else:
        f1_num = 1000
        f1_display_interv = 250
        g2_num = 1000
        g2_display_interv = 250

    f1_free_flux = FreeVariable(name='F1', total_num=f1_num, var_range=f1_range, display_interv=f1_display_interv)
    g2_free_flux = FreeVariable(name='G2', total_num=g2_num, var_range=g2_range, display_interv=g2_display_interv)
    min_flux_value = 1
    max_flux_value = 5000
    optimization_repeat_time = 8
    obj_tolerance = 0.1

    balance_list, mid_constraint_list = model1_construction(model_mid_data_dict)
    flux_balance_matrix, flux_balance_constant_vector = flux_balance_constraint_constructor(
        balance_list, complete_flux_dict)
    substrate_mid_matrix, flux_sum_matrix, target_mid_vector, optimal_obj_value = mid_constraint_constructor(
        mid_constraint_list, complete_flux_dict)

    iter_parameter_list = []
    matrix_loc_list = []
    for f1_index, f1 in enumerate(f1_free_flux):
        for g2_index, g2 in enumerate(g2_free_flux):
            new_constant_flux_dict = dict(constant_flux_dict)
            new_constant_flux_dict.update({f1_free_flux.flux_name: f1, g2_free_flux.flux_name: g2})
            var_parameter_dict = {'constant_flux_dict': new_constant_flux_dict}
            iter_parameter_list.append(var_parameter_dict)
            matrix_loc_list.append((f1_index, g2_index))
    const_parameter_dict = {
        'flux_balance_matrix': flux_balance_matrix, 'flux_balance_constant_vector': flux_balance_constant_vector,
        'substrate_mid_matrix': substrate_mid_matrix, 'flux_sum_matrix': flux_sum_matrix,
        'target_mid_vector': target_mid_vector, 'optimal_obj_value': optimal_obj_value,
        'complete_flux_dict': complete_flux_dict, 'min_flux_value': min_flux_value,
        'max_flux_value': max_flux_value,

        'optimization_repeat_time': optimization_repeat_time,
        'matrix_loc_list': matrix_loc_list, 'f1_free_flux': f1_free_flux, 'g2_free_flux': g2_free_flux,
        'obj_tolerance': obj_tolerance, 'output_direct': output_direct
    }
    return const_parameter_dict, iter_parameter_list


def dynamic_range_model2(model_mid_data_dict: dict, total_output_direct, **other_parameters):
    output_direct = "{}/model2".format(total_output_direct)
    if not os.path.isdir(output_direct):
        os.mkdir(output_direct)
    complete_flux_list = ['F{}'.format(i + 1) for i in range(9)] + ['G{}'.format(i + 1) for i in range(9)] + \
                         ['Fin', 'Fcirc_lac']
    complete_flux_dict = {var: i for i, var in enumerate(complete_flux_list)}
    constant_flux_dict = {'Fin': 111.1, 'Fcirc_lac': 500}

    # f1_range = [1, 250]
    # g2_range = [1, 250]
    min_flux_value = 1
    max_flux_value = 8000
    max_free_flux_value = 250
    optimization_repeat_time = 8
    obj_tolerance = 0.3
    f1_range = [min_flux_value, max_free_flux_value]
    g2_range = [min_flux_value, max_free_flux_value]

    if platform.node() == 'BaranLiu-PC':
        f1_num = 101
        f1_display_interv = 100
        g2_num = 101
        g2_display_interv = 100
    else:
        f1_num = 1500
        f1_display_interv = 250
        g2_num = 1500
        g2_display_interv = 250

    f1_free_flux = FreeVariable(name='F1', total_num=f1_num, var_range=f1_range, display_interv=f1_display_interv)
    g2_free_flux = FreeVariable(name='G2', total_num=g2_num, var_range=g2_range, display_interv=g2_display_interv)

    balance_list, mid_constraint_list = model2_construction(model_mid_data_dict)
    flux_balance_matrix, flux_balance_constant_vector = flux_balance_constraint_constructor(
        balance_list, complete_flux_dict)
    substrate_mid_matrix, flux_sum_matrix, target_mid_vector, optimal_obj_value = mid_constraint_constructor(
        mid_constraint_list, complete_flux_dict)

    iter_parameter_list = []
    matrix_loc_list = []
    for f1_index, f1 in enumerate(f1_free_flux):
        for g2_index, g2 in enumerate(g2_free_flux):
            new_constant_flux_dict = dict(constant_flux_dict)
            new_constant_flux_dict.update({f1_free_flux.flux_name: f1, g2_free_flux.flux_name: g2})
            var_parameter_dict = {'constant_flux_dict': new_constant_flux_dict}
            iter_parameter_list.append(var_parameter_dict)
            matrix_loc_list.append((f1_index, g2_index))
    const_parameter_dict = {
        'flux_balance_matrix': flux_balance_matrix, 'flux_balance_constant_vector': flux_balance_constant_vector,
        'substrate_mid_matrix': substrate_mid_matrix, 'flux_sum_matrix': flux_sum_matrix,
        'target_mid_vector': target_mid_vector, 'optimal_obj_value': optimal_obj_value,
        'complete_flux_dict': complete_flux_dict, 'min_flux_value': min_flux_value,
        'max_flux_value': max_flux_value,

        'optimization_repeat_time': optimization_repeat_time,
        'matrix_loc_list': matrix_loc_list, 'f1_free_flux': f1_free_flux, 'g2_free_flux': g2_free_flux,
        'obj_tolerance': obj_tolerance, 'output_direct': output_direct
    }
    return const_parameter_dict, iter_parameter_list


def parameter_generator_single(free_flux_value, free_fluxes_list, constant_flux_dict):
    new_constant_flux_dict = dict(constant_flux_dict)
    new_constant_flux_dict.update(
        {flux_name: value for flux_name, value in zip(free_fluxes_list, free_flux_value)})
    var_parameter_dict = {'constant_flux_dict': new_constant_flux_dict}
    return var_parameter_dict


def dynamic_range_model3(model_mid_data_dict: dict, total_output_direct, parallel_num, **other_parameters):
    output_direct = "{}/model3".format(total_output_direct)
    if not os.path.isdir(output_direct):
        os.mkdir(output_direct)
    complete_flux_list = ['F{}'.format(i + 1) for i in range(12)] + ['G{}'.format(i + 1) for i in range(11)] + \
                         ['H{}'.format(i + 1) for i in range(3)] + ['Fcirc_glc', 'Fcirc_lac', 'Fcirc_pyr']
    complete_flux_dict = {var: i for i, var in enumerate(complete_flux_list)}
    constant_flux_dict = {'Fcirc_glc': 150.9, 'Fcirc_lac': 374.4, 'Fcirc_pyr': 57.3, 'F12': 100}

    min_flux_value = 1
    max_flux_value = 5000
    optimization_repeat_time = 8
    obj_tolerance = 0.5
    ternary_sigma = 0.15
    sample = True

    free_fluxes_list = ['F1', 'G2', 'F9', 'G10', 'F3']
    free_fluxes_range_list = [
        [min_flux_value, constant_flux_dict['Fcirc_glc']],
        [min_flux_value, constant_flux_dict['Fcirc_glc']],
        [min_flux_value, constant_flux_dict['Fcirc_pyr']],
        [min_flux_value, constant_flux_dict['Fcirc_pyr']],
        [min_flux_value, constant_flux_dict['Fcirc_lac']],
    ]

    if platform.node() == 'BaranLiu-PC':
        total_point_num = int(1e3)
        # point_interval_list = [30, 30, 12, 12, 80]
        ternary_resolution = int(2 ** 7)
    else:
        total_point_num = int(3e6)
        # point_interval_list = [10, 10, 4, 4, 20]
        ternary_resolution = int(2 ** 8)
    point_num_each_axis = np.round(np.power(total_point_num, 1 / len(free_fluxes_list))).astype('int')

    if sample:
        free_flux_raw_list = [
            np.linspace(*free_fluxes_range, total_point_num) for free_fluxes_range in free_fluxes_range_list]
        for row_index, _ in enumerate(free_fluxes_range_list):
            np.random.shuffle(free_flux_raw_list[row_index])
        free_flux_value_list = np.array(free_flux_raw_list).T
        list_length = total_point_num
    else:
        # free_fluxes_sequence_list = [
        #     np.arange(*flux_range, point_interval) for flux_range, point_interval
        #     in zip(free_fluxes_range_list, point_interval_list)]
        free_fluxes_sequence_list = [
            np.linspace(*flux_range, point_num_each_axis) for flux_range in free_fluxes_range_list]
        free_flux_value_list = it.product(*free_fluxes_sequence_list)
        list_length = np.prod([len(sequence) for sequence in free_fluxes_sequence_list])

    balance_list, mid_constraint_list = model3_construction(model_mid_data_dict)
    flux_balance_matrix, flux_balance_constant_vector = flux_balance_constraint_constructor(
        balance_list, complete_flux_dict)
    substrate_mid_matrix, flux_sum_matrix, target_mid_vector, optimal_obj_value = mid_constraint_constructor(
        mid_constraint_list, complete_flux_dict)

    # iter_parameter_list = []
    chunk_size = 1000
    with mp.Pool(processes=parallel_num) as pool:
        raw_result_iter = pool.imap(
            partial(
                parameter_generator_single, constant_flux_dict=constant_flux_dict,
                free_fluxes_list=free_fluxes_list),
            free_flux_value_list, chunk_size)
        iter_parameter_list = list(tqdm.tqdm(
            raw_result_iter, total=list_length, smoothing=0, maxinterval=5,
            desc="Parameter generation progress"))

    const_parameter_dict = {
        'flux_balance_matrix': flux_balance_matrix, 'flux_balance_constant_vector': flux_balance_constant_vector,
        'substrate_mid_matrix': substrate_mid_matrix, 'flux_sum_matrix': flux_sum_matrix,
        'target_mid_vector': target_mid_vector, 'optimal_obj_value': optimal_obj_value,
        'complete_flux_dict': complete_flux_dict, 'min_flux_value': min_flux_value,
        'max_flux_value': max_flux_value,

        'optimization_repeat_time': optimization_repeat_time,
        'obj_tolerance': obj_tolerance, 'output_direct': output_direct,
        'free_fluxes_list': free_fluxes_list,

        'ternary_sigma': ternary_sigma, 'ternary_resolution': ternary_resolution
    }
    return const_parameter_dict, iter_parameter_list


def dynamic_range_model4(model_mid_data_dict: dict, total_output_direct, parallel_num, **other_parameters):
    output_direct = "{}/model4".format(total_output_direct)
    if not os.path.isdir(output_direct):
        os.mkdir(output_direct)
    complete_flux_list = ['F{}'.format(i + 1) for i in range(11)] + ['G{}'.format(i + 1) for i in range(11)] + \
                         ['H{}'.format(i + 1) for i in range(3)] + ['Fcirc_lac', 'Fcirc_pyr', 'Fin']
    complete_flux_dict = {var: i for i, var in enumerate(complete_flux_list)}
    constant_flux_dict = {'Fin': 111.1, 'Fcirc_lac': 500, 'Fcirc_pyr': 100}
    fcirc_glc_max = 250

    min_flux_value = 1
    max_flux_value = 5000
    optimization_repeat_time = 8
    obj_tolerance = 0.4
    ternary_sigma = 0.15
    sample = False

    free_fluxes_list = ['F1', 'G2', 'F9', 'G10', 'F3']
    free_fluxes_range_list = [
        [min_flux_value, fcirc_glc_max],
        [min_flux_value, fcirc_glc_max],
        [min_flux_value, constant_flux_dict['Fcirc_pyr']],
        [min_flux_value, constant_flux_dict['Fcirc_pyr']],
        [min_flux_value, constant_flux_dict['Fcirc_lac']],
    ]

    if platform.node() == 'BaranLiu-PC':
        total_point_num = int(3e3)
        # point_interval_list = [50, 50, 20, 20, 100]
        ternary_resolution = int(2 ** 7)
    else:
        total_point_num = int(3e6)
        # point_interval_list = [25, 25, 5, 5, 25]
        ternary_resolution = int(2 ** 8)
    point_num_each_axis = np.round(np.power(total_point_num, 1 / len(free_fluxes_list))).astype('int')

    if sample:
        free_flux_raw_list = [
            np.linspace(*free_fluxes_range, total_point_num) for free_fluxes_range in free_fluxes_range_list]
        for row_index, _ in enumerate(free_fluxes_range_list):
            np.random.shuffle(free_flux_raw_list[row_index])
        free_flux_value_list = np.array(free_flux_raw_list).T
        list_length = total_point_num
    else:
        # free_fluxes_sequence_list = [
        #     np.arange(*flux_range, point_interval) for flux_range, point_interval
        #     in zip(free_fluxes_range_list, point_interval_list)]
        free_fluxes_sequence_list = [
            np.linspace(*flux_range, point_num_each_axis) for flux_range in free_fluxes_range_list]
        free_flux_value_list = it.product(*free_fluxes_sequence_list)
        list_length = np.prod([len(sequence) for sequence in free_fluxes_sequence_list])

    balance_list, mid_constraint_list = model4_construction(model_mid_data_dict)
    flux_balance_matrix, flux_balance_constant_vector = flux_balance_constraint_constructor(
        balance_list, complete_flux_dict)
    substrate_mid_matrix, flux_sum_matrix, target_mid_vector, optimal_obj_value = mid_constraint_constructor(
        mid_constraint_list, complete_flux_dict)

    # iter_parameter_list = []
    chunk_size = 1000
    with mp.Pool(processes=parallel_num) as pool:
        raw_result_iter = pool.imap(
            partial(
                parameter_generator_single, constant_flux_dict=constant_flux_dict,
                free_fluxes_list=free_fluxes_list),
            free_flux_value_list, chunk_size)
        iter_parameter_list = list(tqdm.tqdm(
            raw_result_iter, total=list_length, smoothing=0, maxinterval=5,
            desc="Parameter generation progress"))

    const_parameter_dict = {
        'flux_balance_matrix': flux_balance_matrix, 'flux_balance_constant_vector': flux_balance_constant_vector,
        'substrate_mid_matrix': substrate_mid_matrix, 'flux_sum_matrix': flux_sum_matrix,
        'target_mid_vector': target_mid_vector, 'optimal_obj_value': optimal_obj_value,
        'complete_flux_dict': complete_flux_dict, 'min_flux_value': min_flux_value,
        'max_flux_value': max_flux_value,

        'optimization_repeat_time': optimization_repeat_time,
        'obj_tolerance': obj_tolerance, 'output_direct': output_direct,
        'free_fluxes_list': free_fluxes_list,

        'ternary_sigma': ternary_sigma, 'ternary_resolution': ternary_resolution
    }
    return const_parameter_dict, iter_parameter_list


def mid_data_loader(
        data_collection_dict, label_list, mouse_id_list, source_tissue_marker, sink_tissue_marker):
    mid_data_dict = {
        'glc_source': collect_all_data(
            data_collection_dict, 'glucose', label_list, source_tissue_marker, mouse_id_list),
        'pyr_source': collect_all_data(
            data_collection_dict, 'pyruvate', label_list, source_tissue_marker, mouse_id_list),
        'lac_source': collect_all_data(
            data_collection_dict, 'lactate', label_list, source_tissue_marker, mouse_id_list),
        'glc_plasma': collect_all_data(
            data_collection_dict, 'glucose', label_list, plasma_marker, mouse_id_list),
        'pyr_plasma': collect_all_data(
            data_collection_dict, 'pyruvate', label_list, plasma_marker, mouse_id_list),
        'lac_plasma': collect_all_data(
            data_collection_dict, 'lactate', label_list, plasma_marker, mouse_id_list),
        'glc_sink': collect_all_data(
            data_collection_dict, 'glucose', label_list, sink_tissue_marker, mouse_id_list),
        'pyr_sink': collect_all_data(
            data_collection_dict, 'pyruvate', label_list, sink_tissue_marker, mouse_id_list),
        'lac_sink': collect_all_data(
            data_collection_dict, 'lactate', label_list, sink_tissue_marker, mouse_id_list),
        'glc_natural': natural_dist(c13_ratio, 6),
        'glc_infused': np.array([0, 0, 0, 0, 0, 0, 1], dtype='float'),
        'pyr_to_glc_source': collect_all_data(
            data_collection_dict, 'pyruvate', label_list, source_tissue_marker, mouse_id_list, convolve=True),
        'glc_to_pyr_source': collect_all_data(
            data_collection_dict, 'glucose', label_list, source_tissue_marker, mouse_id_list, split=3),
        'pyr_to_glc_sink': collect_all_data(
            data_collection_dict, 'pyruvate', label_list, sink_tissue_marker, mouse_id_list, convolve=True),
        'glc_to_pyr_sink': collect_all_data(
            data_collection_dict, 'glucose', label_list, sink_tissue_marker, mouse_id_list, split=3),
        'glc_to_pyr_plasma': collect_all_data(
            data_collection_dict, 'glucose', label_list, plasma_marker, mouse_id_list, split=3),
    }

    eps_of_mid = 1e-5
    for name, mid_vector in mid_data_dict.items():
        if abs(np.sum(mid_vector) - 1) > 0.001:
            raise ValueError('Sum of MID is not 1: {}'.format(name))
        mid_data_dict[name] += eps_of_mid
        mid_data_dict[name] /= np.sum(mid_data_dict[name])
    return mid_data_dict


def model1_construction(mid_data_dict):
    # Balance equations:
    glc_source_balance_eq = {'input': ['F1', 'F6', 'F10'], 'output': ['F2', 'F5']}
    pyr_source_balance_eq = {'input': ['F5', 'F7'], 'output': ['F6', 'F8', 'F9']}
    lac_source_balance_eq = {'input': ['F3', 'F8'], 'output': ['F4', 'F7']}
    glc_plasma_balance_eq = {'input': ['F2', 'G2'], 'output': ['F1', 'G1']}
    lac_plasma_balance_eq = {'input': ['F4', 'G4'], 'output': ['F3', 'G3']}
    glc_sink_balance_eq = {'input': ['G1', 'G6'], 'output': ['G2', 'G5']}
    pyr_sink_balance_eq = {'input': ['G5', 'G7'], 'output': ['G6', 'G8', 'G9']}
    lac_sink_balance_eq = {'input': ['G3', 'G8'], 'output': ['G4', 'G7']}
    glc_circ_balance_eq = {'input': ['F2', 'G2'], 'output': ['Fcirc_glc']}
    lac_circ_balance_eq = {'input': ['F4', 'G4'], 'output': ['Fcirc_lac']}

    # MID equations:

    glc_source_mid_eq = {
        'F1': mid_data_dict['glc_plasma'], 'F6': mid_data_dict['pyr_to_glc_source'],
        'F10': mid_data_dict['glc_natural'], target_label: mid_data_dict['glc_source']}
    pyr_source_mid_eq = {
        'F5': mid_data_dict['glc_to_pyr_source'], 'F7': mid_data_dict['lac_source'],
        target_label: mid_data_dict['pyr_source']}
    lac_source_mid_eq = {
        'F3': mid_data_dict['lac_plasma'], 'F8': mid_data_dict['pyr_source'],
        target_label: mid_data_dict['lac_source']}
    glc_sink_mid_eq = {
        'G1': mid_data_dict['glc_plasma'], 'G6': mid_data_dict['pyr_to_glc_sink'],
        target_label: mid_data_dict['glc_sink']}
    pyr_sink_mid_eq = {
        'G5': mid_data_dict['glc_to_pyr_sink'], 'G7': mid_data_dict['lac_sink'],
        target_label: mid_data_dict['pyr_sink']}
    lac_sink_mid_eq = {
        'G3': mid_data_dict['lac_plasma'], 'G8': mid_data_dict['pyr_sink'],
        target_label: mid_data_dict['lac_sink']}

    balance_list = [
        glc_source_balance_eq, pyr_source_balance_eq, lac_source_balance_eq, glc_plasma_balance_eq,
        lac_plasma_balance_eq, glc_sink_balance_eq, pyr_sink_balance_eq, lac_sink_balance_eq,
        glc_circ_balance_eq, lac_circ_balance_eq]
    mid_constraint_list = [
        glc_source_mid_eq, pyr_source_mid_eq, lac_source_mid_eq, glc_sink_mid_eq,
        pyr_sink_mid_eq, lac_sink_mid_eq]

    return balance_list, mid_constraint_list


def model2_construction(mid_data_dict):
    # Balance equations:
    glc_source_balance_eq = {'input': ['F1', 'F6'], 'output': ['F2', 'F5']}
    pyr_source_balance_eq = {'input': ['F5', 'F7'], 'output': ['F6', 'F8', 'F9']}
    lac_source_balance_eq = {'input': ['F3', 'F8'], 'output': ['F4', 'F7']}
    glc_plasma_balance_eq = {'input': ['F2', 'G2', 'Fin'], 'output': ['F1', 'G1']}
    lac_plasma_balance_eq = {'input': ['F4', 'G4'], 'output': ['F3', 'G3']}
    glc_sink_balance_eq = {'input': ['G1', 'G6'], 'output': ['G2', 'G5']}
    pyr_sink_balance_eq = {'input': ['G5', 'G7'], 'output': ['G6', 'G8', 'G9']}
    lac_sink_balance_eq = {'input': ['G3', 'G8'], 'output': ['G4', 'G7']}
    lac_circ_balance_eq = {'input': ['F4', 'G4'], 'output': ['Fcirc_lac']}

    # MID equations:

    glc_source_mid_eq = {
        'F1': mid_data_dict['glc_plasma'], 'F6': mid_data_dict['pyr_to_glc_source'],
        target_label: mid_data_dict['glc_source']}
    pyr_source_mid_eq = {
        'F5': mid_data_dict['glc_to_pyr_source'], 'F7': mid_data_dict['lac_source'],
        target_label: mid_data_dict['pyr_source']}
    lac_source_mid_eq = {
        'F3': mid_data_dict['lac_plasma'], 'F8': mid_data_dict['pyr_source'],
        target_label: mid_data_dict['lac_source']}
    glc_sink_mid_eq = {
        'G1': mid_data_dict['glc_plasma'], 'G6': mid_data_dict['pyr_to_glc_sink'],
        target_label: mid_data_dict['glc_sink']}
    pyr_sink_mid_eq = {
        'G5': mid_data_dict['glc_to_pyr_sink'], 'G7': mid_data_dict['lac_sink'],
        target_label: mid_data_dict['pyr_sink']}
    lac_sink_mid_eq = {
        'G3': mid_data_dict['lac_plasma'], 'G8': mid_data_dict['pyr_sink'],
        target_label: mid_data_dict['lac_sink']}
    glc_plasma_mid_eq = {
        'F2': mid_data_dict['glc_source'], 'G2': mid_data_dict['glc_sink'],
        'Fin': mid_data_dict['glc_natural'], target_label: mid_data_dict['glc_plasma']}

    balance_list = [
        glc_source_balance_eq, pyr_source_balance_eq, lac_source_balance_eq, glc_plasma_balance_eq,
        lac_plasma_balance_eq, glc_sink_balance_eq, pyr_sink_balance_eq, lac_sink_balance_eq,
        lac_circ_balance_eq]
    mid_constraint_list = [
        glc_source_mid_eq, pyr_source_mid_eq, lac_source_mid_eq, glc_sink_mid_eq,
        pyr_sink_mid_eq, lac_sink_mid_eq, glc_plasma_mid_eq]

    return balance_list, mid_constraint_list


def model3_construction(mid_data_dict):
    # Balance equations:
    glc_source_balance_eq = {'input': ['F1', 'F6', 'F12'], 'output': ['F2', 'F5']}
    pyr_source_balance_eq = {'input': ['F5', 'F7', 'F9'], 'output': ['F6', 'F8', 'F10', 'F11']}
    lac_source_balance_eq = {'input': ['F3', 'F8'], 'output': ['F4', 'F7']}
    glc_plasma_balance_eq = {'input': ['F2', 'G2'], 'output': ['F1', 'G1', 'H1']}
    pyr_plasma_balance_eq = {'input': ['F10', 'G10', 'H1', 'H3'], 'output': ['F9', 'G9', 'H2']}
    lac_plasma_balance_eq = {'input': ['F4', 'G4', 'H2'], 'output': ['F3', 'G3', 'H3']}
    glc_sink_balance_eq = {'input': ['G1', 'G6'], 'output': ['G2', 'G5']}
    pyr_sink_balance_eq = {'input': ['G5', 'G7', 'G9'], 'output': ['G6', 'G8', 'G10', 'G11']}
    lac_sink_balance_eq = {'input': ['G3', 'G8'], 'output': ['G4', 'G7']}
    glc_circ_balance_eq = {'input': ['F1', 'G1'], 'output': ['Fcirc_glc']}
    lac_circ_balance_eq = {'input': ['F3', 'G3'], 'output': ['Fcirc_lac']}
    pyr_circ_balance_eq = {'input': ['F9', 'G9'], 'output': ['Fcirc_pyr']}

    # MID equations:

    glc_source_mid_eq = {
        'F1': mid_data_dict['glc_plasma'], 'F6': mid_data_dict['pyr_to_glc_source'],
        'F10': mid_data_dict['glc_natural'], target_label: mid_data_dict['glc_source']}
    pyr_source_mid_eq = {
        'F5': mid_data_dict['glc_to_pyr_source'], 'F7': mid_data_dict['lac_source'],
        target_label: mid_data_dict['pyr_source']}
    lac_source_mid_eq = {
        'F3': mid_data_dict['lac_plasma'], 'F8': mid_data_dict['pyr_source'],
        target_label: mid_data_dict['lac_source']}
    lac_plasma_mid_eq = {
        'G4': mid_data_dict['lac_sink'], 'F4': mid_data_dict['lac_source'],
        'H2': mid_data_dict['pyr_plasma'], target_label: mid_data_dict['lac_plasma']}
    pyr_plasma_mid_eq = {
        'G10': mid_data_dict['pyr_sink'], 'F10': mid_data_dict['pyr_source'],
        'H1': mid_data_dict['glc_to_pyr_plasma'], 'H3': mid_data_dict['lac_plasma'],
        target_label: mid_data_dict['pyr_plasma']}
    glc_sink_mid_eq = {
        'G1': mid_data_dict['glc_plasma'], 'G6': mid_data_dict['pyr_to_glc_sink'],
        target_label: mid_data_dict['glc_sink']}
    pyr_sink_mid_eq = {
        'G5': mid_data_dict['glc_to_pyr_sink'], 'G7': mid_data_dict['lac_sink'],
        target_label: mid_data_dict['pyr_sink']}
    lac_sink_mid_eq = {
        'G3': mid_data_dict['lac_plasma'], 'G8': mid_data_dict['pyr_sink'],
        target_label: mid_data_dict['lac_sink']}

    balance_list = [
        glc_source_balance_eq, pyr_source_balance_eq, lac_source_balance_eq, glc_plasma_balance_eq,
        pyr_plasma_balance_eq, lac_plasma_balance_eq, glc_sink_balance_eq, pyr_sink_balance_eq,
        lac_sink_balance_eq, glc_circ_balance_eq, lac_circ_balance_eq, pyr_circ_balance_eq]
    mid_constraint_list = [
        glc_source_mid_eq, pyr_source_mid_eq, lac_source_mid_eq, lac_plasma_mid_eq, pyr_plasma_mid_eq,
        glc_sink_mid_eq, pyr_sink_mid_eq, lac_sink_mid_eq]

    return balance_list, mid_constraint_list


def model4_construction(mid_data_dict):
    # Balance equations:
    glc_source_balance_eq = {'input': ['F1', 'F6'], 'output': ['F2', 'F5']}
    pyr_source_balance_eq = {'input': ['F5', 'F7', 'F9'], 'output': ['F6', 'F8', 'F10', 'F11']}
    lac_source_balance_eq = {'input': ['F3', 'F8'], 'output': ['F4', 'F7']}
    glc_plasma_balance_eq = {'input': ['F2', 'G2', 'Fin'], 'output': ['F1', 'G1', 'H1']}
    pyr_plasma_balance_eq = {'input': ['F10', 'G10', 'H1', 'H3'], 'output': ['F9', 'G9', 'H2']}
    lac_plasma_balance_eq = {'input': ['F4', 'G4', 'H2'], 'output': ['F3', 'G3', 'H3']}
    glc_sink_balance_eq = {'input': ['G1', 'G6'], 'output': ['G2', 'G5']}
    pyr_sink_balance_eq = {'input': ['G5', 'G7', 'G9'], 'output': ['G6', 'G8', 'G10', 'G11']}
    lac_sink_balance_eq = {'input': ['G3', 'G8'], 'output': ['G4', 'G7']}
    lac_circ_balance_eq = {'input': ['F3', 'G3'], 'output': ['Fcirc_lac']}
    pyr_circ_balance_eq = {'input': ['F9', 'G9'], 'output': ['Fcirc_pyr']}

    # MID equations:
    glc_source_mid_eq = {
        'F1': mid_data_dict['glc_plasma'], 'F6': mid_data_dict['pyr_to_glc_source'],
        target_label: mid_data_dict['glc_source']}
    pyr_source_mid_eq = {
        'F5': mid_data_dict['glc_to_pyr_source'], 'F7': mid_data_dict['lac_source'],
        target_label: mid_data_dict['pyr_source']}
    lac_source_mid_eq = {
        'F3': mid_data_dict['lac_plasma'], 'F8': mid_data_dict['pyr_source'],
        target_label: mid_data_dict['lac_source']}
    glc_plasma_mid_eq = {
        'G2': mid_data_dict['glc_sink'], 'F2': mid_data_dict['glc_source'],
        'Fin': mid_data_dict['glc_infused'], target_label: mid_data_dict['glc_plasma']}
    lac_plasma_mid_eq = {
        'G4': mid_data_dict['lac_sink'], 'F4': mid_data_dict['lac_source'],
        'H2': mid_data_dict['pyr_plasma'], target_label: mid_data_dict['lac_plasma']}
    pyr_plasma_mid_eq = {
        'G10': mid_data_dict['pyr_sink'], 'F10': mid_data_dict['pyr_source'],
        'H1': mid_data_dict['glc_to_pyr_plasma'], 'H3': mid_data_dict['lac_plasma'],
        target_label: mid_data_dict['pyr_plasma']}
    glc_sink_mid_eq = {
        'G1': mid_data_dict['glc_plasma'], 'G6': mid_data_dict['pyr_to_glc_sink'],
        target_label: mid_data_dict['glc_sink']}
    pyr_sink_mid_eq = {
        'G5': mid_data_dict['glc_to_pyr_sink'], 'G7': mid_data_dict['lac_sink'],
        target_label: mid_data_dict['pyr_sink']}
    lac_sink_mid_eq = {
        'G3': mid_data_dict['lac_plasma'], 'G8': mid_data_dict['pyr_sink'],
        target_label: mid_data_dict['lac_sink']}

    balance_list = [
        glc_source_balance_eq, pyr_source_balance_eq, lac_source_balance_eq, glc_plasma_balance_eq,
        pyr_plasma_balance_eq, lac_plasma_balance_eq, glc_sink_balance_eq, pyr_sink_balance_eq,
        lac_sink_balance_eq, lac_circ_balance_eq, pyr_circ_balance_eq]
    mid_constraint_list = [
        glc_source_mid_eq, pyr_source_mid_eq, lac_source_mid_eq, glc_plasma_mid_eq, lac_plasma_mid_eq,
        pyr_plasma_mid_eq, glc_sink_mid_eq, pyr_sink_mid_eq, lac_sink_mid_eq]

    return balance_list, mid_constraint_list


def result_processing_each_iteration_model12(result: Result, **other_parameter_dict):
    processed_dict = {}
    # if result.success and current_obj_value - minimal_obj_value < obj_tolerance:
    if result.success:
        processed_dict['obj_diff'] = result.obj_value - result.minimal_obj_value
        processed_dict['valid'] = True
        glucose_contribution = solve_glucose_contribution_model12(result.result_dict)
        processed_dict['glucose_contribution'] = glucose_contribution
    else:
        processed_dict['obj_diff'] = np.nan
        processed_dict['valid'] = False
        processed_dict['glucose_contribution'] = -1
    return processed_dict


def result_processing_each_iteration_model34(result: Result, **other_parameter_dict):
    processed_dict = {}
    if result.success:
        processed_dict['obj_diff'] = result.obj_value - result.minimal_obj_value
        processed_dict['valid'] = True
        contribution_list = solve_glucose_contribution_model34(result.result_dict)
        processed_dict['contribution_array'] = np.array(contribution_list)
    else:
        processed_dict['obj_diff'] = np.nan
        processed_dict['valid'] = False
        processed_dict['contribution_array'] = np.array([])
    return processed_dict


def model1_print_result(result_dict, constant_flux_dict):
    var_string_list = ["{} = {:.3e}".format(var_name, value) for var_name, value in result_dict.items()]
    const_string_list = ["{} = {:.3f}".format(const_name, value) for const_name, value in constant_flux_dict.items()]
    print("Variables:\n{}\n".format("\n".join(var_string_list)))
    print("Constants:\n{}".format("\n".join(const_string_list)))


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
        print(save_path)
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

    # ternary.plt.show()


# Each row of data matrix is a point in triple tuple
# In cartesian cor, the left bottom corner of triangle is the origin.
# The scale of all triangle points is 1.
# Order of ternary cor: x1: bottom (to right) x2: right (to left) x3: left (to bottom)
def plot_ternary_density(tri_data_matrix, sigma: float = 1, bin_num: int = 2 ** 8, save_path=None):
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
    tick_labels = list(np.linspace(0, bin_num, 11) / bin_num)
    tax.ticks(axis='lbr', ticks=tick_labels, linewidth=1, tick_formats="")
    tax.clear_matplotlib_ticks()
    plt.tight_layout()
    if save_path:
        print(save_path)
        fig.savefig(save_path, dpi=fig.dpi)
    # tax.show()


def final_result_processing_and_plotting_model12(
        result_list, processed_result_list, const_parameter_dict, var_parameter_list):
    f1_free_flux: FreeVariable = const_parameter_dict['f1_free_flux']
    g2_free_flux: FreeVariable = const_parameter_dict['g2_free_flux']
    matrix_loc_list = const_parameter_dict['matrix_loc_list']
    output_direct = const_parameter_dict['output_direct']
    obj_tolerance = const_parameter_dict['obj_tolerance']

    valid_matrix = np.zeros([f1_free_flux.total_num, g2_free_flux.total_num])
    glucose_contri_matrix = np.zeros_like(valid_matrix)
    objective_function_matrix = np.zeros_like(valid_matrix)
    well_fit_glucose_contri_list = []

    for solver_result, processed_dict, matrix_loc in zip(result_list, processed_result_list, matrix_loc_list):
        if processed_dict['valid']:
            valid_matrix[matrix_loc] = 1
            glucose_contri_matrix[matrix_loc] = processed_dict['glucose_contribution']
            objective_function_matrix[matrix_loc] = processed_dict['obj_diff']
            if processed_dict['obj_diff'] < obj_tolerance:
                well_fit_glucose_contri_list.append(processed_dict['glucose_contribution'])
        else:
            valid_matrix[matrix_loc] = 0
            glucose_contri_matrix[matrix_loc] = np.nan
            objective_function_matrix[matrix_loc] = np.nan

    filtered_obj_function_matrix = objective_function_matrix.copy()
    filtered_obj_function_matrix[objective_function_matrix > obj_tolerance] = np.nan

    plot_heat_map(valid_matrix, g2_free_flux, f1_free_flux, save_path="{}/dynamic_range.png".format(output_direct))
    plot_heat_map(
        glucose_contri_matrix, g2_free_flux, f1_free_flux, cmap='cool', cbar_name='Glucose Contribution',
        save_path="{}/glucose_contribution_heatmap.png".format(output_direct))
    plot_heat_map(
        objective_function_matrix, g2_free_flux, f1_free_flux, cmap='cool', cbar_name='Objective difference',
        save_path="{}/objective_function.png".format(output_direct))
    plot_heat_map(
        filtered_obj_function_matrix, g2_free_flux, f1_free_flux, cmap='cool',
        cbar_name='Filtered objective difference',
        save_path="{}/filtered_objective_function.png".format(output_direct))

    if len(well_fit_glucose_contri_list) == 0:
        raise ValueError('No point fit the constraint for contribution of carbon sources!')
    # glucose_contri_vector = glucose_contri_matrix.reshape([-1])
    # glucose_contri_vector = glucose_contri_vector[~np.isnan(glucose_contri_vector)]
    glucose_contri_vector = np.array(well_fit_glucose_contri_list)
    fig, ax = main_functions.violin_plot({"normal": glucose_contri_vector})
    fig.savefig("{}/glucose_contribution_violin.png".format(output_direct), dpi=fig.dpi)

    with open("{}/glucose_contri_matrix".format(output_direct), 'wb') as f_out:
        pickle.dump(glucose_contri_matrix, f_out)
    with open("{}/valid_matrix".format(output_direct), 'wb') as f_out:
        pickle.dump(valid_matrix, f_out)
    with open("{}/objective_function_matrix".format(output_direct), 'wb') as f_out:
        pickle.dump(objective_function_matrix, f_out)

    if platform.node() == 'BaranLiu-PC':
        plt.show()


def final_result_processing_and_plotting_model34(
        result_list, processed_result_list, const_parameter_dict, var_parameter_list):
    output_direct = const_parameter_dict['output_direct']
    free_fluxes_list = const_parameter_dict['free_fluxes_list']
    ternary_sigma = const_parameter_dict['ternary_sigma']
    ternary_resolution = const_parameter_dict['ternary_resolution']
    obj_tolerance = const_parameter_dict['obj_tolerance']

    valid_point_list = []
    invalid_point_list = []
    well_fit_three_contri_list = []
    obj_diff_value_list = []

    for solver_result, processed_dict, var_parameter in zip(
            result_list, processed_result_list, var_parameter_list):
        constant_fluxes_dict = var_parameter['constant_flux_dict']
        free_fluxes_array = np.array([constant_fluxes_dict[flux_name] for flux_name in free_fluxes_list])
        if processed_dict['valid']:
            valid_point_list.append(free_fluxes_array)
            obj_diff_value_list.append(processed_dict['obj_diff'])
            if processed_dict['obj_diff'] < obj_tolerance:
                well_fit_three_contri_list.append(processed_dict['contribution_array'])
        else:
            invalid_point_list.append(free_fluxes_array)

    if len(well_fit_three_contri_list) == 0:
        raise ValueError('No point fit the constraint for contribution of carbon sources!')
    contribution_matrix = np.array(well_fit_three_contri_list)
    output_data_dict = {
        'valid_point_list': valid_point_list,
        'invalid_point_list': invalid_point_list,
        'contribution_matrix': contribution_matrix,
        'obj_diff_value_list': obj_diff_value_list,
    }

    with open("{}/output_data_dict".format(output_direct), 'wb') as f_out:
        pickle.dump(output_data_dict, f_out)

    # Ternary plot for density of contribution
    plot_ternary_density(
        contribution_matrix, ternary_sigma, ternary_resolution,
        save_path="{}/glucose_contribution_heatmap.png".format(output_direct))
    # Violin plot for objective function
    fig, ax = main_functions.violin_plot({'normal': np.array(obj_diff_value_list)})
    fig.savefig("{}/objective_function_diff_violin.png".format(output_direct), dpi=fig.dpi)

    if platform.node() == 'BaranLiu-PC':
        plt.show()


def model_solver_single(
        var_parameter_dict, const_parameter_dict, hook_in_each_iteration):
    # var_parameter_dict, q = complete_parameter_tuple
    result = sample_and_one_case_solver_slsqp(**const_parameter_dict, **var_parameter_dict)
    hook_result = hook_in_each_iteration(result, **const_parameter_dict, **var_parameter_dict)
    return result, hook_result


def model_solver_slsqp_parallel_pool(
        data_collection_obj, data_collection_func, data_collection_kwargs, parameter_construction_func,
        parameter_construction_kwargs, hook_in_each_iteration, hook_after_all_iterations):
    raw_data_collection_dict = data_collection_obj.mid_data
    model_mid_data_dict = data_collection_func(raw_data_collection_dict, **data_collection_kwargs)

    # manager = multiprocessing.Manager()
    # q = manager.Queue()
    # result = pool.map_async(task, [(x, q) for x in range(10)])

    if platform.node() == 'BaranLiu-PC':
        chunk_size = 10
        parallel_num = 7
    else:
        chunk_size = 100
        parallel_num = 12

    const_parameter_dict, var_parameter_list = parameter_construction_func(
        model_mid_data_dict, parallel_num=parallel_num, **parameter_construction_kwargs)

    with mp.Pool(processes=parallel_num) as pool:
        raw_result_iter = pool.imap(
            partial(
                model_solver_single, const_parameter_dict=const_parameter_dict,
                hook_in_each_iteration=hook_in_each_iteration),
            var_parameter_list, chunk_size)
        raw_result_list = list(tqdm.tqdm(
            raw_result_iter, total=len(var_parameter_list), smoothing=0, maxinterval=5, desc="Computation progress"))

    result_iter, hook_result_iter = zip(*raw_result_list)
    result_list = list(result_iter)
    hook_result_list = list(hook_result_iter)

    # print(len(result_list))
    # print(result_list[0])
    # print(len(hook_result_list))
    # print(hook_result_list[0])
    hook_after_all_iterations(result_list, hook_result_list, const_parameter_dict, var_parameter_list)


def model1_dynamic_range_glucose_contribution():
    file_path = "data_collection.xlsx"
    experiment_name_prefix = "Sup_Fig_5_fasted"
    total_output_direct = "new_models"
    # label_list = ["glucose", "lactate"]
    label_list = ["glucose"]
    data_collection_func = mid_data_loader
    data_collection_kwargs = {
        'label_list': label_list, 'mouse_id_list': ['M1'],
        'source_tissue_marker': liver_marker, 'sink_tissue_marker': heart_marker}

    parameter_construction_func = dynamic_range_model1
    parameter_construction_kwargs = {'total_output_direct': total_output_direct}
    hook_in_each_iteration = result_processing_each_iteration_model12
    hook_after_all_iterations = final_result_processing_and_plotting_model12
    # solver_func = model_solver_slsqp_parallel
    solver_func = model_solver_slsqp_parallel_pool

    data_collection = data_parser.data_parser(file_path, experiment_name_prefix, label_list)
    data_collection = data_parser.data_checker(
        data_collection, ["glucose", "pyruvate", "lactate"], ["glucose", "pyruvate", "lactate"])
    start = time.time()
    solver_func(
        data_collection, data_collection_func, data_collection_kwargs, parameter_construction_func,
        parameter_construction_kwargs, hook_in_each_iteration, hook_after_all_iterations)
    duration = time.time() - start
    print("Time elapsed: {:.3f}s".format(duration))


def model2_dynamic_range_glucose_contribution():
    file_path = "data_collection_from_Dan.xlsx"
    experiment_name_prefix = "no_tumor"
    total_output_direct = "new_models"
    # label_list = ["glucose", "lactate"]
    label_list = ["glucose"]
    data_collection_func = mid_data_loader
    data_collection_kwargs = {
        'label_list': label_list, 'mouse_id_list': ['M1'],
        'source_tissue_marker': liver_marker, 'sink_tissue_marker': muscle_marker}

    parameter_construction_func = dynamic_range_model2
    parameter_construction_kwargs = {'total_output_direct': total_output_direct}
    hook_in_each_iteration = result_processing_each_iteration_model12
    hook_after_all_iterations = final_result_processing_and_plotting_model12
    # solver_func = model_solver_slsqp_parallel
    solver_func = model_solver_slsqp_parallel_pool

    data_collection = data_parser.data_parser(file_path, experiment_name_prefix, label_list)
    data_collection = data_parser.data_checker(
        data_collection, ["glucose", "pyruvate", "lactate"], ["glucose", "pyruvate", "lactate"])
    start = time.time()
    solver_func(
        data_collection, data_collection_func, data_collection_kwargs, parameter_construction_func,
        parameter_construction_kwargs, hook_in_each_iteration, hook_after_all_iterations)
    duration = time.time() - start
    print("Time elapsed: {:.3f}s".format(duration))


def model3_dynamic_range_glucose_contribution():
    file_path = "data_collection.xlsx"
    experiment_name_prefix = "Sup_Fig_5_fasted"
    total_output_direct = "new_models"
    # label_list = ["glucose", "lactate"]
    label_list = ["glucose"]
    data_collection_func = mid_data_loader
    data_collection_kwargs = {
        'label_list': label_list, 'mouse_id_list': ['M1'],
        'source_tissue_marker': liver_marker, 'sink_tissue_marker': heart_marker}

    parameter_construction_func = dynamic_range_model3
    parameter_construction_kwargs = {'total_output_direct': total_output_direct}
    hook_in_each_iteration = result_processing_each_iteration_model34
    hook_after_all_iterations = final_result_processing_and_plotting_model34

    data_collection = data_parser.data_parser(file_path, experiment_name_prefix, label_list)
    data_collection = data_parser.data_checker(
        data_collection, ["glucose", "pyruvate", "lactate"], ["glucose", "pyruvate", "lactate"])
    start = time.time()
    model_solver_slsqp_parallel_pool(
        data_collection, data_collection_func, data_collection_kwargs, parameter_construction_func,
        parameter_construction_kwargs, hook_in_each_iteration, hook_after_all_iterations)
    duration = time.time() - start
    print("Time elapsed: {:.3f}s".format(duration))


def model4_dynamic_range_glucose_contribution():
    file_path = "data_collection_from_Dan.xlsx"
    experiment_name_prefix = "no_tumor"
    total_output_direct = "new_models"
    # label_list = ["glucose", "lactate"]
    label_list = ["glucose"]
    data_collection_func = mid_data_loader
    data_collection_kwargs = {
        'label_list': label_list, 'mouse_id_list': ['M1'],
        'source_tissue_marker': liver_marker, 'sink_tissue_marker': muscle_marker}

    parameter_construction_func = dynamic_range_model4
    parameter_construction_kwargs = {'total_output_direct': total_output_direct}
    hook_in_each_iteration = result_processing_each_iteration_model34
    hook_after_all_iterations = final_result_processing_and_plotting_model34

    data_collection = data_parser.data_parser(file_path, experiment_name_prefix, label_list)
    data_collection = data_parser.data_checker(
        data_collection, ["glucose", "pyruvate", "lactate"], ["glucose", "pyruvate", "lactate"])
    start = time.time()
    model_solver_slsqp_parallel_pool(
        data_collection, data_collection_func, data_collection_kwargs, parameter_construction_func,
        parameter_construction_kwargs, hook_in_each_iteration, hook_after_all_iterations)
    duration = time.time() - start
    print("Time elapsed: {:.3f}s".format(duration))


def main():
    # file_path = "data_collection_from_Dan.xlsx"
    # experiment_name_prefix = "no_tumor"
    # label_list = ["glucose"]
    # model1_dynamic_range_glucose_contribution()
    # model2_dynamic_range_glucose_contribution()
    # model3_dynamic_range_glucose_contribution()
    model4_dynamic_range_glucose_contribution()


if __name__ == '__main__':
    main()
