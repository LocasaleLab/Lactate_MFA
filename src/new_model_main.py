#!/usr/bin/env python
#  -*- coding: utf-8 -*-
# (C) Shiyu Liu, Locasale Lab, 2019
# Contact: liushiyu1994@gmail.com
# All rights reserved
# Licensed under MIT License (see LICENSE-MIT)

"""
    Functions for optimization and parallel computation, as well as entrance of main function and arguments parser.
"""

import argparse
import itertools as it
import multiprocessing as mp
import os
from functools import partial

import numpy as np
import scipy.optimize
import tqdm

from src import model_parameter_functions, common_functions, config

# Import common functions from model_specific_functions:

constant_set = config.Constants()
global_output_direct = constant_set.output_direct
eps_for_log = constant_set.eps_for_log

constant_flux_constraint_constructor = common_functions.constant_flux_constraint_constructor


# Function constructor functions are designed to construct necessary functions that feed optimizer.

def cross_entropy_obj_func_constructor(substrate_mid_matrix, flux_sum_matrix, target_mid_vector):
    """
    Function that return objective function, which takes a flux vector as only argument and return its objective value.

    :param substrate_mid_matrix: Matrix that contains information of metabolite MID.
    :param flux_sum_matrix: Matrix that contains information of sum of some fluxes.
    :param target_mid_vector: Vector that contains target MID of metabolites. Used to compare with predicted MID vector.
    :return: Function to predict MID and calculate cross entropy of predicted and target MID.
    """

    def cross_entropy_objective_func(complete_vector):
        complete_vector = complete_vector.reshape([-1, 1])
        predicted_mid_vector = (
                (substrate_mid_matrix @ complete_vector) / (flux_sum_matrix @ complete_vector) + eps_for_log)
        cross_entropy = -target_mid_vector.reshape([1, -1]) @ np.log(predicted_mid_vector)
        return cross_entropy

    return cross_entropy_objective_func


def cross_entropy_jacobi_func_constructor(substrate_mid_matrix, flux_sum_matrix, target_mid_vector):
    """
    Function that return Jacobian vector, which takes a flux vector as only argument and return the derivative of
    objective function to each element of flux vector.

    :param substrate_mid_matrix: Matrix that contains information of metabolite MID.
    :param flux_sum_matrix: Matrix that contains information of sum of some fluxes.
    :param target_mid_vector: Vector that contains target MID of metabolites. Used to compare with predicted MID vector.
    :return: Function to calculate Jacobian vector for optimizer.
    """

    def cross_entropy_jacobi_func(complete_vector):
        complete_vector = complete_vector.reshape([-1, 1])
        substrate_mid_part = substrate_mid_matrix / (substrate_mid_matrix @ complete_vector)
        flux_sum_part = flux_sum_matrix / (flux_sum_matrix @ complete_vector)
        jacobian_vector = target_mid_vector.reshape([1, -1]) @ (flux_sum_part - substrate_mid_part)
        return jacobian_vector.reshape([-1])

    return cross_entropy_jacobi_func


def eq_func_constructor(complete_balance_matrix, complete_balance_vector):
    """
    Function that return equality constraints function. If the equal constraint is A @ x = -b, this function will return
    a function which take a flux vector as only argument and return value of A @ x + b.

    :param complete_balance_matrix: Matrix of flux balance.
    :param complete_balance_vector: Vector that on the right side of flux balance.
    :return: Function to calculate difference to equality constraints.
    """

    def eq_func(complete_vector):
        result = complete_balance_matrix @ complete_vector.reshape([-1, 1]) + complete_balance_vector.reshape([-1, 1])
        return result.reshape([-1])

    return eq_func


def eq_func_jacob_constructor(complete_balance_matrix, complete_balance_vector):
    """
    Function that return Jacobian vector to equality constraints function, which takes a flux vector as only argument and
    return the derivative of equality constraints function to each element of flux vector.

    :param complete_balance_matrix: Matrix of flux balance.
    :param complete_balance_vector: Vector that on the right side of flux balance.
    :return: Function to calculate derivative of difference to equality constraints.
    """

    def eq_func_jacob(complete_vector):
        return complete_balance_matrix

    return eq_func_jacob


# Core functions in computation, such as initialization, optimization and parallel-computation.

def start_point_generator(
        complete_balance_matrix, complete_balance_vector, bounds, maximal_failed_time=10):
    """
    Generate a random start point with satisfying constraints based on linear programming. Constraints are perturbed
    randomly to ensure a evenly distribution of sampling. Feasible solution may not exist under current condition.

    :param complete_balance_matrix: Matrix for equality constraints, including constraints for constant flux.
    :param complete_balance_vector: Vector for equality constraints, including constraints for constant flux.
    :param bounds: Inequality constraints, including range of fluxes.
    :param maximal_failed_time: maximal failed time to abandon searching.
    :return: A feasible solution that satisfies current condition. None if not exists.
    """

    a_eq = complete_balance_matrix
    b_eq = -complete_balance_vector
    raw_lb, raw_ub = bounds
    result = None
    failed_time = 0
    num_variable = a_eq.shape[1]
    while failed_time < maximal_failed_time:
        random_obj = np.random.random(num_variable) - 0.4
        lp_lb = raw_lb + np.random.random(num_variable) * 4 + 1
        lp_ub = raw_ub * (np.random.random(num_variable) * 0.2 + 0.8)
        bounds_matrix = np.vstack([lp_lb, lp_ub]).T
        try:
            res = scipy.optimize.linprog(
                random_obj, A_eq=a_eq, b_eq=b_eq, bounds=bounds_matrix, method="simplex",
                options={'tol': 1e-10})  # "disp": True
        except ValueError:
            failed_time += 1
            continue
        else:
            if res.success:
                result = np.array(res.x)
                break
            failed_time += 1
    return result


def one_case_solver_slsqp(
        flux_balance_matrix, flux_balance_constant_vector, substrate_mid_matrix, flux_sum_matrix, target_mid_vector,
        optimal_obj_value, complete_flux_dict, constant_flux_dict, bounds,
        optimization_repeat_time, label=None, fitted=True, **other_parameters):
    """
    Core function to optimize flux vector that minimize objective function given equality and inequality constraints.
    It invokes sequential quadratic programming in scipy package (scipy.optimize.minimize).

    :param flux_balance_matrix: Matrix of flux balance constraints.
    :param flux_balance_constant_vector: Vector of flux balance constraints.
    :param substrate_mid_matrix: Matrix to calculate MID of substrate.
    :param flux_sum_matrix: Matrix to calculate sum of flues.
    :param target_mid_vector: Vector of target MID.
    :param optimal_obj_value: Optimal value of objective function.
    :param complete_flux_dict: Flux name to index dict to ensure all fluxes have same order.
    :param constant_flux_dict: Dict of constant fluxes set in this optimization.
    :param bounds: Lower and upper bounds for all fluxes.
    :param optimization_repeat_time: Repeat time of optimization.
    :param label: Label of this result.
    :param fitted:  Whether the model is fitted. Used in unfitted negative control.
    :param other_parameters: Placeholder for other parameters.
    :return: A Result object that contains optimization result.
    """

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
    bound_object = scipy.optimize.Bounds(*bounds)
    start_vector = start_point_generator(
        complete_balance_matrix, complete_balance_vector, bounds)
    # gradient_validation(cross_entropy_objective_func, cross_entropy_jacobi, start_vector)
    if start_vector is None:
        result_dict = {}
        obj_value = 999999
        success = False
    else:
        if not fitted:
            obj_value = cross_entropy_objective_func(start_vector)[0][0]
            success = True
            result_dict = {
                flux_name: flux_value for flux_name, flux_value
                in zip(complete_flux_dict.keys(), start_vector)}
        else:
            result_dict = {}
            success = False
            obj_value = 999999
            for _ in range(optimization_repeat_time):
                start_vector = start_point_generator(
                    complete_balance_matrix, complete_balance_vector, bounds)
                if start_vector is None:
                    continue
                current_result = scipy.optimize.minimize(
                    cross_entropy_objective_func, start_vector, method='SLSQP', jac=cross_entropy_jacobi_func,
                    constraints=[eq_cons], options={'ftol': 1e-9, 'maxiter': 500}, bounds=bound_object)  # 'disp': True,
                if current_result.success and current_result.fun < obj_value:
                    result_dict = {
                        flux_name: flux_value for flux_name, flux_value
                        in zip(complete_flux_dict.keys(), current_result.x)}
                    obj_value = current_result.fun
                    success = current_result.success
    return config.Result(result_dict, obj_value, success, optimal_obj_value, label)


def result_processing_each_iteration_template(result: config.Result, contribution_func):
    """
    Function template that process each result during computation. This template will calculate objective value and
    invoke function parameter to calculate contribution ratio.

    :param result: Result object in each computation, including optimized flux result.
    :param contribution_func: Function that takes result object as input and calculates contribution ratio from
        metabolites.
    :return: Dict that contains processed results, such as objective value and contribution ratios.
    """

    processed_dict = {}
    if result.success:
        processed_dict['obj_diff'] = result.obj_value - result.minimal_obj_value
        processed_dict['valid'], processed_dict['contribution_dict'] = contribution_func(
            result.result_dict)
    else:
        processed_dict['obj_diff'] = np.nan
        processed_dict['valid'], processed_dict['contribution_dict'] = contribution_func(
            result.result_dict, empty=True)
    return processed_dict


def parallel_solver_single(
        var_parameter_dict, const_parameter_dict, one_case_solver_func, hook_in_each_iteration):
    """
    Function that runs repetitively for each given parameter set. All parameters, including constant and
    variable parameters, are fed into solver function, and returned results are fed into result processing function.
    Raw result and processed result are returned back.

    :param var_parameter_dict: Dict for all variable parameters.
    :param const_parameter_dict: Dict for all constant parameters.
    :param one_case_solver_func: Solver function that optimizes fluxes in current model.
    :param hook_in_each_iteration: Function that utilized to process result. Currently it is contribution function.
    :return: Raw result object and processed result dict.
    """

    result = one_case_solver_func(**const_parameter_dict, **var_parameter_dict)
    hook_result = result_processing_each_iteration_template(
        result, contribution_func=hook_in_each_iteration)
    return result, hook_result


def parallel_solver(
        data_loader_func, parameter_construction_func,
        one_case_solver_func, hook_in_each_iteration, model_name,
        hook_after_all_iterations, parallel_num, **other_parameters):
    """
    Main function that organizes the whole computation. Parallel computation by Python package multiprocessing is
    introduced to reduce running time in millions of sampled points in solution space. For each set of parameter,
    parallel_solver_single is called to execute computation. After computation for all samples are finished, analysis
    function is called to analyze results, plot figures and save results to files.

    :param data_loader_func: Data loader function to load and check data from data file.
    :param parameter_construction_func: Parameter construction function to prepare dict of all parameters
        for computation.
    :param one_case_solver_func: Core function to optimize flux vector that minimize objective function
        given equality and inequality constraints.
    :param hook_in_each_iteration: Function that utilized to process result. Currently it is contribution function.
    :param model_name: Name of model.
    :param hook_after_all_iterations: Function that analyze results, plot figures and save results to files.
    :param parallel_num: Number of parallel process. If not specified, it will be set based on core number.
    :param other_parameters: Placeholder for other parameters.
    :return: None.
    """

    # manager = multiprocessing.Manager()
    # q = manager.Queue()
    # result = pool.map_async(task, [(x, q) for x in range(10)])
    debug = False

    if parallel_num is None:
        cpu_count = os.cpu_count()
        if cpu_count < 10:
            parallel_num = cpu_count - 1
        else:
            parallel_num = min(cpu_count, 16)
    if parallel_num < 8:
        chunk_size = 40
    else:
        chunk_size = 80

    model_mid_data_dict = data_loader_func(**other_parameters)
    const_parameter_dict, var_parameter_list = parameter_construction_func(
        model_mid_data_dict=model_mid_data_dict, parallel_num=parallel_num, model_name=model_name,
        **other_parameters)

    if not isinstance(var_parameter_list, list):
        var_parameter_list, var_parameter_list2 = it.tee(var_parameter_list)
        total_length = const_parameter_dict['iter_length']
    else:
        var_parameter_list2 = var_parameter_list
        total_length = len(var_parameter_list)

    if debug:
        result_list = []
        hook_result_list = []
        for var_parameter_dict in var_parameter_list:
            result, hook_result = parallel_solver_single(
                var_parameter_dict, const_parameter_dict, one_case_solver_func, hook_in_each_iteration)
            result_list.append(result)
            hook_result_list.append(hook_result)
    else:
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
    if not os.path.isdir(global_output_direct):
        os.mkdir(global_output_direct)
    hook_after_all_iterations(result_list, hook_result_list, const_parameter_dict, var_parameter_list2)


def parser_main():
    """
    Entry function that collects input arguments and calls the main parallel-computation function with appropriate
    parameters. Main parameter is model name, which includes all different experiments that readers could run. Readers
    can also choose to running in test mode, which will consume less time and computation resources, and number of
    processes in parallel computation, which will lead to different running time.

    :return: None.
    """

    parameter_dict = {
        'model1': model_parameter_functions.model1_parameters,
        'model1_m5': model_parameter_functions.model1_m5_parameters,
        'model1_m9': model_parameter_functions.model1_m9_parameters,
        'model1_lactate': model_parameter_functions.model1_lactate_parameters,
        'model1_lactate_m4': model_parameter_functions.model1_lactate_m4_parameters,
        'model1_lactate_m10': model_parameter_functions.model1_lactate_m10_parameters,
        'model1_lactate_m11': model_parameter_functions.model1_lactate_m11_parameters,
        'model1_all': model_parameter_functions.model1_all_tissue,
        'model1_all_m5': model_parameter_functions.model1_all_tissue_m5,
        'model1_all_m9': model_parameter_functions.model1_all_tissue_m9,
        'model1_all_lactate': model_parameter_functions.model1_all_tissue_lactate,
        'model1_all_lactate_m4': model_parameter_functions.model1_all_tissue_lactate_m4,
        'model1_all_lactate_m10': model_parameter_functions.model1_all_tissue_lactate_m10,
        'model1_all_lactate_m11': model_parameter_functions.model1_all_tissue_lactate_m11,
        'model1_all_hypoxia': model_parameter_functions.model1_hypoxia_correction,
        'model1_unfitted': model_parameter_functions.model1_unfitted_parameters,
        'parameter': model_parameter_functions.model1_parameter_sensitivity,
        'model3': model_parameter_functions.model3_parameters,
        'model3_all': model_parameter_functions.model3_all_tissue,
        'model3_all_m5': model_parameter_functions.model3_all_tissue_m5,
        'model3_all_m9': model_parameter_functions.model3_all_tissue_m9,
        'model3_all_lactate': model_parameter_functions.model3_all_tissue_lactate,
        'model3_all_lactate_m4': model_parameter_functions.model3_all_tissue_lactate_m4,
        'model3_all_lactate_m10': model_parameter_functions.model3_all_tissue_lactate_m10,
        'model3_all_lactate_m11': model_parameter_functions.model3_all_tissue_lactate_m11,
        'model3_unfitted': model_parameter_functions.model3_unfitted_parameters,
        'model5': model_parameter_functions.model5_parameters,
        'model5_comb2': model_parameter_functions.model5_comb2_parameters,
        'model5_comb3': model_parameter_functions.model5_comb3_parameters,
        'model5_unfitted': model_parameter_functions.model5_unfitted_parameters,
        'model6': model_parameter_functions.model6_parameters,
        'model6_m2': model_parameter_functions.model6_m2_parameters,
        'model6_m3': model_parameter_functions.model6_m3_parameters,
        'model6_m4': model_parameter_functions.model6_m4_parameters,
        'model6_unfitted': model_parameter_functions.model6_unfitted_parameters,
        'model7': model_parameter_functions.model7_parameters,
        'model7_m2': model_parameter_functions.model7_m2_parameters,
        'model7_m3': model_parameter_functions.model7_m3_parameters,
        'model7_m4': model_parameter_functions.model7_m3_parameters,
        'model7_unfitted': model_parameter_functions.model7_unfitted_parameters}
    parser = argparse.ArgumentParser(description='MFA for multi-tissue model by Shiyu Liu.')
    parser.add_argument(
        'model_name', choices=parameter_dict.keys(), help='The name of model you want to compute.')
    parser.add_argument(
        '-t', '--test_mode', action='store_true', default=False,
        help='Whether the code is executed in test mode, which means less sample number and shorter time.')
    parser.add_argument(
        '-p', '--parallel_num', type=int, default=None,
        help='Number of parallel processes. If not provided, it will be selected according to CPU cores.')

    args = parser.parse_args()
    current_model_parameter_dict = parameter_dict[args.model_name](args.test_mode)
    parallel_solver(
        **current_model_parameter_dict, parallel_num=args.parallel_num, one_case_solver_func=one_case_solver_slsqp)


if __name__ == '__main__':
    parser_main()
