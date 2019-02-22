import os
import itertools as it
import gzip
import pickle
import multiprocessing as mp
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import tqdm

import data_parser
import config
import new_model_main as common_functions

constant_set = config.Constants()
color_set = config.Color()
test_running = config.test_running


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


def data_loader_rabinowitz(data_collection_func, data_collection_kwargs):
    file_path = "data_collection.xlsx"
    experiment_name_prefix = "Sup_Fig_5_fasted"
    label_list = ["glucose"]
    data_collection = data_parser.data_parser(file_path, experiment_name_prefix, label_list)
    data_collection = data_parser.data_checker(
        data_collection, ["glucose", "pyruvate", "lactate"], ["glucose", "pyruvate", "lactate"])
    model_mid_data_dict = data_collection_func(data_collection.mid_data, **data_collection_kwargs)
    return model_mid_data_dict


def data_loader_dan(data_collection_func, data_collection_kwargs):
    file_path = "data_collection_from_Dan.xlsx"
    experiment_name_prefix = "no_tumor"
    label_list = ["glucose"]
    data_collection = data_parser.data_parser(file_path, experiment_name_prefix, label_list)
    data_collection = data_parser.data_checker(
        data_collection, ["glucose", "pyruvate", "lactate"], ["glucose", "pyruvate", "lactate"])
    model_mid_data_dict = data_collection_func(data_collection.mid_data, **data_collection_kwargs)
    return model_mid_data_dict


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

    f1_free_flux = FreeVariable(name='F1', total_num=f1_num, var_range=f1_range, display_interv=f1_display_interv)
    g2_free_flux = FreeVariable(name='G2', total_num=g2_num, var_range=g2_range, display_interv=g2_display_interv)
    min_flux_value = 1
    max_flux_value = 5000
    optimization_repeat_time = 10
    obj_tolerance = 0.1

    balance_list, mid_constraint_list = model1_construction(model_mid_data_dict)
    flux_balance_matrix, flux_balance_constant_vector = common_functions.flux_balance_constraint_constructor(
        balance_list, complete_flux_dict)
    (
        substrate_mid_matrix, flux_sum_matrix, target_mid_vector,
        optimal_obj_value) = common_functions.mid_constraint_constructor(
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
    optimization_repeat_time = 10
    obj_tolerance = 0.3
    f1_range = [min_flux_value, max_free_flux_value]
    g2_range = [min_flux_value, max_free_flux_value]

    if test_running:
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
    flux_balance_matrix, flux_balance_constant_vector = common_functions.flux_balance_constraint_constructor(
        balance_list, complete_flux_dict)
    (
        substrate_mid_matrix, flux_sum_matrix, target_mid_vector,
        optimal_obj_value) = common_functions.mid_constraint_constructor(
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


def parameter_generator_single(free_flux_value, free_fluxes_name_list, constant_flux_dict):
    new_constant_flux_dict = dict(constant_flux_dict)
    new_constant_flux_dict.update(
        {flux_name: value for flux_name, value in zip(free_fluxes_name_list, free_flux_value)})
    var_parameter_dict = {'constant_flux_dict': new_constant_flux_dict}
    return var_parameter_dict


def parameter_generator_parallel(
        constant_flux_dict, free_fluxes_name_list, free_flux_value_list, list_length, parallel_num,
        chunk_size):
    with mp.Pool(processes=parallel_num) as pool:
        raw_result_iter = pool.imap(
            partial(
                parameter_generator_single, constant_flux_dict=constant_flux_dict,
                free_fluxes_name_list=free_fluxes_name_list),
            free_flux_value_list, chunk_size)
        iter_parameter_list = list(tqdm.tqdm(
            raw_result_iter, total=list_length, smoothing=0, maxinterval=5,
            desc="Parameter generation progress"))
    return iter_parameter_list


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
    optimization_repeat_time = 10
    obj_tolerance = 0.5
    ternary_sigma = 0.15
    sample = True

    free_fluxes_name_list = ['F1', 'G2', 'F9', 'G10', 'F3']
    free_fluxes_range_list = [
        [min_flux_value, constant_flux_dict['Fcirc_glc']],
        [min_flux_value, constant_flux_dict['Fcirc_glc']],
        [min_flux_value, constant_flux_dict['Fcirc_pyr']],
        [min_flux_value, constant_flux_dict['Fcirc_pyr']],
        [min_flux_value, constant_flux_dict['Fcirc_lac']],
    ]

    if test_running:
        total_point_num = int(3e3)
        # point_interval_list = [30, 30, 12, 12, 80]
        ternary_resolution = int(2 ** 7)
    else:
        total_point_num = int(3e6)
        # point_interval_list = [10, 10, 4, 4, 20]
        ternary_resolution = int(2 ** 8)
    point_num_each_axis = np.round(np.power(total_point_num, 1 / len(free_fluxes_name_list))).astype('int')

    if sample:
        free_flux_raw_list = [
            np.linspace(*free_fluxes_range, total_point_num) for free_fluxes_range in free_fluxes_range_list]
        for row_index, _ in enumerate(free_fluxes_range_list):
            np.random.shuffle(free_flux_raw_list[row_index])
        free_flux_value_list = np.array(free_flux_raw_list).T
        list_length = total_point_num
    else:
        free_fluxes_sequence_list = [
            np.linspace(*flux_range, point_num_each_axis) for flux_range in free_fluxes_range_list]
        free_flux_value_list = it.product(*free_fluxes_sequence_list)
        list_length = np.prod([len(sequence) for sequence in free_fluxes_sequence_list])

    balance_list, mid_constraint_list = model3_construction(model_mid_data_dict)
    flux_balance_matrix, flux_balance_constant_vector = common_functions.flux_balance_constraint_constructor(
        balance_list, complete_flux_dict)
    (
        substrate_mid_matrix, flux_sum_matrix, target_mid_vector,
        optimal_obj_value) = common_functions.mid_constraint_constructor(
        mid_constraint_list, complete_flux_dict)

    # iter_parameter_list = []
    chunk_size = 1000
    iter_parameter_list = parameter_generator_parallel(
        constant_flux_dict, free_fluxes_name_list, free_flux_value_list, list_length, parallel_num,
        chunk_size)

    const_parameter_dict = {
        'flux_balance_matrix': flux_balance_matrix, 'flux_balance_constant_vector': flux_balance_constant_vector,
        'substrate_mid_matrix': substrate_mid_matrix, 'flux_sum_matrix': flux_sum_matrix,
        'target_mid_vector': target_mid_vector, 'optimal_obj_value': optimal_obj_value,
        'complete_flux_dict': complete_flux_dict, 'min_flux_value': min_flux_value,
        'max_flux_value': max_flux_value,

        'optimization_repeat_time': optimization_repeat_time,
        'obj_tolerance': obj_tolerance, 'output_direct': output_direct,
        'free_fluxes_name_list': free_fluxes_name_list,

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
    optimization_repeat_time = 10
    obj_tolerance = 0.4
    ternary_sigma = 0.15
    sample = False

    free_fluxes_name_list = ['F1', 'G2', 'F9', 'G10', 'F3']
    free_fluxes_range_list = [
        [min_flux_value, fcirc_glc_max],
        [min_flux_value, fcirc_glc_max],
        [min_flux_value, constant_flux_dict['Fcirc_pyr']],
        [min_flux_value, constant_flux_dict['Fcirc_pyr']],
        [min_flux_value, constant_flux_dict['Fcirc_lac']],
    ]

    if test_running:
        total_point_num = int(3e3)
        # point_interval_list = [50, 50, 20, 20, 100]
        ternary_resolution = int(2 ** 7)
    else:
        total_point_num = int(3e6)
        # point_interval_list = [25, 25, 5, 5, 25]
        ternary_resolution = int(2 ** 8)
    point_num_each_axis = np.round(np.power(total_point_num, 1 / len(free_fluxes_name_list))).astype('int')

    if sample:
        free_flux_raw_list = [
            np.linspace(*free_fluxes_range, total_point_num) for free_fluxes_range in free_fluxes_range_list]
        for row_index, _ in enumerate(free_fluxes_range_list):
            np.random.shuffle(free_flux_raw_list[row_index])
        free_flux_value_list = np.array(free_flux_raw_list).T
        list_length = total_point_num
    else:
        free_fluxes_sequence_list = [
            np.linspace(*flux_range, point_num_each_axis) for flux_range in free_fluxes_range_list]
        free_flux_value_list = it.product(*free_fluxes_sequence_list)
        list_length = np.prod([len(sequence) for sequence in free_fluxes_sequence_list])

    balance_list, mid_constraint_list = model4_construction(model_mid_data_dict)
    flux_balance_matrix, flux_balance_constant_vector = common_functions.flux_balance_constraint_constructor(
        balance_list, complete_flux_dict)
    (
        substrate_mid_matrix, flux_sum_matrix, target_mid_vector,
        optimal_obj_value) = common_functions.mid_constraint_constructor(
        mid_constraint_list, complete_flux_dict)

    # iter_parameter_list = []
    chunk_size = 1000
    iter_parameter_list = parameter_generator_parallel(
        constant_flux_dict, free_fluxes_name_list, free_flux_value_list, list_length, parallel_num,
        chunk_size)

    const_parameter_dict = {
        'flux_balance_matrix': flux_balance_matrix, 'flux_balance_constant_vector': flux_balance_constant_vector,
        'substrate_mid_matrix': substrate_mid_matrix, 'flux_sum_matrix': flux_sum_matrix,
        'target_mid_vector': target_mid_vector, 'optimal_obj_value': optimal_obj_value,
        'complete_flux_dict': complete_flux_dict, 'min_flux_value': min_flux_value,
        'max_flux_value': max_flux_value,

        'optimization_repeat_time': optimization_repeat_time,
        'obj_tolerance': obj_tolerance, 'output_direct': output_direct,
        'free_fluxes_name_list': free_fluxes_name_list,

        'ternary_sigma': ternary_sigma, 'ternary_resolution': ternary_resolution
    }
    return const_parameter_dict, iter_parameter_list


def dynamic_range_model5(model_mid_data_dict: dict, total_output_direct, parallel_num, **other_parameters):
    output_direct = "{}/model5".format(total_output_direct)
    if not os.path.isdir(output_direct):
        os.mkdir(output_direct)
    complete_flux_list = ['F{}'.format(i + 1) for i in range(10)] + ['G{}'.format(i + 1) for i in range(9)] + \
                         ['H{}'.format(i + 1) for i in range(9)] + ['Fcirc_lac', 'Fcirc_glc']
    complete_flux_dict = {var: i for i, var in enumerate(complete_flux_list)}
    constant_flux_dict = {'F10': 100, 'Fcirc_lac': 374.4, 'Fcirc_glc': 150.9}

    min_flux_value = 1
    max_flux_value = 5000
    optimization_repeat_time = 10
    obj_tolerance = 0.4
    ternary_sigma = 0.15
    sample = False

    free_fluxes_name_list = ['F1', 'G2', 'H1', 'F3', 'G4']
    free_fluxes_range_list = [
        [min_flux_value, constant_flux_dict['Fcirc_glc']],
        [min_flux_value, constant_flux_dict['Fcirc_glc']],
        [min_flux_value, constant_flux_dict['Fcirc_glc']],
        [min_flux_value, constant_flux_dict['Fcirc_lac']],
        [min_flux_value, constant_flux_dict['Fcirc_lac']],
    ]

    if test_running:
        total_point_num = int(3e3)
        # point_interval_list = [50, 50, 20, 20, 100]
        ternary_resolution = int(2 ** 7)
    else:
        total_point_num = int(3e6)
        # point_interval_list = [25, 25, 5, 5, 25]
        ternary_resolution = int(2 ** 8)
    point_num_each_axis = np.round(np.power(total_point_num, 1 / len(free_fluxes_name_list))).astype('int')

    if sample:
        free_flux_raw_list = [
            np.linspace(*free_fluxes_range, total_point_num) for free_fluxes_range in free_fluxes_range_list]
        for row_index, _ in enumerate(free_fluxes_range_list):
            np.random.shuffle(free_flux_raw_list[row_index])
        free_flux_value_list = np.array(free_flux_raw_list).T
        list_length = total_point_num
    else:
        free_fluxes_sequence_list = [
            np.linspace(*flux_range, point_num_each_axis) for flux_range in free_fluxes_range_list]
        free_flux_value_list = it.product(*free_fluxes_sequence_list)
        list_length = np.prod([len(sequence) for sequence in free_fluxes_sequence_list])

    balance_list, mid_constraint_list = model5_construction(model_mid_data_dict)
    flux_balance_matrix, flux_balance_constant_vector = common_functions.flux_balance_constraint_constructor(
        balance_list, complete_flux_dict)
    (
        substrate_mid_matrix, flux_sum_matrix, target_mid_vector,
        optimal_obj_value) = common_functions.mid_constraint_constructor(
        mid_constraint_list, complete_flux_dict)

    # iter_parameter_list = []
    chunk_size = 1000
    iter_parameter_list = parameter_generator_parallel(
        constant_flux_dict, free_fluxes_name_list, free_flux_value_list, list_length, parallel_num,
        chunk_size)

    const_parameter_dict = {
        'flux_balance_matrix': flux_balance_matrix, 'flux_balance_constant_vector': flux_balance_constant_vector,
        'substrate_mid_matrix': substrate_mid_matrix, 'flux_sum_matrix': flux_sum_matrix,
        'target_mid_vector': target_mid_vector, 'optimal_obj_value': optimal_obj_value,
        'complete_flux_dict': complete_flux_dict, 'min_flux_value': min_flux_value,
        'max_flux_value': max_flux_value,

        'optimization_repeat_time': optimization_repeat_time,
        'obj_tolerance': obj_tolerance, 'output_direct': output_direct,
        'free_fluxes_name_list': free_fluxes_name_list,

        'ternary_sigma': ternary_sigma, 'ternary_resolution': ternary_resolution
    }
    return const_parameter_dict, iter_parameter_list


def dynamic_range_model6(model_mid_data_dict: dict, total_output_direct, **other_parameters):
    output_direct = "{}/model6".format(total_output_direct)
    if not os.path.isdir(output_direct):
        os.mkdir(output_direct)
    complete_flux_list = ['F{}'.format(i + 1) for i in range(10)] + ['G{}'.format(i + 1) for i in range(9)] + \
                         ['Fin', 'Fcirc_lac']
    complete_flux_dict = {var: i for i, var in enumerate(complete_flux_list)}
    constant_flux_dict = {'Fin': 111.1, 'F10': 50, 'Fcirc_lac': 600}

    # f1_range = [1, 250]
    # g2_range = [1, 250]
    min_flux_value = 1
    max_flux_value = 8000
    max_free_flux_value = 300
    optimization_repeat_time = 10
    obj_tolerance = 0.3
    f1_range = [min_flux_value, max_free_flux_value]
    g2_range = [min_flux_value, max_free_flux_value]

    if test_running:
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

    balance_list, mid_constraint_list = model6_construction(model_mid_data_dict)
    flux_balance_matrix, flux_balance_constant_vector = common_functions.flux_balance_constraint_constructor(
        balance_list, complete_flux_dict)
    (
        substrate_mid_matrix, flux_sum_matrix, target_mid_vector,
        optimal_obj_value) = common_functions.mid_constraint_constructor(
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


def dynamic_range_model7(model_mid_data_dict: dict, total_output_direct, parallel_num, **other_parameters):
    output_direct = "{}/model7".format(total_output_direct)
    if not os.path.isdir(output_direct):
        os.mkdir(output_direct)
    complete_flux_list = ['F{}'.format(i + 1) for i in range(12)] + ['G{}'.format(i + 1) for i in range(11)] + \
                         ['H{}'.format(i + 1) for i in range(3)] + ['Fcirc_lac', 'Fcirc_pyr', 'Fin']
    complete_flux_dict = {var: i for i, var in enumerate(complete_flux_list)}
    constant_flux_dict = {'Fin': 111.1, 'F12': 100, 'Fcirc_lac': 400, 'Fcirc_pyr': 200}
    fcirc_glc_max = 300

    min_flux_value = 1
    max_flux_value = 5000
    optimization_repeat_time = 10
    obj_tolerance = 0.4
    ternary_sigma = 0.15
    sample = False

    free_fluxes_name_list = ['F1', 'G2', 'F9', 'G10', 'F3']
    free_fluxes_range_list = [
        [min_flux_value, fcirc_glc_max],
        [min_flux_value, fcirc_glc_max],
        [min_flux_value, constant_flux_dict['Fcirc_pyr']],
        [min_flux_value, constant_flux_dict['Fcirc_pyr']],
        [min_flux_value, constant_flux_dict['Fcirc_lac']],
    ]

    if test_running:
        total_point_num = int(3e3)
        ternary_resolution = int(2 ** 7)
    else:
        total_point_num = int(3e6)
        ternary_resolution = int(2 ** 8)
    point_num_each_axis = np.round(np.power(total_point_num, 1 / len(free_fluxes_name_list))).astype('int')

    if sample:
        free_flux_raw_list = [
            np.linspace(*free_fluxes_range, total_point_num) for free_fluxes_range in free_fluxes_range_list]
        for row_index, _ in enumerate(free_fluxes_range_list):
            np.random.shuffle(free_flux_raw_list[row_index])
        free_flux_value_list = np.array(free_flux_raw_list).T
        list_length = total_point_num
    else:
        free_fluxes_sequence_list = [
            np.linspace(*flux_range, point_num_each_axis) for flux_range in free_fluxes_range_list]
        free_flux_value_list = it.product(*free_fluxes_sequence_list)
        list_length = np.prod([len(sequence) for sequence in free_fluxes_sequence_list])

    balance_list, mid_constraint_list = model7_construction(model_mid_data_dict)
    flux_balance_matrix, flux_balance_constant_vector = common_functions.flux_balance_constraint_constructor(
        balance_list, complete_flux_dict)
    (
        substrate_mid_matrix, flux_sum_matrix, target_mid_vector,
        optimal_obj_value) = common_functions.mid_constraint_constructor(
        mid_constraint_list, complete_flux_dict)

    # iter_parameter_list = []
    chunk_size = 1000
    iter_parameter_list = parameter_generator_parallel(
        constant_flux_dict, free_fluxes_name_list, free_flux_value_list, list_length, parallel_num,
        chunk_size)

    const_parameter_dict = {
        'flux_balance_matrix': flux_balance_matrix, 'flux_balance_constant_vector': flux_balance_constant_vector,
        'substrate_mid_matrix': substrate_mid_matrix, 'flux_sum_matrix': flux_sum_matrix,
        'target_mid_vector': target_mid_vector, 'optimal_obj_value': optimal_obj_value,
        'complete_flux_dict': complete_flux_dict, 'min_flux_value': min_flux_value,
        'max_flux_value': max_flux_value,

        'optimization_repeat_time': optimization_repeat_time,
        'obj_tolerance': obj_tolerance, 'output_direct': output_direct,
        'free_fluxes_name_list': free_fluxes_name_list,

        'ternary_sigma': ternary_sigma, 'ternary_resolution': ternary_resolution
    }
    return const_parameter_dict, iter_parameter_list


def mid_data_loader_model1234(
        data_collection_dict, label_list, mouse_id_list, source_tissue_marker, sink_tissue_marker):
    mid_data_dict = {
        'glc_source': common_functions.collect_all_data(
            data_collection_dict, 'glucose', label_list, source_tissue_marker, mouse_id_list),
        'pyr_source': common_functions.collect_all_data(
            data_collection_dict, 'pyruvate', label_list, source_tissue_marker, mouse_id_list),
        'lac_source': common_functions.collect_all_data(
            data_collection_dict, 'lactate', label_list, source_tissue_marker, mouse_id_list),
        'glc_plasma': common_functions.collect_all_data(
            data_collection_dict, 'glucose', label_list, constant_set.plasma_marker, mouse_id_list),
        'pyr_plasma': common_functions.collect_all_data(
            data_collection_dict, 'pyruvate', label_list, constant_set.plasma_marker, mouse_id_list),
        'lac_plasma': common_functions.collect_all_data(
            data_collection_dict, 'lactate', label_list, constant_set.plasma_marker, mouse_id_list),
        'glc_sink': common_functions.collect_all_data(
            data_collection_dict, 'glucose', label_list, sink_tissue_marker, mouse_id_list),
        'pyr_sink': common_functions.collect_all_data(
            data_collection_dict, 'pyruvate', label_list, sink_tissue_marker, mouse_id_list),
        'lac_sink': common_functions.collect_all_data(
            data_collection_dict, 'lactate', label_list, sink_tissue_marker, mouse_id_list),
        'glc_natural': common_functions.natural_dist(constant_set.c13_ratio, 6),
        'glc_infused': np.array([0, 0, 0, 0, 0, 0, 1], dtype='float'),
        'pyr_to_glc_source': common_functions.collect_all_data(
            data_collection_dict, 'pyruvate', label_list, source_tissue_marker, mouse_id_list, convolve=True),
        'glc_to_pyr_source': common_functions.collect_all_data(
            data_collection_dict, 'glucose', label_list, source_tissue_marker, mouse_id_list, split=3),
        'pyr_to_glc_sink': common_functions.collect_all_data(
            data_collection_dict, 'pyruvate', label_list, sink_tissue_marker, mouse_id_list, convolve=True),
        'glc_to_pyr_sink': common_functions.collect_all_data(
            data_collection_dict, 'glucose', label_list, sink_tissue_marker, mouse_id_list, split=3),
        'glc_to_pyr_plasma': common_functions.collect_all_data(
            data_collection_dict, 'glucose', label_list, constant_set.plasma_marker, mouse_id_list, split=3),
    }

    eps_of_mid = 1e-5
    for name, mid_vector in mid_data_dict.items():
        if abs(np.sum(mid_vector) - 1) > 0.001:
            raise ValueError('Sum of MID is not 1: {}'.format(name))
        mid_data_dict[name] += eps_of_mid
        mid_data_dict[name] /= np.sum(mid_data_dict[name])
    return mid_data_dict


def mid_data_loader_model5(
        data_collection_dict, label_list, mouse_id_list, source_tissue_marker, sink1_tissue_marker,
        sink2_tissue_marker):
    mid_data_dict = {
        'glc_source': common_functions.collect_all_data(
            data_collection_dict, 'glucose', label_list, source_tissue_marker, mouse_id_list),
        'pyr_source': common_functions.collect_all_data(
            data_collection_dict, 'pyruvate', label_list, source_tissue_marker, mouse_id_list),
        'lac_source': common_functions.collect_all_data(
            data_collection_dict, 'lactate', label_list, source_tissue_marker, mouse_id_list),
        'glc_plasma': common_functions.collect_all_data(
            data_collection_dict, 'glucose', label_list, constant_set.plasma_marker, mouse_id_list),
        'pyr_plasma': common_functions.collect_all_data(
            data_collection_dict, 'pyruvate', label_list, constant_set.plasma_marker, mouse_id_list),
        'lac_plasma': common_functions.collect_all_data(
            data_collection_dict, 'lactate', label_list, constant_set.plasma_marker, mouse_id_list),
        'glc_sink1': common_functions.collect_all_data(
            data_collection_dict, 'glucose', label_list, sink1_tissue_marker, mouse_id_list),
        'pyr_sink1': common_functions.collect_all_data(
            data_collection_dict, 'pyruvate', label_list, sink1_tissue_marker, mouse_id_list),
        'lac_sink1': common_functions.collect_all_data(
            data_collection_dict, 'lactate', label_list, sink1_tissue_marker, mouse_id_list),
        'glc_sink2': common_functions.collect_all_data(
            data_collection_dict, 'glucose', label_list, sink2_tissue_marker, mouse_id_list),
        'pyr_sink2': common_functions.collect_all_data(
            data_collection_dict, 'pyruvate', label_list, sink2_tissue_marker, mouse_id_list),
        'lac_sink2': common_functions.collect_all_data(
            data_collection_dict, 'lactate', label_list, sink2_tissue_marker, mouse_id_list),
        'glc_natural': common_functions.natural_dist(constant_set.c13_ratio, 6),
        'glc_infused': np.array([0, 0, 0, 0, 0, 0, 1], dtype='float'),
        'pyr_to_glc_source': common_functions.collect_all_data(
            data_collection_dict, 'pyruvate', label_list, source_tissue_marker, mouse_id_list, convolve=True),
        'glc_to_pyr_source': common_functions.collect_all_data(
            data_collection_dict, 'glucose', label_list, source_tissue_marker, mouse_id_list, split=3),
        'pyr_to_glc_sink1': common_functions.collect_all_data(
            data_collection_dict, 'pyruvate', label_list, sink1_tissue_marker, mouse_id_list, convolve=True),
        'glc_to_pyr_sink1': common_functions.collect_all_data(
            data_collection_dict, 'glucose', label_list, sink1_tissue_marker, mouse_id_list, split=3),
        'pyr_to_glc_sink2': common_functions.collect_all_data(
            data_collection_dict, 'pyruvate', label_list, sink2_tissue_marker, mouse_id_list, convolve=True),
        'glc_to_pyr_sink2': common_functions.collect_all_data(
            data_collection_dict, 'glucose', label_list, sink2_tissue_marker, mouse_id_list, split=3),
        'glc_to_pyr_plasma': common_functions.collect_all_data(
            data_collection_dict, 'glucose', label_list, constant_set.plasma_marker, mouse_id_list, split=3),
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
        'F10': mid_data_dict['glc_natural'], constant_set.target_label: mid_data_dict['glc_source']}
    pyr_source_mid_eq = {
        'F5': mid_data_dict['glc_to_pyr_source'], 'F7': mid_data_dict['lac_source'],
        constant_set.target_label: mid_data_dict['pyr_source']}
    lac_source_mid_eq = {
        'F3': mid_data_dict['lac_plasma'], 'F8': mid_data_dict['pyr_source'],
        constant_set.target_label: mid_data_dict['lac_source']}
    glc_sink_mid_eq = {
        'G1': mid_data_dict['glc_plasma'], 'G6': mid_data_dict['pyr_to_glc_sink'],
        constant_set.target_label: mid_data_dict['glc_sink']}
    pyr_sink_mid_eq = {
        'G5': mid_data_dict['glc_to_pyr_sink'], 'G7': mid_data_dict['lac_sink'],
        constant_set.target_label: mid_data_dict['pyr_sink']}
    lac_sink_mid_eq = {
        'G3': mid_data_dict['lac_plasma'], 'G8': mid_data_dict['pyr_sink'],
        constant_set.target_label: mid_data_dict['lac_sink']}

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
        constant_set.target_label: mid_data_dict['glc_source']}
    pyr_source_mid_eq = {
        'F5': mid_data_dict['glc_to_pyr_source'], 'F7': mid_data_dict['lac_source'],
        constant_set.target_label: mid_data_dict['pyr_source']}
    lac_source_mid_eq = {
        'F3': mid_data_dict['lac_plasma'], 'F8': mid_data_dict['pyr_source'],
        constant_set.target_label: mid_data_dict['lac_source']}
    glc_sink_mid_eq = {
        'G1': mid_data_dict['glc_plasma'], 'G6': mid_data_dict['pyr_to_glc_sink'],
        constant_set.target_label: mid_data_dict['glc_sink']}
    pyr_sink_mid_eq = {
        'G5': mid_data_dict['glc_to_pyr_sink'], 'G7': mid_data_dict['lac_sink'],
        constant_set.target_label: mid_data_dict['pyr_sink']}
    lac_sink_mid_eq = {
        'G3': mid_data_dict['lac_plasma'], 'G8': mid_data_dict['pyr_sink'],
        constant_set.target_label: mid_data_dict['lac_sink']}
    glc_plasma_mid_eq = {
        'F2': mid_data_dict['glc_source'], 'G2': mid_data_dict['glc_sink'],
        'Fin': mid_data_dict['glc_infused'], constant_set.target_label: mid_data_dict['glc_plasma']}

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
        'F10': mid_data_dict['glc_natural'], constant_set.target_label: mid_data_dict['glc_source']}
    pyr_source_mid_eq = {
        'F5': mid_data_dict['glc_to_pyr_source'], 'F7': mid_data_dict['lac_source'],
        constant_set.target_label: mid_data_dict['pyr_source']}
    lac_source_mid_eq = {
        'F3': mid_data_dict['lac_plasma'], 'F8': mid_data_dict['pyr_source'],
        constant_set.target_label: mid_data_dict['lac_source']}
    lac_plasma_mid_eq = {
        'G4': mid_data_dict['lac_sink'], 'F4': mid_data_dict['lac_source'],
        'H2': mid_data_dict['pyr_plasma'], constant_set.target_label: mid_data_dict['lac_plasma']}
    pyr_plasma_mid_eq = {
        'G10': mid_data_dict['pyr_sink'], 'F10': mid_data_dict['pyr_source'],
        'H1': mid_data_dict['glc_to_pyr_plasma'], 'H3': mid_data_dict['lac_plasma'],
        constant_set.target_label: mid_data_dict['pyr_plasma']}
    glc_sink_mid_eq = {
        'G1': mid_data_dict['glc_plasma'], 'G6': mid_data_dict['pyr_to_glc_sink'],
        constant_set.target_label: mid_data_dict['glc_sink']}
    pyr_sink_mid_eq = {
        'G5': mid_data_dict['glc_to_pyr_sink'], 'G7': mid_data_dict['lac_sink'],
        constant_set.target_label: mid_data_dict['pyr_sink']}
    lac_sink_mid_eq = {
        'G3': mid_data_dict['lac_plasma'], 'G8': mid_data_dict['pyr_sink'],
        constant_set.target_label: mid_data_dict['lac_sink']}

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
        constant_set.target_label: mid_data_dict['glc_source']}
    pyr_source_mid_eq = {
        'F5': mid_data_dict['glc_to_pyr_source'], 'F7': mid_data_dict['lac_source'],
        constant_set.target_label: mid_data_dict['pyr_source']}
    lac_source_mid_eq = {
        'F3': mid_data_dict['lac_plasma'], 'F8': mid_data_dict['pyr_source'],
        constant_set.target_label: mid_data_dict['lac_source']}
    glc_plasma_mid_eq = {
        'G2': mid_data_dict['glc_sink'], 'F2': mid_data_dict['glc_source'],
        'Fin': mid_data_dict['glc_infused'], constant_set.target_label: mid_data_dict['glc_plasma']}
    lac_plasma_mid_eq = {
        'G4': mid_data_dict['lac_sink'], 'F4': mid_data_dict['lac_source'],
        'H2': mid_data_dict['pyr_plasma'], constant_set.target_label: mid_data_dict['lac_plasma']}
    pyr_plasma_mid_eq = {
        'G10': mid_data_dict['pyr_sink'], 'F10': mid_data_dict['pyr_source'],
        'H1': mid_data_dict['glc_to_pyr_plasma'], 'H3': mid_data_dict['lac_plasma'],
        constant_set.target_label: mid_data_dict['pyr_plasma']}
    glc_sink_mid_eq = {
        'G1': mid_data_dict['glc_plasma'], 'G6': mid_data_dict['pyr_to_glc_sink'],
        constant_set.target_label: mid_data_dict['glc_sink']}
    pyr_sink_mid_eq = {
        'G5': mid_data_dict['glc_to_pyr_sink'], 'G7': mid_data_dict['lac_sink'],
        constant_set.target_label: mid_data_dict['pyr_sink']}
    lac_sink_mid_eq = {
        'G3': mid_data_dict['lac_plasma'], 'G8': mid_data_dict['pyr_sink'],
        constant_set.target_label: mid_data_dict['lac_sink']}

    balance_list = [
        glc_source_balance_eq, pyr_source_balance_eq, lac_source_balance_eq, glc_plasma_balance_eq,
        pyr_plasma_balance_eq, lac_plasma_balance_eq, glc_sink_balance_eq, pyr_sink_balance_eq,
        lac_sink_balance_eq, lac_circ_balance_eq, pyr_circ_balance_eq]
    mid_constraint_list = [
        glc_source_mid_eq, pyr_source_mid_eq, lac_source_mid_eq, glc_plasma_mid_eq, lac_plasma_mid_eq,
        pyr_plasma_mid_eq, glc_sink_mid_eq, pyr_sink_mid_eq, lac_sink_mid_eq]

    return balance_list, mid_constraint_list


def model5_construction(mid_data_dict):
    # Balance equations:
    glc_source_balance_eq = {'input': ['F1', 'F6', 'F10'], 'output': ['F2', 'F5']}
    pyr_source_balance_eq = {'input': ['F5', 'F7'], 'output': ['F6', 'F8', 'F9']}
    lac_source_balance_eq = {'input': ['F3', 'F8'], 'output': ['F4', 'F7']}
    glc_sink1_balance_eq = {'input': ['G1', 'G6'], 'output': ['G2', 'G5']}
    pyr_sink1_balance_eq = {'input': ['G5', 'G7'], 'output': ['G6', 'G8', 'G9']}
    lac_sink1_balance_eq = {'input': ['G3', 'G8'], 'output': ['G4', 'G7']}
    glc_sink2_balance_eq = {'input': ['H1', 'H6'], 'output': ['H2', 'H5']}
    pyr_sink2_balance_eq = {'input': ['H5', 'H7'], 'output': ['H6', 'H8', 'H9']}
    lac_sink2_balance_eq = {'input': ['H3', 'H8'], 'output': ['H4', 'H7']}

    glc_plasma_balance_eq = {'input': ['F2', 'G2', 'H2'], 'output': ['F1', 'G1', 'H1']}
    lac_plasma_balance_eq = {'input': ['F4', 'G4', 'H4'], 'output': ['F3', 'G3', 'H3']}
    glc_circ_balance_eq = {'input': ['F2', 'G2', 'H2'], 'output': ['Fcirc_glc']}
    lac_circ_balance_eq = {'input': ['F4', 'G4', 'H4'], 'output': ['Fcirc_lac']}

    # MID equations:
    # Source tissue:
    glc_source_mid_eq = {
        'F1': mid_data_dict['glc_plasma'], 'F6': mid_data_dict['pyr_to_glc_source'],
        'F10': mid_data_dict['glc_natural'], constant_set.target_label: mid_data_dict['glc_source']}
    pyr_source_mid_eq = {
        'F5': mid_data_dict['glc_to_pyr_source'], 'F7': mid_data_dict['lac_source'],
        constant_set.target_label: mid_data_dict['pyr_source']}
    lac_source_mid_eq = {
        'F3': mid_data_dict['lac_plasma'], 'F8': mid_data_dict['pyr_source'],
        constant_set.target_label: mid_data_dict['lac_source']}
    # Sink tissue 1:
    glc_sink1_mid_eq = {
        'G1': mid_data_dict['glc_plasma'], 'G6': mid_data_dict['pyr_to_glc_sink1'],
        constant_set.target_label: mid_data_dict['glc_sink1']}
    pyr_sink1_mid_eq = {
        'G5': mid_data_dict['glc_to_pyr_sink1'], 'G7': mid_data_dict['lac_sink1'],
        constant_set.target_label: mid_data_dict['pyr_sink1']}
    lac_sink1_mid_eq = {
        'G3': mid_data_dict['lac_plasma'], 'G8': mid_data_dict['pyr_sink1'],
        constant_set.target_label: mid_data_dict['lac_sink1']}
    # Sink tissue 2:
    glc_sink2_mid_eq = {
        'H1': mid_data_dict['glc_plasma'], 'H6': mid_data_dict['pyr_to_glc_sink2'],
        constant_set.target_label: mid_data_dict['glc_sink2']}
    pyr_sink2_mid_eq = {
        'H5': mid_data_dict['glc_to_pyr_sink2'], 'H7': mid_data_dict['lac_sink2'],
        constant_set.target_label: mid_data_dict['pyr_sink2']}
    lac_sink2_mid_eq = {
        'H3': mid_data_dict['lac_plasma'], 'H8': mid_data_dict['pyr_sink2'],
        constant_set.target_label: mid_data_dict['lac_sink2']}

    balance_list = [
        glc_source_balance_eq, pyr_source_balance_eq, lac_source_balance_eq,
        glc_sink1_balance_eq, pyr_sink1_balance_eq, lac_sink1_balance_eq,
        glc_sink2_balance_eq, pyr_sink2_balance_eq, lac_sink2_balance_eq,
        glc_plasma_balance_eq, lac_plasma_balance_eq, glc_circ_balance_eq, lac_circ_balance_eq]
    mid_constraint_list = [
        glc_source_mid_eq, pyr_source_mid_eq, lac_source_mid_eq, glc_sink1_mid_eq,
        pyr_sink1_mid_eq, lac_sink1_mid_eq, glc_sink2_mid_eq, pyr_sink2_mid_eq, lac_sink2_mid_eq]

    return balance_list, mid_constraint_list


def model6_construction(mid_data_dict):
    # Balance equations:
    glc_source_balance_eq = {'input': ['F1', 'F6', 'F10'], 'output': ['F2', 'F5']}
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
        'F10': mid_data_dict['glc_natural'], constant_set.target_label: mid_data_dict['glc_source']}
    pyr_source_mid_eq = {
        'F5': mid_data_dict['glc_to_pyr_source'], 'F7': mid_data_dict['lac_source'],
        constant_set.target_label: mid_data_dict['pyr_source']}
    lac_source_mid_eq = {
        'F3': mid_data_dict['lac_plasma'], 'F8': mid_data_dict['pyr_source'],
        constant_set.target_label: mid_data_dict['lac_source']}
    glc_sink_mid_eq = {
        'G1': mid_data_dict['glc_plasma'], 'G6': mid_data_dict['pyr_to_glc_sink'],
        constant_set.target_label: mid_data_dict['glc_sink']}
    pyr_sink_mid_eq = {
        'G5': mid_data_dict['glc_to_pyr_sink'], 'G7': mid_data_dict['lac_sink'],
        constant_set.target_label: mid_data_dict['pyr_sink']}
    lac_sink_mid_eq = {
        'G3': mid_data_dict['lac_plasma'], 'G8': mid_data_dict['pyr_sink'],
        constant_set.target_label: mid_data_dict['lac_sink']}
    glc_plasma_mid_eq = {
        'F2': mid_data_dict['glc_source'], 'G2': mid_data_dict['glc_sink'],
        'Fin': mid_data_dict['glc_infused'], constant_set.target_label: mid_data_dict['glc_plasma']}

    balance_list = [
        glc_source_balance_eq, pyr_source_balance_eq, lac_source_balance_eq, glc_plasma_balance_eq,
        lac_plasma_balance_eq, glc_sink_balance_eq, pyr_sink_balance_eq, lac_sink_balance_eq,
        lac_circ_balance_eq]
    mid_constraint_list = [
        glc_source_mid_eq, pyr_source_mid_eq, lac_source_mid_eq, glc_sink_mid_eq,
        pyr_sink_mid_eq, lac_sink_mid_eq, glc_plasma_mid_eq]

    return balance_list, mid_constraint_list


def model7_construction(mid_data_dict):
    # Balance equations:
    glc_source_balance_eq = {'input': ['F1', 'F6', 'F12'], 'output': ['F2', 'F5']}
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
        'F12': mid_data_dict['glc_natural'], constant_set.target_label: mid_data_dict['glc_source']}
    pyr_source_mid_eq = {
        'F5': mid_data_dict['glc_to_pyr_source'], 'F7': mid_data_dict['lac_source'],
        constant_set.target_label: mid_data_dict['pyr_source']}
    lac_source_mid_eq = {
        'F3': mid_data_dict['lac_plasma'], 'F8': mid_data_dict['pyr_source'],
        constant_set.target_label: mid_data_dict['lac_source']}
    glc_plasma_mid_eq = {
        'G2': mid_data_dict['glc_sink'], 'F2': mid_data_dict['glc_source'],
        'Fin': mid_data_dict['glc_infused'], constant_set.target_label: mid_data_dict['glc_plasma']}
    lac_plasma_mid_eq = {
        'G4': mid_data_dict['lac_sink'], 'F4': mid_data_dict['lac_source'],
        'H2': mid_data_dict['pyr_plasma'], constant_set.target_label: mid_data_dict['lac_plasma']}
    pyr_plasma_mid_eq = {
        'G10': mid_data_dict['pyr_sink'], 'F10': mid_data_dict['pyr_source'],
        'H1': mid_data_dict['glc_to_pyr_plasma'], 'H3': mid_data_dict['lac_plasma'],
        constant_set.target_label: mid_data_dict['pyr_plasma']}
    glc_sink_mid_eq = {
        'G1': mid_data_dict['glc_plasma'], 'G6': mid_data_dict['pyr_to_glc_sink'],
        constant_set.target_label: mid_data_dict['glc_sink']}
    pyr_sink_mid_eq = {
        'G5': mid_data_dict['glc_to_pyr_sink'], 'G7': mid_data_dict['lac_sink'],
        constant_set.target_label: mid_data_dict['pyr_sink']}
    lac_sink_mid_eq = {
        'G3': mid_data_dict['lac_plasma'], 'G8': mid_data_dict['pyr_sink'],
        constant_set.target_label: mid_data_dict['lac_sink']}

    balance_list = [
        glc_source_balance_eq, pyr_source_balance_eq, lac_source_balance_eq, glc_plasma_balance_eq,
        pyr_plasma_balance_eq, lac_plasma_balance_eq, glc_sink_balance_eq, pyr_sink_balance_eq,
        lac_sink_balance_eq, lac_circ_balance_eq, pyr_circ_balance_eq]
    mid_constraint_list = [
        glc_source_mid_eq, pyr_source_mid_eq, lac_source_mid_eq, glc_plasma_mid_eq, lac_plasma_mid_eq,
        pyr_plasma_mid_eq, glc_sink_mid_eq, pyr_sink_mid_eq, lac_sink_mid_eq]

    return balance_list, mid_constraint_list


def result_processing_each_iteration_model12(result: common_functions.Result, **other_parameters):
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


def result_processing_each_iteration_model34(result: common_functions.Result, **other_parameters):
    def solve_glucose_contribution_model34(result_dict: dict):
        f56 = result_dict['F5'] - result_dict['F6']
        f78 = result_dict['F7'] - result_dict['F8']
        f910 = result_dict['F9'] - result_dict['F10']
        g56 = result_dict['G5'] - result_dict['G6']
        g78 = result_dict['G7'] - result_dict['G8']
        g910 = result_dict['G9'] - result_dict['G10']

        source_tissue_flux_list = common_functions.calculate_one_tissue_tca_contribution([f56, f78, f910])
        sink_tissue_flux_list = common_functions.calculate_one_tissue_tca_contribution([g56, g78, g910])
        glucose_flux, lactate_flux, pyruvate_flux = [
            source_flux + sink_flux for source_flux, sink_flux in zip(source_tissue_flux_list, sink_tissue_flux_list)]
        total_flux = glucose_flux + lactate_flux + pyruvate_flux
        glucose_ratio = glucose_flux / total_flux
        lactate_ratio = lactate_flux / total_flux
        pyruvate_ratio = pyruvate_flux / total_flux
        return glucose_ratio, lactate_ratio, pyruvate_ratio

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


def result_processing_each_iteration_model5(result: common_functions.Result, **other_parameters):
    def solve_glucose_contribution_model5(result_dict: dict):
        f56 = result_dict['F5'] - result_dict['F6']
        f78 = result_dict['F7'] - result_dict['F8']
        g56 = result_dict['G5'] - result_dict['G6']
        g78 = result_dict['G7'] - result_dict['G8']
        h56 = result_dict['H5'] - result_dict['H6']
        h78 = result_dict['H7'] - result_dict['H8']

        source_tissue_flux_list = common_functions.calculate_one_tissue_tca_contribution([f56, f78])
        sink1_tissue_flux_list = common_functions.calculate_one_tissue_tca_contribution([g56, g78])
        sink2_tissue_flux_list = common_functions.calculate_one_tissue_tca_contribution([h56, h78])
        glucose_flux, lactate_flux = [
            source_flux + sink1_flux + sink2_flux for source_flux, sink1_flux, sink2_flux in zip(
                source_tissue_flux_list, sink1_tissue_flux_list, sink2_tissue_flux_list)]
        total_flux = glucose_flux + lactate_flux
        glucose_ratio = glucose_flux / total_flux
        lactate_ratio = lactate_flux / total_flux
        return glucose_ratio, lactate_ratio

    processed_dict = {}
    # if result.success and current_obj_value - minimal_obj_value < obj_tolerance:
    if result.success:
        processed_dict['obj_diff'] = result.obj_value - result.minimal_obj_value
        processed_dict['valid'] = True
        glucose_contribution, lactate_contribution = solve_glucose_contribution_model5(result.result_dict)
        processed_dict['glucose_contribution'] = glucose_contribution
    else:
        processed_dict['obj_diff'] = np.nan
        processed_dict['valid'] = False
        processed_dict['glucose_contribution'] = -1
    return processed_dict


def model1_print_result(result_dict, constant_flux_dict):
    var_string_list = ["{} = {:.3e}".format(var_name, value) for var_name, value in result_dict.items()]
    const_string_list = ["{} = {:.3f}".format(const_name, value) for const_name, value in constant_flux_dict.items()]
    print("Variables:\n{}\n".format("\n".join(var_string_list)))
    print("Constants:\n{}".format("\n".join(const_string_list)))


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

    common_functions.plot_heat_map(
        valid_matrix, g2_free_flux, f1_free_flux, save_path="{}/dynamic_range.png".format(output_direct))
    common_functions.plot_heat_map(
        glucose_contri_matrix, g2_free_flux, f1_free_flux, cmap='cool', cbar_name='Glucose Contribution',
        save_path="{}/glucose_contribution_heatmap.png".format(output_direct))
    common_functions.plot_heat_map(
        objective_function_matrix, g2_free_flux, f1_free_flux, cmap='cool', cbar_name='Objective difference',
        save_path="{}/objective_function.png".format(output_direct))
    common_functions.plot_heat_map(
        filtered_obj_function_matrix, g2_free_flux, f1_free_flux, cmap='cool',
        cbar_name='Filtered objective difference',
        save_path="{}/filtered_objective_function.png".format(output_direct))

    if len(well_fit_glucose_contri_list) == 0:
        raise ValueError('No point fit the constraint for contribution of carbon sources!')
    common_functions.plot_violin_distribution(
        {'normal': np.array(well_fit_glucose_contri_list)},
        {'normal': color_set.blue},
        save_path="{}/glucose_contribution_violin.png".format(output_direct))

    output_data_dict = {
        'result_list': result_list,
        'processed_result_list': processed_result_list,
        'valid_matrix': valid_matrix,
        'glucose_contri_matrix': glucose_contri_matrix,
        'objective_function_matrix': objective_function_matrix
    }
    with gzip.open("{}/output_data_dict.gz".format(output_direct), 'wb') as f_out:
        pickle.dump(output_data_dict, f_out)

    if test_running:
        plt.show()


def final_result_processing_and_plotting_model34(
        result_list, processed_result_list, const_parameter_dict, var_parameter_list):
    output_direct = const_parameter_dict['output_direct']
    free_fluxes_name_list = const_parameter_dict['free_fluxes_name_list']
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
        free_fluxes_array = np.array([constant_fluxes_dict[flux_name] for flux_name in free_fluxes_name_list])
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
        'result_list': result_list,
        'processed_result_list': processed_result_list,
        'valid_point_list': valid_point_list,
        'invalid_point_list': invalid_point_list,
        'contribution_matrix': contribution_matrix,
        'obj_diff_value_list': obj_diff_value_list,
    }

    with gzip.open("{}/output_data_dict.gz".format(output_direct), 'wb') as f_out:
        pickle.dump(output_data_dict, f_out)

    # Ternary plot for density of contribution
    common_functions.plot_ternary_density(
        contribution_matrix, ternary_sigma, ternary_resolution,
        save_path="{}/glucose_contribution_heatmap.png".format(output_direct))
    # Violin plot for objective function
    common_functions.plot_violin_distribution(
        {'normal': np.array(obj_diff_value_list)},
        {'normal': color_set.blue},
        save_path="{}/objective_function_diff_violin.png".format(output_direct))
    # fig, ax = main_functions.violin_plot({'normal': np.array(obj_diff_value_list)})
    # fig.savefig("{}/objective_function_diff_violin.png".format(output_direct), dpi=fig.dpi)

    if test_running:
        plt.show()


def final_result_processing_and_plotting_model5(
        result_list, processed_result_list, const_parameter_dict, var_parameter_list):
    output_direct = const_parameter_dict['output_direct']
    free_fluxes_name_list = const_parameter_dict['free_fluxes_name_list']
    obj_tolerance = const_parameter_dict['obj_tolerance']

    valid_point_list = []
    invalid_point_list = []
    well_fit_glucose_contri_list = []
    obj_diff_value_list = []

    for solver_result, processed_dict, var_parameter in zip(
            result_list, processed_result_list, var_parameter_list):
        constant_fluxes_dict = var_parameter['constant_flux_dict']
        free_fluxes_array = np.array([constant_fluxes_dict[flux_name] for flux_name in free_fluxes_name_list])
        if processed_dict['valid']:
            valid_point_list.append(free_fluxes_array)
            obj_diff_value_list.append(processed_dict['obj_diff'])
            if processed_dict['obj_diff'] < obj_tolerance:
                well_fit_glucose_contri_list.append(processed_dict['glucose_contribution'])
        else:
            invalid_point_list.append(free_fluxes_array)

    if len(well_fit_glucose_contri_list) == 0:
        raise ValueError('No point fit the constraint for contribution of carbon sources!')
    contribution_matrix = np.array(well_fit_glucose_contri_list)
    output_data_dict = {
        'result_list': result_list,
        'processed_result_list': processed_result_list,
        'valid_point_list': valid_point_list,
        'invalid_point_list': invalid_point_list,
        'contribution_matrix': contribution_matrix,
        'obj_diff_value_list': obj_diff_value_list,
    }

    with gzip.open("{}/output_data_dict.gz".format(output_direct), 'wb') as f_out:
        pickle.dump(output_data_dict, f_out)

    # Violin plot for objective function
    # fig, ax = main_functions.violin_plot({'normal': np.array(obj_diff_value_list)})
    # fig.savefig("{}/objective_function_diff_violin.png".format(output_direct), dpi=fig.dpi)
    common_functions.plot_violin_distribution(
        {'normal': np.array(well_fit_glucose_contri_list)},
        {'normal': color_set.blue},
        save_path="{}/glucose_contribution_violin.png".format(output_direct))
    common_functions.plot_violin_distribution(
        {'normal': np.array(obj_diff_value_list)},
        {'normal': color_set.blue},
        save_path="{}/objective_function_diff_violin.png".format(output_direct))

    if test_running:
        plt.show()


def model1_parameters():
    model_name = "model1"
    total_output_direct = "new_models"
    data_collection_kwargs = {
        'label_list': ["glucose"], 'mouse_id_list': ['M1'],
        'source_tissue_marker': constant_set.liver_marker, 'sink_tissue_marker': constant_set.heart_marker}

    model_mid_data_dict = data_loader_rabinowitz(mid_data_loader_model1234, data_collection_kwargs)

    parameter_construction_func = dynamic_range_model1
    parameter_construction_kwargs = {'total_output_direct': total_output_direct}
    hook_in_each_iteration = result_processing_each_iteration_model12
    hook_after_all_iterations = final_result_processing_and_plotting_model12
    model_construction_func = model1_construction

    output_parameter_dict = {
        'model_name': model_name,
        'model_mid_data_dict': model_mid_data_dict,
        'parameter_construction_func': parameter_construction_func,
        'parameter_construction_kwargs': parameter_construction_kwargs,
        'model_construction_func': model_construction_func,
        'hook_in_each_iteration': hook_in_each_iteration,
        'hook_after_all_iterations': hook_after_all_iterations
    }
    return output_parameter_dict


def model2_parameters():
    model_name = "model2"
    total_output_direct = "new_models"

    data_collection_kwargs = {
        'label_list': ["glucose"], 'mouse_id_list': ['M1'],
        'source_tissue_marker': constant_set.liver_marker, 'sink_tissue_marker': constant_set.muscle_marker}
    model_mid_data_dict = data_loader_dan(mid_data_loader_model1234, data_collection_kwargs)

    parameter_construction_func = dynamic_range_model2
    parameter_construction_kwargs = {'total_output_direct': total_output_direct}
    hook_in_each_iteration = result_processing_each_iteration_model12
    hook_after_all_iterations = final_result_processing_and_plotting_model12
    model_construction_func = model2_construction

    output_parameter_dict = {
        'model_name': model_name,
        'model_mid_data_dict': model_mid_data_dict,
        'parameter_construction_func': parameter_construction_func,
        'parameter_construction_kwargs': parameter_construction_kwargs,
        'model_construction_func': model_construction_func,
        'hook_in_each_iteration': hook_in_each_iteration,
        'hook_after_all_iterations': hook_after_all_iterations
    }
    return output_parameter_dict


def model3_parameters():
    model_name = "model3"
    total_output_direct = "new_models"

    data_collection_kwargs = {
        'label_list': ["glucose"], 'mouse_id_list': ['M1'],
        'source_tissue_marker': constant_set.liver_marker, 'sink_tissue_marker': constant_set.heart_marker}
    model_mid_data_dict = data_loader_rabinowitz(mid_data_loader_model1234, data_collection_kwargs)

    parameter_construction_func = dynamic_range_model3
    parameter_construction_kwargs = {'total_output_direct': total_output_direct}
    hook_in_each_iteration = result_processing_each_iteration_model34
    hook_after_all_iterations = final_result_processing_and_plotting_model34
    model_construction_func = model3_construction

    output_parameter_dict = {
        'model_name': model_name,
        'model_mid_data_dict': model_mid_data_dict,
        'parameter_construction_func': parameter_construction_func,
        'parameter_construction_kwargs': parameter_construction_kwargs,
        'model_construction_func': model_construction_func,
        'hook_in_each_iteration': hook_in_each_iteration,
        'hook_after_all_iterations': hook_after_all_iterations
    }
    return output_parameter_dict


def model4_parameters():
    model_name = "model4"
    total_output_direct = "new_models"

    data_collection_kwargs = {
        'label_list': ["glucose"], 'mouse_id_list': ['M1'],
        'source_tissue_marker': constant_set.liver_marker, 'sink_tissue_marker': constant_set.muscle_marker}
    model_mid_data_dict = data_loader_dan(mid_data_loader_model1234, data_collection_kwargs)

    parameter_construction_func = dynamic_range_model4
    parameter_construction_kwargs = {'total_output_direct': total_output_direct}
    hook_in_each_iteration = result_processing_each_iteration_model34
    hook_after_all_iterations = final_result_processing_and_plotting_model34
    model_construction_func = model4_construction

    output_parameter_dict = {
        'model_name': model_name,
        'model_mid_data_dict': model_mid_data_dict,
        'parameter_construction_func': parameter_construction_func,
        'parameter_construction_kwargs': parameter_construction_kwargs,
        'model_construction_func': model_construction_func,
        'hook_in_each_iteration': hook_in_each_iteration,
        'hook_after_all_iterations': hook_after_all_iterations
    }
    return output_parameter_dict


def model5_parameters():
    model_name = "model5"
    total_output_direct = "new_models"

    data_collection_kwargs = {
        'label_list': ["glucose"], 'mouse_id_list': ['M1'],
        'source_tissue_marker': constant_set.liver_marker, 'sink1_tissue_marker': constant_set.heart_marker,
        'sink2_tissue_marker': constant_set.muscle_marker}
    model_mid_data_dict = data_loader_rabinowitz(mid_data_loader_model5, data_collection_kwargs)

    parameter_construction_func = dynamic_range_model5
    parameter_construction_kwargs = {'total_output_direct': total_output_direct}
    hook_in_each_iteration = result_processing_each_iteration_model5
    hook_after_all_iterations = final_result_processing_and_plotting_model5
    model_construction_func = model5_construction

    output_parameter_dict = {
        'model_name': model_name,
        'model_mid_data_dict': model_mid_data_dict,
        'parameter_construction_func': parameter_construction_func,
        'parameter_construction_kwargs': parameter_construction_kwargs,
        'model_construction_func': model_construction_func,
        'hook_in_each_iteration': hook_in_each_iteration,
        'hook_after_all_iterations': hook_after_all_iterations
    }
    return output_parameter_dict


def model6_parameters():
    model_name = "model6"
    total_output_direct = "new_models"
    data_collection_kwargs = {
        'label_list': ["glucose"], 'mouse_id_list': ['M1'],
        'source_tissue_marker': constant_set.liver_marker, 'sink_tissue_marker': constant_set.muscle_marker}
    model_mid_data_dict = data_loader_dan(mid_data_loader_model1234, data_collection_kwargs)

    parameter_construction_func = dynamic_range_model6
    parameter_construction_kwargs = {'total_output_direct': total_output_direct}
    hook_in_each_iteration = result_processing_each_iteration_model12
    hook_after_all_iterations = final_result_processing_and_plotting_model12
    model_construction_func = model6_construction

    output_parameter_dict = {
        'model_name': model_name,
        'model_mid_data_dict': model_mid_data_dict,
        'parameter_construction_func': parameter_construction_func,
        'parameter_construction_kwargs': parameter_construction_kwargs,
        'model_construction_func': model_construction_func,
        'hook_in_each_iteration': hook_in_each_iteration,
        'hook_after_all_iterations': hook_after_all_iterations
    }
    return output_parameter_dict


def model7_parameters():
    model_name = "model7"
    total_output_direct = "new_models"
    data_collection_kwargs = {
        'label_list': ["glucose"], 'mouse_id_list': ['M1'],
        'source_tissue_marker': constant_set.liver_marker, 'sink_tissue_marker': constant_set.muscle_marker}
    model_mid_data_dict = data_loader_dan(mid_data_loader_model1234, data_collection_kwargs)

    parameter_construction_func = dynamic_range_model7
    parameter_construction_kwargs = {'total_output_direct': total_output_direct}
    hook_in_each_iteration = result_processing_each_iteration_model34
    hook_after_all_iterations = final_result_processing_and_plotting_model34
    model_construction_func = model7_construction

    output_parameter_dict = {
        'model_name': model_name,
        'model_mid_data_dict': model_mid_data_dict,
        'parameter_construction_func': parameter_construction_func,
        'parameter_construction_kwargs': parameter_construction_kwargs,
        'model_construction_func': model_construction_func,
        'hook_in_each_iteration': hook_in_each_iteration,
        'hook_after_all_iterations': hook_after_all_iterations
    }
    return output_parameter_dict
