import os
import itertools as it
import gzip
import pickle
import multiprocessing as mp
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import tqdm
from scipy.stats import truncnorm

from src import new_model_main as common_functions, data_parser, config

constant_set = config.Constants()
color_set = config.Color()
empty_2_vector = config.empty_vector(2)
empty_3_vector = config.empty_vector(3)


def data_loader_rabinowitz(
        data_collection_func, data_collection_kwargs,
        experiment_name_prefix="Sup_Fig_5_fasted", **other_parameters):
    file_path = "{}/data_collection.xlsx".format(constant_set.data_direct)
    label_list = data_collection_kwargs['label_list']
    data_collection = data_parser.data_parser(file_path, experiment_name_prefix, label_list)
    data_collection = data_parser.data_checker(
        data_collection, ["glucose", "pyruvate", "lactate"], ["glucose", "pyruvate", "lactate"])
    model_mid_data_dict = data_collection_func(data_collection.mid_data, **data_collection_kwargs)
    return model_mid_data_dict


def data_loader_dan(
        data_collection_func, data_collection_kwargs,
        experiment_name_prefix="no_tumor", **other_parameters):
    file_path = "{}/data_collection_from_Dan.xlsx".format(constant_set.data_direct)
    label_list = data_collection_kwargs['label_list']
    data_collection = data_parser.data_parser(file_path, experiment_name_prefix, label_list)
    data_collection = data_parser.data_checker(
        data_collection, ["glucose", "pyruvate", "lactate"], ["glucose", "pyruvate", "lactate"])
    model_mid_data_dict = data_collection_func(data_collection.mid_data, **data_collection_kwargs)
    return model_mid_data_dict


def dynamic_range_model12(
        model_mid_data_dict: dict, model_construction_func, output_direct, constant_flux_dict, complete_flux_dict,
        optimization_repeat_time, bounds, obj_tolerance,
        f1_num, f1_range, f1_display_interv, g2_num, g2_range, g2_display_interv, model_name, fitted=True,
        **other_parameters):
    def iter_parameter_generator_constructor(
            _f1_free_flux, _g2_free_flux, _constant_flux_dict):
        for f1_index, f1 in enumerate(_f1_free_flux):
            for g2_index, g2 in enumerate(_g2_free_flux):
                new_iter_parameter_dict = {
                    'constant_flux_dict': {
                        _f1_free_flux.flux_name: f1, _g2_free_flux.flux_name: g2},
                    'label': {'matrix_loc': (f1_index, g2_index)}}
                new_iter_parameter_dict['constant_flux_dict'].update(_constant_flux_dict)
                yield new_iter_parameter_dict

    balance_list, mid_constraint_list = model_construction_func(model_mid_data_dict)
    flux_balance_matrix, flux_balance_constant_vector = common_functions.flux_balance_constraint_constructor(
        balance_list, complete_flux_dict)
    (
        substrate_mid_matrix, flux_sum_matrix, target_mid_vector,
        optimal_obj_value) = common_functions.mid_constraint_constructor(
        mid_constraint_list, complete_flux_dict)

    f1_free_flux = config.FreeVariable(
        name='F1', total_num=f1_num, var_range=f1_range, display_interv=f1_display_interv)
    g2_free_flux = config.FreeVariable(
        name='G2', total_num=g2_num, var_range=g2_range, display_interv=g2_display_interv)
    total_iter_num = (f1_num + 1) * (g2_num + 1)

    const_parameter_dict = {
        'flux_balance_matrix': flux_balance_matrix, 'flux_balance_constant_vector': flux_balance_constant_vector,
        'substrate_mid_matrix': substrate_mid_matrix, 'flux_sum_matrix': flux_sum_matrix,
        'target_mid_vector': target_mid_vector, 'optimal_obj_value': optimal_obj_value,
        'complete_flux_dict': complete_flux_dict, 'bounds': bounds, 'iter_length': total_iter_num,
        'raw_constant_flux_dict': constant_flux_dict,

        'optimization_repeat_time': optimization_repeat_time,
        'f1_free_flux': f1_free_flux, 'g2_free_flux': g2_free_flux,
        'obj_tolerance': obj_tolerance, 'output_direct': output_direct,
        'fitted': fitted, 'model_name': model_name
    }
    iter_parameter_list = iter_parameter_generator_constructor(
        f1_free_flux, g2_free_flux, constant_flux_dict)
    return const_parameter_dict, iter_parameter_list


def iter_parameter_generator_constructor_all_tissue(
        _f1_free_flux, _g2_free_flux, _model_parameter_dict_list):
    for model_parameter_dict in _model_parameter_dict_list:
        for f1_index, f1 in enumerate(_f1_free_flux):
            for g2_index, g2 in enumerate(_g2_free_flux):
                new_iter_parameter_dict = model_parameter_dict.copy()
                new_iter_parameter_dict['constant_flux_dict'] = model_parameter_dict['constant_flux_dict'].copy()
                new_iter_parameter_dict['label'] = model_parameter_dict['label'].copy()
                new_iter_parameter_dict['constant_flux_dict'].update({
                    _f1_free_flux.flux_name: f1, _g2_free_flux.flux_name: g2})
                new_iter_parameter_dict['label']['matrix_loc'] = (f1_index, g2_index)
                yield new_iter_parameter_dict


def all_tissue_model1(
        model_mid_data_dict: dict, model_construction_func, output_direct, constant_flux_dict, complete_flux_dict,
        optimization_repeat_time, bounds, obj_tolerance,
        f1_num, f1_range, f1_display_interv, g2_num, g2_range, g2_display_interv, model_name, **other_parameters):
    f1_free_flux = config.FreeVariable(
        name='F1', total_num=f1_num, var_range=f1_range, display_interv=f1_display_interv)
    g2_free_flux = config.FreeVariable(
        name='G2', total_num=g2_num, var_range=g2_range, display_interv=g2_display_interv)
    each_iter_num = (f1_num + 1) * (g2_num + 1)
    total_iter_num = each_iter_num * len(model_mid_data_dict)

    model_parameter_dict_list = []
    for tissue_name, specific_tissue_mid_data_dict in model_mid_data_dict.items():
        balance_list, mid_constraint_list = model_construction_func(specific_tissue_mid_data_dict)
        flux_balance_matrix, flux_balance_constant_vector = common_functions.flux_balance_constraint_constructor(
            balance_list, complete_flux_dict)
        (
            substrate_mid_matrix, flux_sum_matrix, target_mid_vector,
            optimal_obj_value) = common_functions.mid_constraint_constructor(
            mid_constraint_list, complete_flux_dict)
        var_parameter_dict = {
            'flux_balance_matrix': flux_balance_matrix,
            'flux_balance_constant_vector': flux_balance_constant_vector,
            'substrate_mid_matrix': substrate_mid_matrix, 'flux_sum_matrix': flux_sum_matrix,
            'target_mid_vector': target_mid_vector, 'optimal_obj_value': optimal_obj_value,
            'constant_flux_dict': constant_flux_dict,
            'label': {'tissue': tissue_name}}
        model_parameter_dict_list.append(var_parameter_dict)

    const_parameter_dict = {
        'complete_flux_dict': complete_flux_dict, 'bounds': bounds,
        'tissue_name_list': list(model_mid_data_dict.keys()),
        'raw_constant_flux_dict': constant_flux_dict,

        'optimization_repeat_time': optimization_repeat_time,
        'f1_free_flux': f1_free_flux, 'g2_free_flux': g2_free_flux, 'iter_length': total_iter_num,
        'obj_tolerance': obj_tolerance, 'output_direct': output_direct, 'model_name': model_name
    }
    iter_parameter_list = iter_parameter_generator_constructor_all_tissue(
        f1_free_flux, g2_free_flux, model_parameter_dict_list)
    return const_parameter_dict, iter_parameter_list


def all_tissue_hypoxia_correction(
        model_mid_data_dict: dict, model_construction_func, output_direct, constant_flux_dict, complete_flux_dict,
        optimization_repeat_time, bounds, obj_tolerance,
        f1_num, f1_range, f1_display_interv, g2_num, g2_range, g2_display_interv,
        hypoxia_correction_parameter_dict, model_name, **other_parameters):
    def single_mid_correction(mid_vector, final_correction_ratio):
        correction_ratio = final_correction_ratio / (1 - mid_vector[-1] - mid_vector[-1] * final_correction_ratio)
        new_mid_vector = mid_vector.copy()
        new_mid_vector[-1] *= (1 + correction_ratio)
        new_mid_vector /= np.sum(new_mid_vector)
        return new_mid_vector

    def hypoxia_correction(mid_data_dict, parameter_dict):
        new_mid_data_dict = dict(mid_data_dict)
        glc_source_ratio = parameter_dict['glc_source']
        new_mid_data_dict['glc_source'] = single_mid_correction(mid_data_dict['glc_source'], glc_source_ratio)
        lac_sink_ratio = parameter_dict['lac_sink']
        new_mid_data_dict['lac_sink'] = single_mid_correction(mid_data_dict['lac_sink'], lac_sink_ratio)
        pyruvate_sink_ratio = parameter_dict['pyr_sink']
        new_mid_data_dict['pyr_sink'] = single_mid_correction(mid_data_dict['pyr_sink'], pyruvate_sink_ratio)
        return new_mid_data_dict

    f1_free_flux = config.FreeVariable(
        name='F1', total_num=f1_num, var_range=f1_range, display_interv=f1_display_interv)
    g2_free_flux = config.FreeVariable(
        name='G2', total_num=g2_num, var_range=g2_range, display_interv=g2_display_interv)
    each_iter_num = (f1_num + 1) * (g2_num + 1)
    total_iter_num = each_iter_num * len(model_mid_data_dict)

    model_parameter_dict_list = []
    for tissue_name, specific_tissue_mid_data_dict in model_mid_data_dict.items():
        corrected_tissue_mid_data_dict = hypoxia_correction(
            specific_tissue_mid_data_dict, hypoxia_correction_parameter_dict)
        balance_list, mid_constraint_list = model_construction_func(corrected_tissue_mid_data_dict)
        flux_balance_matrix, flux_balance_constant_vector = common_functions.flux_balance_constraint_constructor(
            balance_list, complete_flux_dict)
        (
            substrate_mid_matrix, flux_sum_matrix, target_mid_vector,
            optimal_obj_value) = common_functions.mid_constraint_constructor(
            mid_constraint_list, complete_flux_dict)
        var_parameter_dict = {
            'flux_balance_matrix': flux_balance_matrix,
            'flux_balance_constant_vector': flux_balance_constant_vector,
            'substrate_mid_matrix': substrate_mid_matrix, 'flux_sum_matrix': flux_sum_matrix,
            'target_mid_vector': target_mid_vector, 'optimal_obj_value': optimal_obj_value,
            'constant_flux_dict': constant_flux_dict,
            'label': {'tissue': tissue_name}}
        model_parameter_dict_list.append(var_parameter_dict)

    const_parameter_dict = {
        'complete_flux_dict': complete_flux_dict, 'bounds': bounds,
        'tissue_name_list': list(model_mid_data_dict.keys()),
        'raw_constant_flux_dict': constant_flux_dict,

        'optimization_repeat_time': optimization_repeat_time,
        'f1_free_flux': f1_free_flux, 'g2_free_flux': g2_free_flux, 'iter_length': total_iter_num,
        'obj_tolerance': obj_tolerance, 'output_direct': output_direct, 'model_name': model_name
    }
    iter_parameter_list = iter_parameter_generator_constructor_all_tissue(
        f1_free_flux, g2_free_flux, model_parameter_dict_list)
    return const_parameter_dict, iter_parameter_list


def parameter_sensitivity_model1(
        model_mid_data_dict: dict, model_construction_func, output_direct, constant_flux_dict, complete_flux_dict,
        optimization_repeat_time, bounds, obj_tolerance, sigma_dict,
        parameter_sampling_num, deviation_factor_dict,
        f1_num, f1_range, f1_display_interv, g2_num, g2_range, g2_display_interv, model_name,
        **other_parameters):
    def iter_parameter_generator_constructor(
            _f1_free_flux, _g2_free_flux, _model_parameter_dict_list):
        for model_parameter_dict in _model_parameter_dict_list:
            for f1_index, f1 in enumerate(_f1_free_flux):
                for g2_index, g2 in enumerate(_g2_free_flux):
                    new_iter_parameter_dict = model_parameter_dict.copy()
                    new_iter_parameter_dict['constant_flux_dict'] = model_parameter_dict['constant_flux_dict'].copy()
                    new_iter_parameter_dict['label'] = model_parameter_dict['label'].copy()
                    new_iter_parameter_dict['constant_flux_dict'].update({
                        _f1_free_flux.flux_name: f1, _g2_free_flux.flux_name: g2})
                    new_iter_parameter_dict['label']['matrix_loc'] = (f1_index, g2_index)
                    yield new_iter_parameter_dict

    def construct_model_parameter(
            _current_mid_data_dict, _constant_flux_dict, _sample_type, _sample_index):
        balance_list, mid_constraint_list = model_construction_func(_current_mid_data_dict)
        flux_balance_matrix, flux_balance_constant_vector = common_functions.flux_balance_constraint_constructor(
            balance_list, complete_flux_dict)
        (
            substrate_mid_matrix, flux_sum_matrix, target_mid_vector,
            optimal_obj_value) = common_functions.mid_constraint_constructor(
            mid_constraint_list, complete_flux_dict)
        model_parameter_dict = {
            'flux_balance_matrix': flux_balance_matrix,
            'flux_balance_constant_vector': flux_balance_constant_vector,
            'substrate_mid_matrix': substrate_mid_matrix, 'flux_sum_matrix': flux_sum_matrix,
            'target_mid_vector': target_mid_vector, 'optimal_obj_value': optimal_obj_value,
            'constant_flux_dict': _constant_flux_dict,
            'label': {
                'sample_type': _sample_type, 'sample_index': _sample_index, }
        }
        return model_parameter_dict

    def model_parameter_generator_constructor(
            _model_mid_data_dict, _mid_sigma, _flux_sigma):
        for _sample_index in range(parameter_sampling_num):
            _current_mid_data_dict = mid_perturbation(_model_mid_data_dict, _mid_sigma)
            _model_parameter_dict = construct_model_parameter(
                _current_mid_data_dict, constant_flux_dict, 'mid', _sample_index)
            yield _model_parameter_dict
        for _constant_flux_name, _constant_flux_value in constant_flux_dict.items():
            for _sample_index in range(parameter_sampling_num):
                _current_constant_flux_dict = dict(constant_flux_dict)
                _current_constant_flux_dict[_constant_flux_name] = perturb_array(
                    _constant_flux_value, _flux_sigma, 0, *deviation_factor_dict['flux'])
                _model_parameter_dict = construct_model_parameter(
                    _model_mid_data_dict, _current_constant_flux_dict, _constant_flux_name, _sample_index)
                yield _model_parameter_dict

    def perturb_array(original_array, sigma, lower_bias, min_deviation_factor, max_deviation_factor):
        if isinstance(original_array, int) or isinstance(original_array, float):
            array_size = 1
        else:
            array_size = len(original_array)
        # absolute_deviation = np.clip(
        #     np.abs(np.random.normal(scale=sigma, size=array_size)), min_deviation_factor, max_deviation_factor)
        absolute_deviation = truncnorm.rvs(
            min_deviation_factor / sigma, max_deviation_factor / sigma, size=array_size) * sigma
        random_sign = np.power(-1, np.random.randint(low=0, high=2, size=len(absolute_deviation)))
        deviation = absolute_deviation * random_sign + 1
        new_array = original_array * deviation + lower_bias
        return new_array

    def mid_perturbation(mid_data_dict, sigma):
        new_mid_data_dict = {}
        for mid_title, mid_array in mid_data_dict.items():
            new_array = perturb_array(
                mid_array, sigma, constant_set.eps_of_mid, *deviation_factor_dict['mid'])
            new_mid_data_dict[mid_title] = new_array / np.sum(new_array)
        return new_mid_data_dict

    f1_free_flux = config.FreeVariable(
        name='F1', total_num=f1_num, var_range=f1_range, display_interv=f1_display_interv)
    g2_free_flux = config.FreeVariable(
        name='G2', total_num=g2_num, var_range=g2_range, display_interv=g2_display_interv)
    each_iter_num = (f1_num + 1) * (g2_num + 1)
    total_iter_num = each_iter_num * parameter_sampling_num * (1 + len(constant_flux_dict))
    mid_sigma = sigma_dict['mid']
    flux_sigma = sigma_dict['flux']
    sample_type_list = ['mid', *constant_flux_dict.keys()]

    const_parameter_dict = {
        'complete_flux_dict': complete_flux_dict, 'sample_type_list': sample_type_list,
        'bounds': bounds,

        'optimization_repeat_time': optimization_repeat_time,
        'f1_free_flux': f1_free_flux, 'g2_free_flux': g2_free_flux, 'iter_length': total_iter_num,
        'obj_tolerance': obj_tolerance, 'output_direct': output_direct, 'model_name': model_name,
    }
    model_parameter_dict_list = model_parameter_generator_constructor(
        model_mid_data_dict, mid_sigma, flux_sigma)
    iter_parameter_list = iter_parameter_generator_constructor(
        f1_free_flux, g2_free_flux, model_parameter_dict_list)
    return const_parameter_dict, iter_parameter_list


def parameter_generator_single(free_flux_value, free_fluxes_name_list, constant_flux_dict):
    var_parameter_dict = {
        'constant_flux_dict': {flux_name: value for flux_name, value in zip(free_fluxes_name_list, free_flux_value)}}
    var_parameter_dict['constant_flux_dict'].update(constant_flux_dict)
    return var_parameter_dict


def parameter_generator_parallel(
        constant_flux_dict, free_fluxes_name_list, free_flux_value_list, list_length, parallel_num,
        chunk_size, model_name):
    with mp.Pool(processes=parallel_num) as pool:
        raw_result_iter = pool.imap(
            partial(
                parameter_generator_single, constant_flux_dict=constant_flux_dict,
                free_fluxes_name_list=free_fluxes_name_list),
            free_flux_value_list, chunk_size)
        iter_parameter_list = list(tqdm.tqdm(
            raw_result_iter, total=list_length, smoothing=0, maxinterval=5,
            desc="Parameter generation progress of {}".format(model_name)))
    return iter_parameter_list


def dynamic_range_model345(
        model_mid_data_dict: dict, model_construction_func, output_direct, constant_flux_dict, complete_flux_dict,
        optimization_repeat_time, bounds, obj_tolerance,
        total_point_num, free_fluxes_name_list, free_fluxes_range_list, ternary_sigma, ternary_resolution,
        model_name, parallel_num, fitted=True, **other_parameters):
    def iter_parameter_generator_constructor(
            _free_fluxes_name_list, _constant_flux_dict, _free_flux_value_list):
        for free_flux_value in _free_flux_value_list:
            new_iter_parameter_dict = {
                'constant_flux_dict': {
                    flux_name: value for flux_name, value in zip(_free_fluxes_name_list, free_flux_value)}
            }
            new_iter_parameter_dict['constant_flux_dict'].update(_constant_flux_dict)
            yield new_iter_parameter_dict

    sample = True
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

    balance_list, mid_constraint_list = model_construction_func(model_mid_data_dict)
    flux_balance_matrix, flux_balance_constant_vector = common_functions.flux_balance_constraint_constructor(
        balance_list, complete_flux_dict)
    (
        substrate_mid_matrix, flux_sum_matrix, target_mid_vector,
        optimal_obj_value) = common_functions.mid_constraint_constructor(
        mid_constraint_list, complete_flux_dict)

    # iter_parameter_list = []
    chunk_size = 1000
    # iter_parameter_list = parameter_generator_parallel(
    #     constant_flux_dict, free_fluxes_name_list, free_flux_value_list, list_length, parallel_num,
    #     chunk_size, model_name)
    iter_parameter_list = iter_parameter_generator_constructor(
        free_fluxes_name_list, constant_flux_dict, free_flux_value_list)
    total_iter_num = total_point_num

    const_parameter_dict = {
        'flux_balance_matrix': flux_balance_matrix, 'flux_balance_constant_vector': flux_balance_constant_vector,
        'substrate_mid_matrix': substrate_mid_matrix, 'flux_sum_matrix': flux_sum_matrix,
        'target_mid_vector': target_mid_vector, 'optimal_obj_value': optimal_obj_value,
        'complete_flux_dict': complete_flux_dict, 'bounds': bounds,
        'raw_constant_flux_dict': constant_flux_dict,

        'optimization_repeat_time': optimization_repeat_time,
        'obj_tolerance': obj_tolerance, 'output_direct': output_direct,
        'free_fluxes_name_list': free_fluxes_name_list,
        'iter_length': total_iter_num, 'fitted': fitted, 'model_name': model_name,
        'parallel_num': parallel_num,

        'ternary_sigma': ternary_sigma, 'ternary_resolution': ternary_resolution
    }
    return const_parameter_dict, iter_parameter_list


def all_tissue_model3(
        model_mid_data_dict: dict, model_construction_func, output_direct, constant_flux_dict, complete_flux_dict,
        optimization_repeat_time, bounds, obj_tolerance, parallel_num,
        total_point_num, free_fluxes_name_list, free_fluxes_range_list, ternary_sigma, ternary_resolution,
        model_name, **other_parameters):
    def iter_parameter_generator_constructor(
            _free_fluxes_name_list, _constant_flux_dict, _free_flux_value_list, _model_parameter_dict_list):
        for model_parameter_dict in _model_parameter_dict_list:
            for free_flux_value in _free_flux_value_list:
                new_iter_parameter_dict = model_parameter_dict.copy()
                new_iter_parameter_dict['constant_flux_dict'] = {
                    flux_name: value for flux_name, value in zip(_free_fluxes_name_list, free_flux_value)}
                new_iter_parameter_dict['constant_flux_dict'].update(_constant_flux_dict)
                yield new_iter_parameter_dict

    free_flux_raw_list = [
        np.linspace(*free_fluxes_range, total_point_num) for free_fluxes_range in free_fluxes_range_list]
    for row_index, _ in enumerate(free_fluxes_range_list):
        np.random.shuffle(free_flux_raw_list[row_index])
    free_flux_value_list = np.array(free_flux_raw_list).T
    # list_length = total_point_num
    # chunk_size = 1000
    # constant_flux_dict_list = parameter_generator_parallel(
    #     constant_flux_dict, free_fluxes_name_list, free_flux_value_list, list_length, parallel_num,
    #     chunk_size, model_name)

    total_iter_length = len(model_mid_data_dict) * total_point_num
    model_parameter_dict_list = []
    for tissue_name, specific_tissue_mid_data_dict in model_mid_data_dict.items():
        balance_list, mid_constraint_list = model_construction_func(specific_tissue_mid_data_dict)
        flux_balance_matrix, flux_balance_constant_vector = common_functions.flux_balance_constraint_constructor(
            balance_list, complete_flux_dict)
        (
            substrate_mid_matrix, flux_sum_matrix, target_mid_vector,
            optimal_obj_value) = common_functions.mid_constraint_constructor(
            mid_constraint_list, complete_flux_dict)
        var_parameter_dict = {
            'flux_balance_matrix': flux_balance_matrix,
            'flux_balance_constant_vector': flux_balance_constant_vector,
            'substrate_mid_matrix': substrate_mid_matrix, 'flux_sum_matrix': flux_sum_matrix,
            'target_mid_vector': target_mid_vector, 'optimal_obj_value': optimal_obj_value,
            'label': {'tissue': tissue_name}}
        model_parameter_dict_list.append(var_parameter_dict)

    const_parameter_dict = {
        'complete_flux_dict': complete_flux_dict, 'bounds': bounds,
        'tissue_name_list': list(model_mid_data_dict.keys()),
        'raw_constant_flux_dict': constant_flux_dict,

        'optimization_repeat_time': optimization_repeat_time,
        'obj_tolerance': obj_tolerance, 'output_direct': output_direct,
        'free_fluxes_name_list': free_fluxes_name_list,
        'iter_length': total_iter_length, 'model_name': model_name,
        'parallel_num': parallel_num,

        'ternary_sigma': ternary_sigma, 'ternary_resolution': ternary_resolution
    }

    iter_parameter_generator = iter_parameter_generator_constructor(
        free_fluxes_name_list, constant_flux_dict, free_flux_value_list, model_parameter_dict_list)
    return const_parameter_dict, iter_parameter_generator


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
        'glc_label': np.array([0, 0, 0, 0, 0, 0, 1], dtype='float'),
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

    for name, mid_vector in mid_data_dict.items():
        if abs(np.sum(mid_vector) - 1) > 0.001:
            raise ValueError('Sum of MID is not 1: {}'.format(name))
        mid_data_dict[name] += constant_set.eps_of_mid
        mid_data_dict[name] /= np.sum(mid_data_dict[name])
    return mid_data_dict


def mid_data_loader_all_tissue(
        data_collection_dict, label_list, mouse_id_list, source_tissue_marker, sink_tissue_marker_list):
    total_mid_data_dict = {}
    for sink_tissue_marker in sink_tissue_marker_list:
        total_mid_data_dict[sink_tissue_marker] = mid_data_loader_model1234(
            data_collection_dict, label_list, mouse_id_list, source_tissue_marker, sink_tissue_marker)
    return total_mid_data_dict


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
        'glc_label': np.array([0, 0, 0, 0, 0, 0, 1], dtype='float'),
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

    for name, mid_vector in mid_data_dict.items():
        if abs(np.sum(mid_vector) - 1) > 0.001:
            raise ValueError('Sum of MID is not 1: {}'.format(name))
        mid_data_dict[name] += constant_set.eps_of_mid
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
    glc_plasma_balance_eq = {'input': ['F2', 'G2', 'Jin'], 'output': ['F1', 'G1']}
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
        'Jin': mid_data_dict['glc_label'], constant_set.target_label: mid_data_dict['glc_plasma']}

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
    glc_plasma_balance_eq = {'input': ['F2', 'G2'], 'output': ['F1', 'G1', 'J1']}
    pyr_plasma_balance_eq = {'input': ['F10', 'G10', 'J1', 'J3'], 'output': ['F9', 'G9', 'J2']}
    lac_plasma_balance_eq = {'input': ['F4', 'G4', 'J2'], 'output': ['F3', 'G3', 'J3']}
    glc_sink_balance_eq = {'input': ['G1', 'G6'], 'output': ['G2', 'G5']}
    pyr_sink_balance_eq = {'input': ['G5', 'G7', 'G9'], 'output': ['G6', 'G8', 'G10', 'G11']}
    lac_sink_balance_eq = {'input': ['G3', 'G8'], 'output': ['G4', 'G7']}
    glc_circ_balance_eq = {'input': ['F1', 'G1'], 'output': ['Fcirc_glc']}
    lac_circ_balance_eq = {'input': ['F3', 'G3'], 'output': ['Fcirc_lac']}
    pyr_circ_balance_eq = {'input': ['F9', 'G9'], 'output': ['Fcirc_pyr']}

    # MID equations:

    glc_source_mid_eq = {
        'F1': mid_data_dict['glc_plasma'], 'F6': mid_data_dict['pyr_to_glc_source'],
        'F12': mid_data_dict['glc_natural'], constant_set.target_label: mid_data_dict['glc_source']}
    pyr_source_mid_eq = {
        'F5': mid_data_dict['glc_to_pyr_source'], 'F7': mid_data_dict['lac_source'],
        'F9': mid_data_dict['pyr_plasma'], constant_set.target_label: mid_data_dict['pyr_source']}
    lac_source_mid_eq = {
        'F3': mid_data_dict['lac_plasma'], 'F8': mid_data_dict['pyr_source'],
        constant_set.target_label: mid_data_dict['lac_source']}
    lac_plasma_mid_eq = {
        'G4': mid_data_dict['lac_sink'], 'F4': mid_data_dict['lac_source'],
        'J2': mid_data_dict['pyr_plasma'], constant_set.target_label: mid_data_dict['lac_plasma']}
    pyr_plasma_mid_eq = {
        'G10': mid_data_dict['pyr_sink'], 'F10': mid_data_dict['pyr_source'],
        'J1': mid_data_dict['glc_to_pyr_plasma'], 'J3': mid_data_dict['lac_plasma'],
        constant_set.target_label: mid_data_dict['pyr_plasma']}
    glc_sink_mid_eq = {
        'G1': mid_data_dict['glc_plasma'], 'G6': mid_data_dict['pyr_to_glc_sink'],
        constant_set.target_label: mid_data_dict['glc_sink']}
    pyr_sink_mid_eq = {
        'G5': mid_data_dict['glc_to_pyr_sink'], 'G7': mid_data_dict['lac_sink'],
        'G9': mid_data_dict['pyr_plasma'], constant_set.target_label: mid_data_dict['pyr_sink']}
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
    glc_plasma_balance_eq = {'input': ['F2', 'G2', 'Jin'], 'output': ['F1', 'G1', 'H1']}
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
        'Jin': mid_data_dict['glc_label'], constant_set.target_label: mid_data_dict['glc_plasma']}
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
    glc_plasma_balance_eq = {'input': ['F2', 'G2', 'Jin'], 'output': ['F1', 'G1']}
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
        'Jin': mid_data_dict['glc_label'], constant_set.target_label: mid_data_dict['glc_plasma']}

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
    glc_plasma_balance_eq = {'input': ['F2', 'G2', 'Jin'], 'output': ['F1', 'G1', 'J1']}
    pyr_plasma_balance_eq = {'input': ['F10', 'G10', 'J1', 'J3'], 'output': ['F9', 'G9', 'J2']}
    lac_plasma_balance_eq = {'input': ['F4', 'G4', 'J2'], 'output': ['F3', 'G3', 'J3']}
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
        'F9': mid_data_dict['pyr_plasma'], constant_set.target_label: mid_data_dict['pyr_source']}
    lac_source_mid_eq = {
        'F3': mid_data_dict['lac_plasma'], 'F8': mid_data_dict['pyr_source'],
        constant_set.target_label: mid_data_dict['lac_source']}
    glc_plasma_mid_eq = {
        'G2': mid_data_dict['glc_sink'], 'F2': mid_data_dict['glc_source'],
        'Jin': mid_data_dict['glc_label'], constant_set.target_label: mid_data_dict['glc_plasma']}
    lac_plasma_mid_eq = {
        'G4': mid_data_dict['lac_sink'], 'F4': mid_data_dict['lac_source'],
        'J2': mid_data_dict['pyr_plasma'], constant_set.target_label: mid_data_dict['lac_plasma']}
    pyr_plasma_mid_eq = {
        'G10': mid_data_dict['pyr_sink'], 'F10': mid_data_dict['pyr_source'],
        'J1': mid_data_dict['glc_to_pyr_plasma'], 'J3': mid_data_dict['lac_plasma'],
        constant_set.target_label: mid_data_dict['pyr_plasma']}
    glc_sink_mid_eq = {
        'G1': mid_data_dict['glc_plasma'], 'G6': mid_data_dict['pyr_to_glc_sink'],
        constant_set.target_label: mid_data_dict['glc_sink']}
    pyr_sink_mid_eq = {
        'G5': mid_data_dict['glc_to_pyr_sink'], 'G7': mid_data_dict['lac_sink'],
        'G9': mid_data_dict['pyr_plasma'], constant_set.target_label: mid_data_dict['pyr_sink']}
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


def solve_glucose_contribution_model12_old(result_dict: dict):
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


def metabolite_contribution_model12(result_dict: dict, empty=False):
    empty_contribution_dict = {'source': empty_2_vector, 'sink': empty_2_vector, 'total': empty_2_vector, }
    if empty:
        return False, empty_contribution_dict
    f9 = result_dict['F9']
    g9 = result_dict['G9']
    if f9 < 2 or g9 < 2:
        return False, empty_contribution_dict

    f56 = result_dict['F5'] - result_dict['F6']
    f78 = result_dict['F7'] - result_dict['F8']
    g56 = result_dict['G5'] - result_dict['G6']
    g78 = result_dict['G7'] - result_dict['G8']
    # source_glucose_flux = sink_glucose_flux = 0
    # source_lactate_flux = sink_lactate_flux = 0
    # if f56 > 0:
    #     source_glucose_flux += f56
    # else:
    #     source_lactate_flux += f56
    # if f78 > 0:
    #     source_lactate_flux += f78
    # else:
    #     source_glucose_flux += f78
    # if g56 > 0:
    #     sink_glucose_flux += g56
    # else:
    #     sink_lactate_flux += g56
    # if g78 > 0:
    #     sink_lactate_flux += g78
    # else:
    #     sink_glucose_flux += g78
    # source_glucose_ratio = source_glucose_flux / (source_glucose_flux + source_lactate_flux)
    # sink_glucose_ratio = sink_glucose_flux / (sink_glucose_flux + sink_lactate_flux)
    # glucose_flux = source_glucose_flux + sink_glucose_flux
    # lactate_flux = source_lactate_flux + sink_lactate_flux
    # total_glucose_ratio = glucose_flux / (glucose_flux + lactate_flux)
    # glucose_contribution_dict = {
    #         'source': source_glucose_ratio,
    #         'sink': sink_glucose_ratio,
    #         'total': total_glucose_ratio,}
    source_tissue_flux_array = common_functions.calculate_one_tissue_tca_contribution([f56, f78])
    source_ratio_array = source_tissue_flux_array / np.sum(source_tissue_flux_array)
    sink_tissue_flux_array = common_functions.calculate_one_tissue_tca_contribution([g56, g78])
    sink_ratio_array = sink_tissue_flux_array / np.sum(sink_tissue_flux_array)
    total_tissue_flux_array = source_tissue_flux_array + sink_tissue_flux_array
    total_ratio_array = total_tissue_flux_array / np.sum(total_tissue_flux_array)
    contribution_dict = {
        'source': source_ratio_array,
        'sink': sink_ratio_array,
        'total': total_ratio_array, }
    return True, contribution_dict


def metabolite_contribution_model34(result_dict: dict, empty=False):
    empty_contribution_dict = {'source': empty_3_vector, 'sink': empty_3_vector, 'total': empty_3_vector, }
    if empty:
        return False, empty_contribution_dict
    f11 = result_dict['F11']
    g11 = result_dict['G11']
    if f11 < 2 or g11 < 2:
        return False, empty_contribution_dict

    f56 = result_dict['F5'] - result_dict['F6']
    f78 = result_dict['F7'] - result_dict['F8']
    f910 = result_dict['F9'] - result_dict['F10']
    g56 = result_dict['G5'] - result_dict['G6']
    g78 = result_dict['G7'] - result_dict['G8']
    g910 = result_dict['G9'] - result_dict['G10']

    source_tissue_flux_array = common_functions.calculate_one_tissue_tca_contribution([f56, f78, f910])
    source_ratio_array = source_tissue_flux_array / np.sum(source_tissue_flux_array)
    sink_tissue_flux_array = common_functions.calculate_one_tissue_tca_contribution([g56, g78, g910])
    sink_ratio_array = sink_tissue_flux_array / np.sum(sink_tissue_flux_array)
    total_flux_array = source_tissue_flux_array + sink_tissue_flux_array
    total_ratio_array = total_flux_array / np.sum(total_flux_array)
    # glucose_flux, lactate_flux, pyruvate_flux = [
    #     source_flux + sink_flux for source_flux, sink_flux in zip(source_tissue_flux_list, sink_tissue_flux_list)]
    # total_flux = glucose_flux + lactate_flux + pyruvate_flux
    # glucose_ratio = glucose_flux / total_flux
    # lactate_ratio = lactate_flux / total_flux
    # pyruvate_ratio = pyruvate_flux / total_flux
    contribution_dict = {
        'source': source_ratio_array,
        'sink': sink_ratio_array,
        'total': total_ratio_array, }
    return True, contribution_dict


def metabolite_contribution_model5(result_dict: dict, empty=False):
    empty_contribution_dict = {
        'source': empty_2_vector, 'sink1': empty_2_vector, 'sink2': empty_2_vector,
        'sink': empty_2_vector, 'total': empty_2_vector, }
    if empty:
        return False, empty_contribution_dict
    g9 = result_dict['G9']
    h9 = result_dict['H9']
    if g9 < 2 and h9 < 2:
        return False, empty_contribution_dict

    f56 = result_dict['F5'] - result_dict['F6']
    f78 = result_dict['F7'] - result_dict['F8']
    g56 = result_dict['G5'] - result_dict['G6']
    g78 = result_dict['G7'] - result_dict['G8']
    h56 = result_dict['H5'] - result_dict['H6']
    h78 = result_dict['H7'] - result_dict['H8']

    source_tissue_flux_array = common_functions.calculate_one_tissue_tca_contribution([f56, f78])
    source_ratio_array = source_tissue_flux_array / np.sum(source_tissue_flux_array)
    sink1_tissue_flux_array = common_functions.calculate_one_tissue_tca_contribution([g56, g78])
    sink1_ratio_array = sink1_tissue_flux_array / np.sum(sink1_tissue_flux_array)
    sink2_tissue_flux_array = common_functions.calculate_one_tissue_tca_contribution([h56, h78])
    sink2_ratio_array = sink2_tissue_flux_array / np.sum(sink2_tissue_flux_array)
    sink_tissue_flux_array = sink1_tissue_flux_array + sink2_tissue_flux_array
    sink_ratio_array = sink_tissue_flux_array / np.sum(sink_tissue_flux_array)
    total_tissue_flux_array = sink_tissue_flux_array + source_tissue_flux_array
    total_ratio_array = total_tissue_flux_array / np.sum(total_tissue_flux_array)
    # glucose_flux, lactate_flux = [
    #     source_flux + sink1_flux + sink2_flux for source_flux, sink1_flux, sink2_flux in zip(
    #         source_tissue_flux_list, sink1_tissue_flux_list, sink2_tissue_flux_list)]
    # total_flux = glucose_flux + lactate_flux
    # glucose_ratio = glucose_flux / total_flux
    # lactate_ratio = lactate_flux / total_flux
    contribution_dict = {
        'source': source_ratio_array,
        'sink1': sink1_ratio_array,
        'sink2': sink2_ratio_array,
        'sink': sink_ratio_array,
        'total': total_ratio_array, }
    return True, contribution_dict


def model1_print_result(result_dict, constant_flux_dict):
    var_string_list = ["{} = {:.3e}".format(var_name, value) for var_name, value in result_dict.items()]
    const_string_list = ["{} = {:.3f}".format(const_name, value) for const_name, value in constant_flux_dict.items()]
    print("Variables:\n{}\n".format("\n".join(var_string_list)))
    print("Constants:\n{}".format("\n".join(const_string_list)))


def func_parallel_wrap(func_tuple):
    func, kwargs = func_tuple
    func(**kwargs)


def final_processing_dynamic_range_model12(
        result_list, processed_result_list, const_parameter_dict, var_parameter_list):
    f1_free_flux: config.FreeVariable = const_parameter_dict['f1_free_flux']
    g2_free_flux: config.FreeVariable = const_parameter_dict['g2_free_flux']
    output_direct = const_parameter_dict['output_direct']
    obj_tolerance = const_parameter_dict['obj_tolerance']
    raw_constant_flux_dict = const_parameter_dict['raw_constant_flux_dict']
    all_tissue = False
    model_name = const_parameter_dict['model_name']
    tissue_name = constant_set.default_tissue_name
    tissue_name_list = [tissue_name]
    if 'all_tissue' in model_name:
        all_tissue = True
        tissue_name_list = const_parameter_dict['tissue_name_list']
    if not os.path.isdir(output_direct):
        os.mkdir(output_direct)

    valid_matrix = np.zeros([f1_free_flux.total_num, g2_free_flux.total_num])
    well_fitted_count_dict = {tissue_name: 0 for tissue_name in tissue_name_list}

    valid_matrix_dict = {
        tissue_name: np.zeros_like(valid_matrix) for tissue_name in tissue_name_list}
    objective_function_matrix_dict = {
        tissue_name: np.zeros_like(valid_matrix) for tissue_name in tissue_name_list}

    glucose_contri_matrix_tissue_dict = {tissue_name: {} for tissue_name in tissue_name_list}
    well_fit_glucose_contri_tissue_dict = {tissue_name: {} for tissue_name in tissue_name_list}
    feasible_flux_distribution_tissue_dict = {tissue_name: {} for tissue_name in tissue_name_list}
    filtered_obj_list_dict = {tissue_name: [] for tissue_name in tissue_name_list}

    for solver_result, processed_dict, var_parameter_dict in zip(
            result_list, processed_result_list, var_parameter_list):
        if all_tissue:
            tissue_name = solver_result.label['tissue']

        valid_matrix = valid_matrix_dict[tissue_name]
        objective_function_matrix = objective_function_matrix_dict[tissue_name]
        glucose_contri_matrix_dict = glucose_contri_matrix_tissue_dict[tissue_name]
        well_fit_glucose_contri_dict = well_fit_glucose_contri_tissue_dict[tissue_name]
        feasible_flux_distribution_dict = feasible_flux_distribution_tissue_dict[tissue_name]

        matrix_loc = solver_result.label['matrix_loc']
        contribution_dict = processed_dict['contribution_dict']
        obj_diff = processed_dict['obj_diff']
        valid = processed_dict['valid']
        valid_matrix[matrix_loc] = int(valid)
        objective_function_matrix[matrix_loc] = obj_diff
        result_dict = solver_result.result_dict
        if valid and obj_diff < obj_tolerance:
            for flux_name, flux_value in result_dict.items():
                if flux_name not in feasible_flux_distribution_dict:
                    feasible_flux_distribution_dict[flux_name] = []
                feasible_flux_distribution_dict[flux_name].append(flux_value)
            filtered_obj_list_dict[tissue_name].append(obj_diff)
            well_fitted_count_dict[tissue_name] += 1
        for contribution_type, contribution_vector in contribution_dict.items():
            if contribution_type not in glucose_contri_matrix_dict:
                glucose_contri_matrix_dict[contribution_type] = np.zeros_like(valid_matrix)
                well_fit_glucose_contri_dict[contribution_type] = []
            glucose_contri_matrix_dict[contribution_type][matrix_loc] = contribution_vector[0]
            if valid and obj_diff < obj_tolerance:
                well_fit_glucose_contri_dict[contribution_type].append(contribution_vector[0])

    for tissue_name, well_fitted_count in well_fitted_count_dict.items():
        if well_fitted_count == 0:
            raise ValueError('No point fit the constraint for contribution of carbon sources!: {}'.format(tissue_name))
        print('Well fitted number in tissue {} is {}'.format(tissue_name, well_fitted_count))

    raw_output_data_dict = {}
    output_data_dict = {}
    sink_tissue_contribution_dict = {}
    sink_label = 'sink'

    for tissue_name in tissue_name_list:
        valid_matrix = valid_matrix_dict[tissue_name]
        objective_function_matrix = objective_function_matrix_dict[tissue_name]
        glucose_contri_matrix_dict = glucose_contri_matrix_tissue_dict[tissue_name]
        well_fit_glucose_contri_dict = {
            contribution_type: np.array(glucose_contri_list)
            for contribution_type, glucose_contri_list in well_fit_glucose_contri_tissue_dict[tissue_name].items()}
        if all_tissue and sink_label in well_fit_glucose_contri_dict:
            sink_tissue_contribution_dict[tissue_name] = well_fit_glucose_contri_dict[sink_label]
        feasible_flux_distribution_dict = {
            flux_name: np.array(flux_list)
            for flux_name, flux_list in feasible_flux_distribution_tissue_dict[tissue_name].items()
            if flux_name not in raw_constant_flux_dict}

        # filtered_obj_function_matrix = objective_function_matrix.copy()
        # filtered_obj_function_matrix[objective_function_matrix > obj_tolerance] = np.nan
        # filtered_obj_array = np.reshape(filtered_obj_function_matrix, [-1])
        # filtered_obj_array = filtered_obj_array[~np.isnan(filtered_obj_array)]
        filtered_obj_array = np.array(filtered_obj_list_dict[tissue_name])

        raw_output_data_dict[tissue_name] = {
            'result_list': result_list,
            'processed_result_list': processed_result_list,
        }
        output_data_dict[tissue_name] = {
            'valid_matrix': valid_matrix,
            'objective_function_matrix': objective_function_matrix,
            'filtered_obj_array': filtered_obj_array,
            'glucose_contri_matrix_dict': glucose_contri_matrix_dict,
            'well_fit_glucose_contri_dict': well_fit_glucose_contri_dict,
            'feasible_flux_distribution_dict': feasible_flux_distribution_dict,
        }

        # common_functions.plot_heat_map(
        #     valid_matrix, g2_free_flux, f1_free_flux,
        #     save_path="{}/dynamic_range_{}.png".format(output_direct, tissue_name))
        # common_functions.plot_heat_map(
        #     objective_function_matrix, g2_free_flux, f1_free_flux, cmap=color_set.blue_orange_cmap,
        #     cbar_name='Objective difference',
        #     title=tissue_name,
        #     save_path="{}/objective_function_{}.png".format(output_direct, tissue_name))
        # common_functions.plot_heat_map(
        #     filtered_obj_function_matrix, g2_free_flux, f1_free_flux, cmap=color_set.blue_orange_cmap,
        #     cbar_name='Filtered objective difference',
        #     title=tissue_name,
        #     save_path="{}/filtered_objective_function_{}.png".format(output_direct, tissue_name))
        common_functions.plot_box_distribution(
            {'normal': filtered_obj_array},
            title=tissue_name,
            save_path="{}/filtered_objective_distribution_{}.png".format(output_direct, tissue_name))
        common_functions.plot_violin_distribution(
            well_fit_glucose_contri_dict,
            color_set.blue,
            title=tissue_name,
            save_path="{}/glucose_contribution_violin_{}.png".format(output_direct, tissue_name))
        common_functions.plot_box_distribution(
            feasible_flux_distribution_dict, title=tissue_name,
            save_path="{}/flux_distribution_boxplot_{}.png".format(output_direct, tissue_name))

        # for contribution_type, glucose_contri_matrix in glucose_contri_matrix_dict.items():
        #     common_functions.plot_heat_map(
        #         glucose_contri_matrix, g2_free_flux, f1_free_flux, cmap=color_set.blue_orange_cmap,
        #         cbar_name='Glucose Contribution',
        #         title=tissue_name,
        #         save_path="{}/glucose_contribution_heatmap_{}.png".format(output_direct, contribution_type))

    if all_tissue:
        common_functions.plot_violin_distribution(
            sink_tissue_contribution_dict,
            color_set.blue,
            title='all_tissue',
            save_path="{}/all_tissue_glucose_contribution_violin.png".format(output_direct))

    with gzip.open("{}/raw_output_data_dict.gz".format(output_direct), 'wb') as f_out:
        pickle.dump(raw_output_data_dict, f_out)
    with gzip.open("{}/output_data_dict.gz".format(output_direct), 'wb') as f_out:
        pickle.dump(output_data_dict, f_out)

    # plt.show()


def final_processing_dynamic_range_model345(
        result_list, processed_result_list, const_parameter_dict, var_parameter_list):
    output_direct = const_parameter_dict['output_direct']
    free_fluxes_name_list = const_parameter_dict['free_fluxes_name_list']
    ternary_sigma = const_parameter_dict['ternary_sigma']
    ternary_resolution = const_parameter_dict['ternary_resolution']
    obj_tolerance = const_parameter_dict['obj_tolerance']
    model_name = const_parameter_dict['model_name']
    parallel_num = const_parameter_dict['parallel_num']
    raw_constant_flux_dict = const_parameter_dict['raw_constant_flux_dict']
    all_tissue = False
    ternary_mean = False
    tissue_name = constant_set.default_tissue_name
    tissue_name_list = [tissue_name]
    if 'all_tissue' in model_name:
        ternary_mean = True
        all_tissue = True
        tissue_name_list = const_parameter_dict['tissue_name_list']
    if not os.path.isdir(output_direct):
        os.mkdir(output_direct)

    well_fitted_count_dict = {tissue_name: 0 for tissue_name in tissue_name_list}

    valid_point_list_dict = {tissue_name: [] for tissue_name in tissue_name_list}
    invalid_point_list_dict = {tissue_name: [] for tissue_name in tissue_name_list}
    objective_value_list_dict = {tissue_name: [] for tissue_name in tissue_name_list}
    well_fit_contri_tissue_dict = {tissue_name: {} for tissue_name in tissue_name_list}
    feasible_flux_distribution_tissue_dict = {tissue_name: {} for tissue_name in tissue_name_list}

    for solver_result, processed_dict, var_parameter in zip(
            result_list, processed_result_list, var_parameter_list):
        if all_tissue:
            tissue_name = solver_result.label['tissue']

        valid_point_list = valid_point_list_dict[tissue_name]
        invalid_point_list = invalid_point_list_dict[tissue_name]
        objective_value_list = objective_value_list_dict[tissue_name]
        well_fit_contri_list_dict = well_fit_contri_tissue_dict[tissue_name]
        feasible_flux_distribution_dict = feasible_flux_distribution_tissue_dict[tissue_name]

        constant_fluxes_dict = var_parameter['constant_flux_dict']
        obj_diff = processed_dict['obj_diff']
        valid = processed_dict['valid']
        contribution_dict = processed_dict['contribution_dict']
        free_fluxes_array = np.array([constant_fluxes_dict[flux_name] for flux_name in free_fluxes_name_list])
        objective_value_list.append(obj_diff)
        if valid:
            valid_point_list.append(free_fluxes_array)
            if obj_diff < obj_tolerance:
                result_dict = solver_result.result_dict
                well_fitted_count_dict[tissue_name] += 1
                for flux_name, flux_value in result_dict.items():
                    if flux_name not in feasible_flux_distribution_dict:
                        feasible_flux_distribution_dict[flux_name] = []
                    feasible_flux_distribution_dict[flux_name].append(flux_value)
                for contribution_type, contribution_vector in contribution_dict.items():
                    if contribution_type not in well_fit_contri_list_dict:
                        well_fit_contri_list_dict[contribution_type] = []
                    well_fit_contri_list_dict[contribution_type].append(contribution_vector)
        else:
            invalid_point_list.append(free_fluxes_array)

    for tissue_name, well_fitted_count in well_fitted_count_dict.items():
        if well_fitted_count == 0:
            raise ValueError('No point fit the constraint for contribution of carbon sources!: {}'.format(tissue_name))
        print('Well fitted number in tissue {} is {}'.format(tissue_name, well_fitted_count))

    raw_output_data_dict = {}
    output_data_dict = {}
    project_list = []

    for tissue_name in tissue_name_list:
        valid_point_list = valid_point_list_dict[tissue_name]
        invalid_point_list = invalid_point_list_dict[tissue_name]
        objective_value_list = objective_value_list_dict[tissue_name]
        well_fit_contri_list_dict = {
            contribution_type: np.array(contri_list)
            for contribution_type, contri_list in well_fit_contri_tissue_dict[tissue_name].items()}
        feasible_flux_distribution_dict = {
            flux_name: np.array(flux_list)
            for flux_name, flux_list in feasible_flux_distribution_tissue_dict[tissue_name].items()
            if flux_name not in raw_constant_flux_dict}

        obj_diff_array = np.array(objective_value_list)
        obj_diff_array = obj_diff_array[~np.isnan(obj_diff_array)]
        filtered_obj_array = obj_diff_array[obj_diff_array < obj_tolerance]
        raw_output_data_dict[tissue_name] = {
            'result_list': result_list,
            'processed_result_list': processed_result_list,
        }
        output_data_dict[tissue_name] = {
            'valid_point_list': valid_point_list,
            'invalid_point_list': invalid_point_list,
            'obj_diff_array': obj_diff_array,
            'filtered_obj_array': filtered_obj_array,
            'well_fit_contri_list_dict': well_fit_contri_list_dict,
            'feasible_flux_distribution_dict': feasible_flux_distribution_dict,
        }

        # Ternary plot for density of contribution

        if "model5" in model_name:
            contribution_matrix_dict = {
                contribution_type: contribution_matrix[:, 0]
                for contribution_type, contribution_matrix in well_fit_contri_list_dict.items()}
            kwargs = {
                'data_dict': contribution_matrix_dict,
                'color_dict': color_set.blue,
                'title': tissue_name,
                'save_path': "{}/glucose_contribution_violin_{}.png".format(output_direct, tissue_name)}
            project_list.append((common_functions.plot_violin_distribution, kwargs))
        else:
            for contribution_type, contribution_matrix in well_fit_contri_list_dict.items():
                kwargs = {
                    'tri_data_matrix': contribution_matrix, 'sigma': ternary_sigma, 'bin_num': ternary_resolution,
                    'title': "{}_{}".format(contribution_type, tissue_name),
                    'mean': ternary_mean,
                    'save_path': "{}/test_glucose_contribution_heatmap_{}_{}.png".format(
                        output_direct, contribution_type, tissue_name)}
                project_list.append((common_functions.plot_ternary_density, kwargs))
                # common_functions.plot_ternary_density(
                #     contribution_matrix, ternary_sigma, ternary_resolution,
                #     save_path="{}/glucose_contribution_heatmap_{}_{}.png".format(
                #         output_direct, contribution_type, tissue_name))
        # Violin plot for objective function
        kwargs = {
            'data_dict': {'normal': obj_diff_array},
            'color_dict': color_set.blue,
            'cutoff': obj_tolerance,
            'title': tissue_name,
            'save_path': "{}/objective_function_diff_violin_{}.png".format(output_direct, tissue_name)}
        project_list.append((common_functions.plot_violin_distribution, kwargs))
        kwargs = {
            'data_dict': {'normal': filtered_obj_array},
            'title': tissue_name,
            'save_path': "{}/filtered_objective_distribution_{}.png".format(output_direct, tissue_name)}
        project_list.append((common_functions.plot_box_distribution, kwargs))
        kwargs = {
            'data_dict': feasible_flux_distribution_dict,
            'title': tissue_name,
            'save_path': "{}/flux_distribution_boxplot_{}.png".format(output_direct, tissue_name)}
        project_list.append((common_functions.plot_box_distribution, kwargs))

    with mp.Pool(parallel_num) as pool:
        pool.map(func_parallel_wrap, project_list)

    with gzip.open("{}/raw_output_data_dict.gz".format(output_direct), 'wb') as f_out:
        pickle.dump(raw_output_data_dict, f_out)
    with gzip.open("{}/output_data_dict.gz".format(output_direct), 'wb') as f_out:
        pickle.dump(output_data_dict, f_out)

    # plt.show()


def final_processing_parameter_sensitivity_model1(
        result_list, processed_result_list, const_parameter_dict, var_parameter_list):
    output_direct = const_parameter_dict['output_direct']
    obj_tolerance = const_parameter_dict['obj_tolerance']
    sample_type_list = const_parameter_dict['sample_type_list']
    if not os.path.isdir(output_direct):
        os.mkdir(output_direct)

    objective_function_list_dict = {
        sample_type: [] for sample_type in sample_type_list}
    objective_function_median_dict = {
        sample_type: [] for sample_type in sample_type_list}
    well_fit_glucose_contri_dict = {
        sample_type: {} for sample_type in sample_type_list}

    for solver_result, processed_dict in zip(result_list, processed_result_list):
        sample_type = solver_result.label['sample_type']
        sample_index = solver_result.label['sample_index']
        obj_diff = processed_dict['obj_diff']
        contribution_dict = processed_dict['glucose_contribution']
        valid = processed_dict['valid']
        glucose_contri_list_dict = well_fit_glucose_contri_dict[sample_type]
        objective_function_list = objective_function_list_dict[sample_type]

        for contribution_type, contribution_vector in contribution_dict.items():
            if contribution_type not in glucose_contri_list_dict:
                glucose_contri_list_dict[contribution_type] = []
            current_glucose_contribution_dict = glucose_contri_list_dict[contribution_type]
            if valid:
                if sample_index >= len(objective_function_list_dict[sample_type]):
                    objective_function_list.append([])
                    current_glucose_contribution_dict.append([])
                objective_function_list[sample_index].append(obj_diff)
                if obj_diff < obj_tolerance:
                    current_glucose_contribution_dict[sample_index].append(contribution_vector[0])

    for sample_type, sample_obj_list in objective_function_list_dict.items():
        for sample_index, obj_list in enumerate(sample_obj_list):
            objective_function_median_dict[sample_type].append(np.median(obj_list))

    well_fit_glucose_contri_array_dict = {
        sample_type: {} for sample_type in sample_type_list}
    well_fit_median_contri_dict = {
        sample_type: {} for sample_type in sample_type_list}
    for sample_type, glucose_contri_list_dict in well_fit_glucose_contri_dict.items():
        for contribution_type, sample_contri_list in glucose_contri_list_dict.items():
            contri_array_list = []
            median_list = []
            for sample_index, contri_list in enumerate(sample_contri_list):
                if len(contri_list) == 0:
                    continue
                new_array = np.array(contri_list)
                contri_array_list.append(new_array)
                median_list.append(np.median(new_array))
            well_fit_glucose_contri_array_dict[sample_type][contribution_type] = contri_array_list
            well_fit_median_contri_dict[sample_type][contribution_type] = median_list

    for sample_type in sample_type_list:
        common_functions.plot_violin_distribution(
            {sample_type: objective_function_median_dict[sample_type]},
            color_set.purple, cutoff=obj_tolerance,
            save_path="{}/objective_function_violin_parameter_sensitivity_{}.png".format(
                output_direct, sample_type))
        current_sample_type_median_dict = well_fit_median_contri_dict[sample_type]
        common_functions.plot_violin_distribution(
            current_sample_type_median_dict,
            color_set.purple, title=sample_type,
            save_path="{}/glucose_contribution_violin_parameter_sensitivity_{}.png".format(
                output_direct, sample_type))

    output_data_dict = {
        'well_fit_glucose_contri_array_dict': well_fit_glucose_contri_array_dict,
        'objective_function_list_dict': objective_function_list_dict
    }
    with gzip.open("{}/output_data_dict.gz".format(output_direct), 'wb') as f_out:
        pickle.dump(output_data_dict, f_out)

    # plt.show()
