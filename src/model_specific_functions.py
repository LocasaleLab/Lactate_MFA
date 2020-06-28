#!/usr/bin/env python
#  -*- coding: utf-8 -*-
# (C) Shiyu Liu, Locasale Lab, 2019
# Contact: liushiyu1994@gmail.com
# All rights reserved
# Licensed under MIT License (see LICENSE-MIT)

"""
    Functions that provide detailed operations for each specific model.
"""

import os
import itertools as it
import gzip
import pickle
import multiprocessing as mp

import numpy as np
from scipy.stats import truncnorm

from src import common_functions, data_parser, config

# Import common functions and some constants:

collect_all_data = common_functions.collect_all_data
natural_dist = common_functions.natural_dist
flux_balance_constraint_constructor = common_functions.flux_balance_constraint_constructor
mid_constraint_constructor = common_functions.mid_constraint_constructor
calculate_one_tissue_tca_contribution = common_functions.calculate_one_tissue_tca_contribution
evaluation_for_one_flux = common_functions.evaluation_for_one_flux
one_case_mid_prediction = common_functions.one_case_mid_prediction
append_flux_distribution = common_functions.append_flux_distribution
mid_prediction_preparation = common_functions.mid_prediction_preparation

plot_box_distribution = common_functions.plot_box_distribution
plot_violin_distribution = common_functions.plot_violin_distribution
plot_ternary_density = common_functions.plot_ternary_density

constant_set = config.Constants()
color_set = config.Color()
empty_2_vector = config.empty_vector(2)
empty_3_vector = config.empty_vector(3)


def func_parallel_wrap(func_tuple):
    """
    An wrapper to wrap function and its arguments to a one-argument function for parallel execution
    :param func_tuple: A two-element tuple to collect function and its keyword arguments.
    :return: None
    """
    func, kwargs = func_tuple
    func(**kwargs)


# Data loader functions are designed load raw MID data from xlsx file, check their completeness, and
# invoke data collection functions to extract required MIDs.

def data_loader_rabinowitz(
        data_collection_func, data_collection_kwargs,
        experiment_name_prefix="Sup_Fig_5_fasted", **other_parameters):
    """
    Function that load and check data from Rabinowitz's data file.
    :param data_collection_func: Data collection functions that convert raw data to compact data dict.
    :param data_collection_kwargs: Arguments for data collection functions.
    :param experiment_name_prefix: Sheet name in xlsx files, usually also as experiment name.
    :param other_parameters: Placeholder for other parameters.
    :return: MID data dict for certain model.
    """
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
    """
    Function that load and check data from Dan's data file.
    :param data_collection_func: Data collection functions that convert raw data to compact data dict.
    :param data_collection_kwargs: Arguments for data collection functions.
    :param experiment_name_prefix: Sheet name in xlsx files, usually also as experiment name.
    :param other_parameters: Placeholder for other parameters.
    :return: MID data dict for certain model.
    """
    file_path = "{}/data_collection_from_Dan.xlsx".format(constant_set.data_direct)
    label_list = data_collection_kwargs['label_list']
    data_collection = data_parser.data_parser(file_path, experiment_name_prefix, label_list)
    data_collection = data_parser.data_checker(
        data_collection, ["glucose", "pyruvate", "lactate"], ["glucose", "pyruvate", "lactate"])
    model_mid_data_dict = data_collection_func(data_collection.mid_data, **data_collection_kwargs)
    return model_mid_data_dict


# Data collection functions are designed to extract required MIDs from raw MID data based on specified arguments,
# such as experimental labels, mouse and tissue.

def mid_data_collection_model1234(
        data_collection_dict, label_list, mouse_id_list, source_tissue_marker, sink_tissue_marker):
    """
    Function to extract data of required MID from raw data collection. Label list, mouse ID, source and sink tissue
    can be specified as argument. This function can be applied for model1, model3, model6 and model7.

    :param data_collection_dict: Dict of raw data collection.
    :param label_list: Required label list.
    :param mouse_id_list: Required mouse ID list.
    :param source_tissue_marker: Required source tissue.
    :param sink_tissue_marker: Required sink tissue.
    :return: MID data dict with certain requirements.
    """

    mid_data_dict = {
        'glc_source': collect_all_data(
            data_collection_dict, 'glucose', label_list, source_tissue_marker, mouse_id_list),
        'pyr_source': collect_all_data(
            data_collection_dict, 'pyruvate', label_list, source_tissue_marker, mouse_id_list),
        'lac_source': collect_all_data(
            data_collection_dict, 'lactate', label_list, source_tissue_marker, mouse_id_list),
        'glc_plasma': collect_all_data(
            data_collection_dict, 'glucose', label_list, constant_set.plasma_marker, mouse_id_list),
        'pyr_plasma': collect_all_data(
            data_collection_dict, 'pyruvate', label_list, constant_set.plasma_marker, mouse_id_list),
        'lac_plasma': collect_all_data(
            data_collection_dict, 'lactate', label_list, constant_set.plasma_marker, mouse_id_list),
        'glc_sink': collect_all_data(
            data_collection_dict, 'glucose', label_list, sink_tissue_marker, mouse_id_list),
        'pyr_sink': collect_all_data(
            data_collection_dict, 'pyruvate', label_list, sink_tissue_marker, mouse_id_list),
        'lac_sink': collect_all_data(
            data_collection_dict, 'lactate', label_list, sink_tissue_marker, mouse_id_list),
        'glc_natural': natural_dist(constant_set.c13_ratio, 6),
        'glc_label': np.array([0, 0, 0, 0, 0, 0, 1], dtype='float'),
        'pyr_to_glc_source': collect_all_data(
            data_collection_dict, 'pyruvate', label_list, source_tissue_marker, mouse_id_list, convolve=True),
        'glc_to_pyr_source': collect_all_data(
            data_collection_dict, 'glucose', label_list, source_tissue_marker, mouse_id_list, split=3),
        'pyr_to_glc_sink': collect_all_data(
            data_collection_dict, 'pyruvate', label_list, sink_tissue_marker, mouse_id_list, convolve=True),
        'glc_to_pyr_sink': collect_all_data(
            data_collection_dict, 'glucose', label_list, sink_tissue_marker, mouse_id_list, split=3),
        'glc_to_pyr_plasma': collect_all_data(
            data_collection_dict, 'glucose', label_list, constant_set.plasma_marker, mouse_id_list, split=3),
    }

    for name, mid_vector in mid_data_dict.items():
        if abs(np.sum(mid_vector) - 1) > 0.001:
            raise ValueError('Sum of MID is not 1: {}'.format(name))
        mid_data_dict[name] += constant_set.eps_of_mid
        mid_data_dict[name] /= np.sum(mid_data_dict[name])
    return mid_data_dict


def mid_data_collection_all_tissue(
        data_collection_dict, label_list, mouse_id_list, source_tissue_marker, sink_tissue_marker_list):
    """
    Function to extract data of required MID from raw data collection. Label list, mouse ID, source and sink tissue
    can be specified as argument.
    The only difference is sink tissue could be a list, and therefore returned dict is also be a nesting dict.
    This function can be applied for model1 and model3, which has results for all kinds of tissue.

    :param data_collection_dict: Dict of raw data collection.
    :param label_list: Required label list.
    :param mouse_id_list: Required mouse ID list.
    :param source_tissue_marker: Required source tissue.
    :param sink_tissue_marker_list: List of required sink tissue.
    :return: Dict of MID data dict with certain requirements for each kind of sink tissue.
    """

    total_mid_data_dict = {}
    for sink_tissue_marker in sink_tissue_marker_list:
        total_mid_data_dict[sink_tissue_marker] = mid_data_collection_model1234(
            data_collection_dict, label_list, mouse_id_list, source_tissue_marker, sink_tissue_marker)
    return total_mid_data_dict


def mid_data_collection_model5(
        data_collection_dict, label_list, mouse_id_list, source_tissue_marker, sink1_tissue_marker,
        sink2_tissue_marker):
    """
    Function to extract data of required MID from raw data collection. Label list, mouse ID, source and
    two kinds of sink tissue can be specified as argument. This function can be applied for model5.

    :param data_collection_dict: Dict of raw data collection.
    :param label_list: Required label list.
    :param mouse_id_list: Required mouse ID list.
    :param source_tissue_marker: Required source tissue.
    :param sink1_tissue_marker: Required sink tissue 1.
    :param sink2_tissue_marker: Required sink tissue 2.
    :return: MID data dict with certain requirements.
    """

    mid_data_dict = {
        'glc_source': collect_all_data(
            data_collection_dict, 'glucose', label_list, source_tissue_marker, mouse_id_list),
        'pyr_source': collect_all_data(
            data_collection_dict, 'pyruvate', label_list, source_tissue_marker, mouse_id_list),
        'lac_source': collect_all_data(
            data_collection_dict, 'lactate', label_list, source_tissue_marker, mouse_id_list),
        'glc_plasma': collect_all_data(
            data_collection_dict, 'glucose', label_list, constant_set.plasma_marker, mouse_id_list),
        'pyr_plasma': collect_all_data(
            data_collection_dict, 'pyruvate', label_list, constant_set.plasma_marker, mouse_id_list),
        'lac_plasma': collect_all_data(
            data_collection_dict, 'lactate', label_list, constant_set.plasma_marker, mouse_id_list),
        'glc_sink1': collect_all_data(
            data_collection_dict, 'glucose', label_list, sink1_tissue_marker, mouse_id_list),
        'pyr_sink1': collect_all_data(
            data_collection_dict, 'pyruvate', label_list, sink1_tissue_marker, mouse_id_list),
        'lac_sink1': collect_all_data(
            data_collection_dict, 'lactate', label_list, sink1_tissue_marker, mouse_id_list),
        'glc_sink2': collect_all_data(
            data_collection_dict, 'glucose', label_list, sink2_tissue_marker, mouse_id_list),
        'pyr_sink2': collect_all_data(
            data_collection_dict, 'pyruvate', label_list, sink2_tissue_marker, mouse_id_list),
        'lac_sink2': collect_all_data(
            data_collection_dict, 'lactate', label_list, sink2_tissue_marker, mouse_id_list),
        'glc_natural': natural_dist(constant_set.c13_ratio, 6),
        'glc_label': np.array([0, 0, 0, 0, 0, 0, 1], dtype='float'),
        'pyr_to_glc_source': collect_all_data(
            data_collection_dict, 'pyruvate', label_list, source_tissue_marker, mouse_id_list, convolve=True),
        'glc_to_pyr_source': collect_all_data(
            data_collection_dict, 'glucose', label_list, source_tissue_marker, mouse_id_list, split=3),
        'pyr_to_glc_sink1': collect_all_data(
            data_collection_dict, 'pyruvate', label_list, sink1_tissue_marker, mouse_id_list, convolve=True),
        'glc_to_pyr_sink1': collect_all_data(
            data_collection_dict, 'glucose', label_list, sink1_tissue_marker, mouse_id_list, split=3),
        'pyr_to_glc_sink2': collect_all_data(
            data_collection_dict, 'pyruvate', label_list, sink2_tissue_marker, mouse_id_list, convolve=True),
        'glc_to_pyr_sink2': collect_all_data(
            data_collection_dict, 'glucose', label_list, sink2_tissue_marker, mouse_id_list, split=3),
        'glc_to_pyr_plasma': collect_all_data(
            data_collection_dict, 'glucose', label_list, constant_set.plasma_marker, mouse_id_list, split=3),
    }

    for name, mid_vector in mid_data_dict.items():
        if abs(np.sum(mid_vector) - 1) > 0.001:
            raise ValueError('Sum of MID is not 1: {}'.format(name))
        mid_data_dict[name] += constant_set.eps_of_mid
        mid_data_dict[name] /= np.sum(mid_data_dict[name])
    return mid_data_dict


# Parameter construction functions are designed to prepare dict of all parameters for computation. Parameters
# can usually be divided to constant parameters, which are invariable for all sampled points, and iterative parameters,
# which are different in each sampled point.

def dynamic_range_model12(
        model_mid_data_dict: dict, model_construction_func, output_direct, constant_flux_dict, complete_flux_dict,
        optimization_repeat_time, bounds, obj_tolerance,
        f1_num, f1_range, f1_display_interv, g2_num, g2_range, g2_display_interv, model_name, fitted=True,
        **other_parameters):
    """
    Function to generate parameters to evenly sample in 2D solution space and calculate glucose distribution.
    The calculation is only for model with 2 free dimensions, such as model1 and model6.

    :param model_mid_data_dict: MID data dict of current model. Each metabolite can only be mapped to one MID vector.
    :param model_construction_func: Function to construct flux balance equations and MID constraints equations.
    :param output_direct: Direct to output results.
    :param constant_flux_dict: Dict of constant fluxes that are presumed previously in parameters.
    :param complete_flux_dict: Flux name to index dict to ensure all fluxes have same order.
    :param optimization_repeat_time: Number of repeat for optimization in each sampled point.
    :param bounds: Bounds for optimization of all fluxes.
    :param obj_tolerance: Tolerance of objective value to be a feasible solution.
    :param f1_num: Sample number of free flux F1.
    :param f1_range: Sample range of free flux F1.
    :param f1_display_interv: Display interval of free flux F1. Deprecated.
    :param g2_num: Sample number of free flux G2.
    :param g2_range: Sample range of free flux G2.
    :param g2_display_interv: Display interval of free flux G2. Deprecated.
    :param model_name: Model name. Also used as name of folders and progress reminding.
    :param fitted: Whether the model is fitted. Used in unfitted negative control.
    :param other_parameters: Placeholder for other parameters.
    :return: const_parameter_dict: Dict of all constant parameters which are same in all samples.
            iter_parameter_list: List of all variable parameters which are different in each sample.
    """

    def iter_parameter_generator_constructor(
            _f1_free_flux, _g2_free_flux, _constant_flux_dict):
        """
        Construct generator of variable parameters.
        :param _f1_free_flux: Free flux object of F1.
        :param _g2_free_flux: Free flux object of G2.
        :param _constant_flux_dict: Dict of constant fluxes.
        :return: Generator for complete parameter dict of variable parameters.
        """
        for f1_index, f1 in enumerate(_f1_free_flux):
            for g2_index, g2 in enumerate(_g2_free_flux):
                new_iter_parameter_dict = {
                    'constant_flux_dict': {
                        _f1_free_flux.flux_name: f1, _g2_free_flux.flux_name: g2},
                    'label': {'matrix_loc': (f1_index, g2_index)}}
                new_iter_parameter_dict['constant_flux_dict'].update(_constant_flux_dict)
                yield new_iter_parameter_dict

    balance_list, mid_constraint_list = model_construction_func(model_mid_data_dict)
    flux_balance_matrix, flux_balance_constant_vector = flux_balance_constraint_constructor(
        balance_list, complete_flux_dict)
    (
        substrate_mid_matrix, flux_sum_matrix, target_mid_vector,
        optimal_obj_value) = mid_constraint_constructor(
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
        'mid_constraint_list_dict': {'': mid_constraint_list},

        'optimization_repeat_time': optimization_repeat_time,
        'f1_free_flux': f1_free_flux, 'g2_free_flux': g2_free_flux,
        'obj_tolerance': obj_tolerance, 'output_direct': output_direct,
        'fitted': fitted, 'model_name': model_name
    }
    iter_parameter_list = iter_parameter_generator_constructor(
        f1_free_flux, g2_free_flux, constant_flux_dict)
    return const_parameter_dict, iter_parameter_list


def iter_parameter_generator_constructor_all_tissue(
        f1_free_flux, g2_free_flux, model_parameter_dict_list):
    """
    Construct generator of variable parameters.
    Generator form will save memory and construct time when dealing with lots of repetitive parameters.

    :param f1_free_flux: Free flux object of F1.
    :param g2_free_flux: Free flux object of G2.
    :param model_parameter_dict_list: List of other parameter dict except value of free fluxes.
    :return: Generator for complete parameter dict of all variable parameters.
    """
    for model_parameter_dict in model_parameter_dict_list:
        for f1_index, f1 in enumerate(f1_free_flux):
            for g2_index, g2 in enumerate(g2_free_flux):
                new_iter_parameter_dict = model_parameter_dict.copy()
                new_iter_parameter_dict['constant_flux_dict'] = model_parameter_dict['constant_flux_dict'].copy()
                new_iter_parameter_dict['label'] = model_parameter_dict['label'].copy()
                new_iter_parameter_dict['constant_flux_dict'].update({
                    f1_free_flux.flux_name: f1, g2_free_flux.flux_name: g2})
                new_iter_parameter_dict['label']['matrix_loc'] = (f1_index, g2_index)
                yield new_iter_parameter_dict


def all_tissue_model1(
        model_mid_data_dict: dict, model_construction_func, output_direct, constant_flux_dict, complete_flux_dict,
        optimization_repeat_time, bounds, obj_tolerance,
        f1_num, f1_range, f1_display_interv, g2_num, g2_range, g2_display_interv, model_name, **other_parameters):
    """
    Function to generate parameters to evenly sample in 2D solution space and calculate glucose distribution.
    The calculation is only for model with 2 free dimensions, such as model1 and model6.
    This function will deal with data from 8 kinds of sink tissue at the same time in one mouse.

    :param model_mid_data_dict: MID data dict of current model for 8 kinds of sink tissue.
        Key is tissue name and value is MID data dict of each tissue.
    :param model_construction_func: Function to construct flux balance equations and MID constraints equations.
    :param output_direct: Direct to output results.
    :param constant_flux_dict: Dict of constant fluxes that are presumed previously in parameters.
    :param complete_flux_dict: Flux name to index dict to ensure all fluxes have same order.
    :param optimization_repeat_time: Number of repeat for optimization in each sampled point.
    :param bounds: Bounds for optimization of all fluxes.
    :param obj_tolerance: Tolerance of objective value to be a feasible solution.
    :param f1_num: Sample number of free flux F1.
    :param f1_range: Sample range of free flux F1.
    :param f1_display_interv: Display interval of free flux F1. Deprecated.
    :param g2_num: Sample number of free flux G2.
    :param g2_range: Sample range of free flux G2.
    :param g2_display_interv: Display interval of free flux G2. Deprecated.
    :param model_name: Model name. Also used as name of folders and progress reminding.
    :param other_parameters: Placeholder for other parameters.
    :return: const_parameter_dict: Dict of all constant parameters which are same in all samples.
            iter_parameter_list: List of all variable parameters which are different in each sample.
    """

    f1_free_flux = config.FreeVariable(
        name='F1', total_num=f1_num, var_range=f1_range, display_interv=f1_display_interv)
    g2_free_flux = config.FreeVariable(
        name='G2', total_num=g2_num, var_range=g2_range, display_interv=g2_display_interv)
    each_iter_num = (f1_num + 1) * (g2_num + 1)
    total_iter_num = each_iter_num * len(model_mid_data_dict)

    model_parameter_dict_list = []
    mid_constraint_list_dict = {}
    for tissue_name, specific_tissue_mid_data_dict in model_mid_data_dict.items():
        balance_list, mid_constraint_list = model_construction_func(specific_tissue_mid_data_dict)
        mid_constraint_list_dict[tissue_name] = mid_constraint_list
        flux_balance_matrix, flux_balance_constant_vector = flux_balance_constraint_constructor(
            balance_list, complete_flux_dict)
        (
            substrate_mid_matrix, flux_sum_matrix, target_mid_vector,
            optimal_obj_value) = mid_constraint_constructor(
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
        'mid_constraint_list_dict': mid_constraint_list_dict,

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
    """
    Function to generate parameters under presumed hypoxia condition and calculate glucose distribution.
    This function will deal with data from 8 kinds of sink tissue.

    :param model_mid_data_dict: MID data dict of current model for 8 kinds of sink tissue.
        Key is tissue name and value is MID data dict of each tissue.
    :param model_construction_func: Function to construct flux balance equations and MID constraints equations.
    :param output_direct: Direct to output results.
    :param constant_flux_dict: Dict of constant fluxes that are presumed previously in parameters.
    :param complete_flux_dict: Flux name to index dict to ensure all fluxes have same order.
    :param optimization_repeat_time: Number of repeat for optimization in each sampled point.
    :param bounds: Bounds for optimization of all fluxes.
    :param obj_tolerance: Tolerance of objective value to be a feasible solution.
    :param f1_num: Sample number of free flux F1.
    :param f1_range: Sample range of free flux F1.
    :param f1_display_interv: Display interval of free flux F1. Deprecated.
    :param g2_num: Sample number of free flux G2.
    :param g2_range: Sample range of free flux G2.
    :param g2_display_interv: Display interval of free flux G2. Deprecated.
    :param hypoxia_correction_parameter_dict: Parameters for hypoxia correction.
    :param model_name: Model name. Also used as name of folders and progress reminding.
    :param other_parameters: Placeholder for other parameters.
    :return: const_parameter_dict: Dict of all constant parameters which are same in all samples.
            iter_parameter_list: List of all variable parameters which are different in each sample.
    """

    def hypoxia_correction(mid_data_dict_one_case, parameter_dict):
        """
        Correct MID from hypoxia effect in one kind of sink tissue.
        :param mid_data_dict_one_case: Measured MID data in one kind of sink tissue before correction.
        :param parameter_dict: Parameter dict of correction.
        :return: Corrected MID data in one kind of sink tissue.
        """

        def single_mid_correction(mixed_mid_vector, current_input_mid_vector, current_correction_ratio):
            new_mid_vector = (mixed_mid_vector - current_correction_ratio * current_input_mid_vector) / \
                             (1 - current_correction_ratio)
            if np.any(new_mid_vector < 0):
                new_mid_vector[new_mid_vector < 0] = constant_set.eps_of_mid
                new_mid_vector /= np.sum(new_mid_vector)
                # raise ValueError(mixed_mid_vector, current_input_mid_vector)
            return new_mid_vector

        new_mid_data_dict = dict(mid_data_dict_one_case)
        for target_mid_name, correction_ratio in parameter_dict.items():
            if target_mid_name == 'glc_source':
                input_mid_vector = mid_data_dict_one_case['glc_natural']
            elif target_mid_name == 'lac_sink':
                input_mid_vector = mid_data_dict_one_case['pyr_sink']
            elif target_mid_name == 'pyr_sink':
                input_mid_vector = mid_data_dict_one_case['glc_to_pyr_sink']
            else:
                raise ValueError()
            new_mid_data_dict[target_mid_name] = single_mid_correction(
                mid_data_dict_one_case[target_mid_name], input_mid_vector, correction_ratio)
        return new_mid_data_dict

    corrected_model_mid_data_dict = {
        tissue_name: hypoxia_correction(specific_tissue_mid_data_dict, hypoxia_correction_parameter_dict)
        for tissue_name, specific_tissue_mid_data_dict in model_mid_data_dict.items()}

    const_parameter_dict, iter_parameter_list = all_tissue_model1(
        corrected_model_mid_data_dict, model_construction_func, output_direct, constant_flux_dict, complete_flux_dict,
        optimization_repeat_time, bounds, obj_tolerance, f1_num, f1_range, f1_display_interv, g2_num, g2_range,
        g2_display_interv, model_name, **other_parameters)

    return const_parameter_dict, iter_parameter_list


def parameter_sensitivity_model1(
        model_mid_data_dict: dict, model_construction_func, output_direct, constant_flux_dict, complete_flux_dict,
        optimization_repeat_time, bounds, obj_tolerance, sigma_dict,
        parameter_sampling_num, deviation_factor_dict,
        f1_num, f1_range, f1_display_interv, g2_num, g2_range, g2_display_interv, model_name,
        **other_parameters):
    """
    Function to generate parameters for sensitivity analysis of fluxes and MID data in model.

    :param model_mid_data_dict: MID data dict of current model.
    :param model_construction_func: Function to construct flux balance equations and MID constraints equations.
    :param output_direct: Direct to output results.
    :param constant_flux_dict: Dict of constant fluxes that are presumed previously in parameters.
    :param complete_flux_dict: Flux name to index dict to ensure all fluxes have same order.
    :param optimization_repeat_time: Number of repeat for optimization in each sampled point.
    :param bounds:Bounds for optimization of all fluxes.
    :param obj_tolerance: Tolerance of objective value to be a feasible solution.
    :param sigma_dict: Dict of standard deviation used in perturbation of MID and fluxes.
    :param parameter_sampling_num: Sampling number under each constant flux set.
    :param deviation_factor_dict: Range of deviation used in perturbation of MID and fluxes.
    :param f1_num: Sample number of free flux F1.
    :param f1_range: Sample range of free flux F1.
    :param f1_display_interv: Display interval of free flux F1. Deprecated.
    :param g2_num: Sample number of free flux G2.
    :param g2_range: Sample range of free flux G2.
    :param g2_display_interv: Display interval of free flux G2. Deprecated.
    :param model_name: Model name. Also used as name of folders and progress reminding.
    :param other_parameters: Placeholder for other parameters.
    :return: const_parameter_dict: Dict of all constant parameters which are same in all samples.
            iter_parameter_list: List of all variable parameters which are different in each sample.
    """

    def iter_parameter_generator_constructor(
            _f1_free_flux, _g2_free_flux, _model_parameter_dict_list):
        """
        Construct generator of variable parameters.
        Generator form will save memory and construct time when dealing with lots of repetitive parameters.
        :param _f1_free_flux: Free flux object of F1.
        :param _g2_free_flux: Free flux object of G2.
        :param _model_parameter_dict_list: List of other parameter dict except value of free fluxes.
        :return: Generator for complete parameter dict of all variable parameters.
        """
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
        """
        Construct parameters given MID data dict in each case after perturbation.
        :param _current_mid_data_dict: MID data dict after perturbation.
        :param _constant_flux_dict: Dict of constant fluxes after perturbation.
        :param _sample_type: Type of current sampling.
        :param _sample_index: Index of current sampling.
        :return: Model-specific parameter dict.
        """
        balance_list, mid_constraint_list = model_construction_func(_current_mid_data_dict)
        flux_balance_matrix, flux_balance_constant_vector = flux_balance_constraint_constructor(
            balance_list, complete_flux_dict)
        (
            substrate_mid_matrix, flux_sum_matrix, target_mid_vector,
            optimal_obj_value) = mid_constraint_constructor(
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
            _model_mid_data_dict, _constant_flux_dict, _mid_sigma, _flux_sigma):
        """
        Construct generator of parameter dict under different perturbed conditions.
        :param _model_mid_data_dict: Raw model MID data dict before perturbation.
        :param _constant_flux_dict: Raw constant fluxes before perturbation.
        :param _mid_sigma: Standard deviation of perturbation for MID.
        :param _flux_sigma: Standard deviation of perturbation for fluxes.
        :return: Generator for all model specific parameters.
        """
        for _sample_index in range(parameter_sampling_num):
            _current_mid_data_dict = mid_perturbation(_model_mid_data_dict, _mid_sigma)
            _model_parameter_dict = construct_model_parameter(
                _current_mid_data_dict, _constant_flux_dict, 'mid', _sample_index)
            yield _model_parameter_dict
        for _constant_flux_name, _constant_flux_value in _constant_flux_dict.items():
            for _sample_index in range(parameter_sampling_num):
                _current_constant_flux_dict = dict(_constant_flux_dict)
                _current_constant_flux_dict[_constant_flux_name] = perturb_array(
                    _constant_flux_value, _flux_sigma, 0, *deviation_factor_dict['flux'])
                _model_parameter_dict = construct_model_parameter(
                    _model_mid_data_dict, _current_constant_flux_dict, _constant_flux_name, _sample_index)
                yield _model_parameter_dict

    def perturb_array(original_array, sigma, lower_bias, min_deviation_factor, max_deviation_factor):
        """
        Function that perturb a 1D array based on Gaussian distribution and a certain range.
        :param original_array: Original array before perturbation.
        :param sigma: Standard deviation of Gaussian distribution.
        :param lower_bias: Lower bias of generated new array.
        :param min_deviation_factor: Lower bound of the range.
        :param max_deviation_factor: Upper bound of the range.
        :return: Perturbed array.
        """
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
        """
        Perturbing function to generate random perturbation for all metabolites.
        :param mid_data_dict: Raw MID data dict before perturbation.
        :param sigma: Standard deviation of Gaussian distribution.
        :return: Perturbed MID data dict.
        """
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
        model_mid_data_dict, constant_flux_dict, mid_sigma, flux_sigma)
    iter_parameter_list = iter_parameter_generator_constructor(
        f1_free_flux, g2_free_flux, model_parameter_dict_list)
    return const_parameter_dict, iter_parameter_list


def dynamic_range_model345(
        model_mid_data_dict: dict, model_construction_func, output_direct, constant_flux_dict, complete_flux_dict,
        optimization_repeat_time, bounds, obj_tolerance,
        total_point_num, free_fluxes_name_list, free_fluxes_range_list, ternary_sigma, ternary_resolution,
        model_name, parallel_num, fitted=True, **other_parameters):
    """
    Function to generate parameters to evenly sample in high-dimensional solution space and calculate
    glucose distribution. Solution space is sampled by Latin hypercube sampling.
    The calculation is for model with multiple free dimensions, such as model3, model4 and model5.

    :param model_mid_data_dict: MID data dict of current model. Each metabolite can only be mapped to one MID vector.
    :param model_construction_func: Function to construct flux balance equations and MID constraints equations.
    :param output_direct: Direct to output results.
    :param constant_flux_dict: Dict of constant fluxes that are presumed previously in parameters.
    :param complete_flux_dict: Flux name to index dict to ensure all fluxes have same order.
    :param optimization_repeat_time: Number of repeat for optimization in each sampled point.
    :param bounds: Bounds for optimization of all fluxes.
    :param obj_tolerance: Tolerance of objective value to be a feasible solution.
    :param total_point_num: Total number of points to sample
    :param free_fluxes_name_list: Name of chosen free fluxes.
    :param free_fluxes_range_list: Range of chosen free fluxes.
    :param ternary_sigma: Parameter of ternary plot. Variance of Gaussian kernel.
    :param ternary_resolution: Resolution of ternary plot, which corresponds to pixels in the triangle.
    :param model_name: Model name. Also used as name of folders and progress reminding.
    :param parallel_num: Deprecated. Parallel number of parameter generation.
    :param fitted: Whether the model is fitted. Used in unfitted negative control.
    :param other_parameters: Placeholder for other parameters.
    :return: const_parameter_dict: Dict of all constant parameters which are same in all samples.
            iter_parameter_list: List of all variable parameters which are different in each sample.
    """

    def iter_parameter_generator_constructor(
            _free_fluxes_name_list, _constant_flux_dict, _free_flux_value_list):
        """
        Construct generator of variable parameters.
        Generator form will save memory and construct time when dealing with lots of repetitive parameters.
        :param _free_fluxes_name_list: Name of chosen free fluxes.
        :param _constant_flux_dict: Dict of constant fluxes.
        :param _free_flux_value_list: Value of chosen free fluxes.
        :return: Generator for complete parameter dict of all variable parameters.
        """
        for free_flux_value in _free_flux_value_list:
            new_iter_parameter_dict = {
                'constant_flux_dict': {
                    flux_name: value for flux_name, value in zip(_free_fluxes_name_list, free_flux_value)}}
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
    flux_balance_matrix, flux_balance_constant_vector = flux_balance_constraint_constructor(
        balance_list, complete_flux_dict)
    (
        substrate_mid_matrix, flux_sum_matrix, target_mid_vector,
        optimal_obj_value) = mid_constraint_constructor(
        mid_constraint_list, complete_flux_dict)

    iter_parameter_list = iter_parameter_generator_constructor(
        free_fluxes_name_list, constant_flux_dict, free_flux_value_list)
    total_iter_num = total_point_num

    const_parameter_dict = {
        'flux_balance_matrix': flux_balance_matrix, 'flux_balance_constant_vector': flux_balance_constant_vector,
        'substrate_mid_matrix': substrate_mid_matrix, 'flux_sum_matrix': flux_sum_matrix,
        'mid_constraint_list_dict': {'': mid_constraint_list},
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
    """
    Function to generate parameters to evenly sample in high-dimensional solution space and calculate
    glucose distribution. Solution space is sampled by Latin hypercube sampling.
    The calculation is for model with multiple free dimensions, such as model3, model4 and model5.
    This function will deal with data from 8 kinds of sink tissue at the same time in one mouse.

    :param model_mid_data_dict: MID data dict of current model. Each metabolite can only be mapped to one MID vector.
    :param model_construction_func: Function to construct flux balance equations and MID constraints equations.
    :param output_direct: Direct to output results.
    :param constant_flux_dict: Dict of constant fluxes that are presumed previously in parameters.
    :param complete_flux_dict: Flux name to index dict to ensure all fluxes have same order.
    :param optimization_repeat_time: Number of repeat for optimization in each sampled point.
    :param bounds: Bounds for optimization of all fluxes.
    :param obj_tolerance: Tolerance of objective value to be a feasible solution.
    :param total_point_num: Total number of points to sample
    :param free_fluxes_name_list: Name of chosen free fluxes.
    :param free_fluxes_range_list: Range of chosen free fluxes.
    :param ternary_sigma: Parameter of ternary plot. Variance of Gaussian kernel.
    :param ternary_resolution: Resolution of ternary plot, which corresponds to pixels in the triangle.
    :param model_name: Model name. Also used as name of folders and progress reminding.
    :param parallel_num: Deprecated. Parallel number of parameter generation.
    :param other_parameters: Placeholder for other parameters.
    :return: const_parameter_dict: Dict of all constant parameters which are same in all samples.
            iter_parameter_list: List of all variable parameters which are different in each sample.
    """

    def iter_parameter_generator_constructor(
            _free_fluxes_name_list, _constant_flux_dict, _free_flux_value_list, _model_parameter_dict_list):
        """
        Construct generator of variable parameters.
        Generator form will save memory and construct time when dealing with lots of repetitive parameters.
        :param _free_fluxes_name_list: Name of chosen free fluxes.
        :param _constant_flux_dict: Dict of constant fluxes.
        :param _free_flux_value_list: Value of chosen free fluxes.
        :param _model_parameter_dict_list: List of other parameter dict except value of free fluxes.
        :return: Generator for complete parameter dict of all variable parameters.
        """
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

    total_iter_length = len(model_mid_data_dict) * total_point_num
    model_parameter_dict_list = []
    mid_constraint_list_dict = {}
    for tissue_name, specific_tissue_mid_data_dict in model_mid_data_dict.items():
        balance_list, mid_constraint_list = model_construction_func(specific_tissue_mid_data_dict)
        mid_constraint_list_dict[tissue_name] = mid_constraint_list
        flux_balance_matrix, flux_balance_constant_vector = flux_balance_constraint_constructor(
            balance_list, complete_flux_dict)
        (
            substrate_mid_matrix, flux_sum_matrix, target_mid_vector,
            optimal_obj_value) = mid_constraint_constructor(
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
        'mid_constraint_list_dict': mid_constraint_list_dict,

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


# Model construction functions are designed to construct necessary flux balance equations and MID equations.
# Those equations can be recognized and processed by following functions.

def model1_construction(mid_data_dict):
    """
    Constructor for model1.
    :param mid_data_dict: Dict of necessary MID data.
    :return: List of flux balance equations and MID constraints.
    """

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
    """
    Deprecated. Constructor for model2.
    :param mid_data_dict: Dict of necessary MID data.
    :return: List of flux balance equations and MID constraints.
    """

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
    """
    Constructor for model3.
    :param mid_data_dict: Dict of necessary MID data.
    :return: List of flux balance equations and MID constraints.
    """

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
    """
    Deprecated. Constructor for model4.
    :param mid_data_dict: Dict of necessary MID data.
    :return: List of flux balance equations and MID constraints.
    """

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
    """
    Constructor for model5.
    :param mid_data_dict: Dict of necessary MID data.
    :return: List of flux balance equations and MID constraints.
    """

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
    """
    Constructor for model6.
    :param mid_data_dict: Dict of necessary MID data.
    :return: List of flux balance equations and MID constraints.
    """

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
    """
    Constructor for model7.
    :param mid_data_dict: Dict of necessary MID data.
    :return: List of flux balance equations and MID constraints.
    """

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


# Functions to calculate contribution from metabolites. After a solution is returned from optimizer, it will be
# processed in this function to calculate contribution from metabolites. Based on length of input list of
# calculate_one_tissue_tca_contribution, an array will be returned to reflect contribution ratio of
# each metabolite respectively.

def metabolite_contribution_model12(result_dict: dict, empty=False):
    """
    Contribution ratio of glucose in model1. TCA can only be fed by glucose and lactate, and therefore dimension of
    contribution ratio array is 2. Three types of contribution ratios, including source, sink and total, are
    encapsulated as a dict and returned. If any of TCA fluxes F9 or G9 is smaller than 2, than this result
    will be regarded as infeasible.

    :param result_dict: Dict of fluxes from optimizer.
    :param empty: Whether this result is invalid in optimization.
    :return: Bool: Whether this result is feasible.
        Dict: Three kinds of contribution ratio array.
    """

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

    source_tissue_flux_array = calculate_one_tissue_tca_contribution([f56, f78])
    source_ratio_array = source_tissue_flux_array / np.sum(source_tissue_flux_array)
    sink_tissue_flux_array = calculate_one_tissue_tca_contribution([g56, g78])
    sink_ratio_array = sink_tissue_flux_array / np.sum(sink_tissue_flux_array)
    total_tissue_flux_array = source_tissue_flux_array + sink_tissue_flux_array
    total_ratio_array = total_tissue_flux_array / np.sum(total_tissue_flux_array)
    contribution_dict = {
        'source': source_ratio_array,
        'sink': sink_ratio_array,
        'total': total_ratio_array, }
    return True, contribution_dict


def metabolite_contribution_model34(result_dict: dict, empty=False):
    """
    Contribution ratio of glucose in model3 and model6. TCA can be fed by glucose, lactate and pyruvate, and therefore
    dimension of contribution ratio array is 3. Three types of contribution ratios, including source, sink and total,
    are encapsulated as a dict and returned. If any of TCA fluxes F11 or G11 is smaller than 2, than this result will
    be regarded as infeasible. Three type of contribution ratios, including source, sink and total, are encapsulated
    as a dict and returned.

    :param result_dict: Dict of fluxes from optimizer.
    :param empty: Whether this result is invalid in optimization.
    :return: Bool: Whether this result is feasible.
        Dict: Three kinds of contribution ratio array.
    """

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

    source_tissue_flux_array = calculate_one_tissue_tca_contribution([f56, f78, f910])
    source_ratio_array = source_tissue_flux_array / np.sum(source_tissue_flux_array)
    sink_tissue_flux_array = calculate_one_tissue_tca_contribution([g56, g78, g910])
    sink_ratio_array = sink_tissue_flux_array / np.sum(sink_tissue_flux_array)
    total_flux_array = source_tissue_flux_array + sink_tissue_flux_array
    total_ratio_array = total_flux_array / np.sum(total_flux_array)

    contribution_dict = {
        'source': source_ratio_array,
        'sink': sink_ratio_array,
        'total': total_ratio_array, }
    return True, contribution_dict


def metabolite_contribution_model5(result_dict: dict, empty=False):
    """
    Contribution ratio of glucose in model3 and model6. TCA can be fed by glucose, lactate and pyruvate, and therefore
    dimension of contribution ratio array is 2. Five types of contribution ratios, including source, sink1, sink2, both
    sink tissue and total, are encapsulated as a dict and returned. If both TCA fluxes in sink tissue, F9 and G9, are
    smaller than 2, than this result will be regarded as infeasible.

    :param result_dict: Dict of fluxes from optimizer.
    :param empty: Whether this result is invalid in optimization.
    :return: Bool: Whether this result is feasible.
        Dict: Five kinds of contribution ratio array.
    """

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

    source_tissue_flux_array = calculate_one_tissue_tca_contribution([f56, f78])
    source_ratio_array = source_tissue_flux_array / np.sum(source_tissue_flux_array)
    sink1_tissue_flux_array = calculate_one_tissue_tca_contribution([g56, g78])
    sink1_ratio_array = sink1_tissue_flux_array / np.sum(sink1_tissue_flux_array)
    sink2_tissue_flux_array = calculate_one_tissue_tca_contribution([h56, h78])
    sink2_ratio_array = sink2_tissue_flux_array / np.sum(sink2_tissue_flux_array)
    sink_tissue_flux_array = sink1_tissue_flux_array + sink2_tissue_flux_array
    sink_ratio_array = sink_tissue_flux_array / np.sum(sink_tissue_flux_array)
    total_tissue_flux_array = sink_tissue_flux_array + source_tissue_flux_array
    total_ratio_array = total_tissue_flux_array / np.sum(total_tissue_flux_array)

    contribution_dict = {
        'source': source_ratio_array,
        'sink1': sink1_ratio_array,
        'sink2': sink2_ratio_array,
        'sink': sink_ratio_array,
        'total': total_ratio_array, }
    return True, contribution_dict


# Final processing functions are designed to collect calculation results, analyze them, plot to figures, and save
# data to files. This function will process results in one tissue or in multiple tissue in the same procedure.

def final_processing_dynamic_range_model12(
        result_list, processed_result_list, const_parameter_dict, var_parameter_list):
    """
    Final processing function for results with model1 and model6. Results are collected and analyzed.
    Because this processing function is for 2D solution space, many kinds of results could be displayed by matrix,
    such as objective function and glucose contribution ratio. Distributions of flux and MID prediction are also
    collected for display.

    :param result_list: List of all raw calculated results.
    :param processed_result_list: List of all proccessed results which includes glucose contribution ratio.
    :param const_parameter_dict: Dict of all constant parameters which are same in all samples.
    :param var_parameter_list: List of all variable parameters which are different in each sample.
    :return: None
    """

    # Construct some parameters for result processing.

    f1_free_flux: config.FreeVariable = const_parameter_dict['f1_free_flux']
    g2_free_flux: config.FreeVariable = const_parameter_dict['g2_free_flux']
    output_direct = const_parameter_dict['output_direct']
    obj_tolerance = const_parameter_dict['obj_tolerance']
    raw_constant_flux_dict = const_parameter_dict['raw_constant_flux_dict']
    mid_constraint_list_dict = const_parameter_dict['mid_constraint_list_dict']
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
    predicted_mid_collection_dict = {tissue_name: {} for tissue_name in tissue_name_list}

    target_vector_dict, mid_size_dict = mid_prediction_preparation(tissue_name_list, mid_constraint_list_dict)

    # Iterate to process and collect results.

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
            append_flux_distribution(result_dict, feasible_flux_distribution_dict)
            one_case_mid_prediction(
                result_dict, mid_constraint_list_dict[tissue_name], mid_size_dict,
                predicted_mid_collection_dict[tissue_name])
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

    # Save and plotting results. Commented codes are unnecessary figures but could be restored.

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
            'predicted_mid_collection_dict': predicted_mid_collection_dict[tissue_name],
            'target_vector_dict': target_vector_dict[tissue_name]
        }

        # plot_heat_map(
        #     valid_matrix, g2_free_flux, f1_free_flux,
        #     save_path="{}/dynamic_range_{}.png".format(output_direct, tissue_name))
        # plot_heat_map(
        #     objective_function_matrix, g2_free_flux, f1_free_flux, cmap=color_set.blue_orange_cmap,
        #     cbar_name='Objective difference',
        #     title=tissue_name,
        #     save_path="{}/objective_function_{}.png".format(output_direct, tissue_name))
        # plot_heat_map(
        #     filtered_obj_function_matrix, g2_free_flux, f1_free_flux, cmap=color_set.blue_orange_cmap,
        #     cbar_name='Filtered objective difference',
        #     title=tissue_name,
        #     save_path="{}/filtered_objective_function_{}.png".format(output_direct, tissue_name))
        plot_box_distribution(
            {'normal': filtered_obj_array},
            title=tissue_name,
            save_path="{}/filtered_objective_distribution_{}.png".format(output_direct, tissue_name))
        plot_violin_distribution(
            well_fit_glucose_contri_dict,
            color_set.blue,
            title=tissue_name,
            save_path="{}/glucose_contribution_violin_{}.png".format(output_direct, tissue_name))
        plot_box_distribution(
            feasible_flux_distribution_dict, title=tissue_name,
            save_path="{}/flux_distribution_boxplot_{}.png".format(output_direct, tissue_name))

        # for contribution_type, glucose_contri_matrix in glucose_contri_matrix_dict.items():
        #     plot_heat_map(
        #         glucose_contri_matrix, g2_free_flux, f1_free_flux, cmap=color_set.blue_orange_cmap,
        #         cbar_name='Glucose Contribution',
        #         title=tissue_name,
        #         save_path="{}/glucose_contribution_heatmap_{}.png".format(output_direct, contribution_type))

    if all_tissue:
        plot_violin_distribution(
            sink_tissue_contribution_dict,
            color_set.blue,
            title='all_tissue',
            save_path="{}/all_tissue_glucose_contribution_violin.png".format(output_direct))

    with gzip.open("{}/raw_output_data_dict.gz".format(output_direct), 'wb') as f_out:
        pickle.dump(raw_output_data_dict, f_out)
    with gzip.open("{}/output_data_dict.gz".format(output_direct), 'wb') as f_out:
        pickle.dump(output_data_dict, f_out)


def final_processing_dynamic_range_model345(
        result_list, processed_result_list, const_parameter_dict, var_parameter_list):
    """
    Final processing function for results with model3, model5 and model7. Results are collected and analyzed.
    Because this processing function is for high-dimensional solution space, all kinds of results can only be
    displayed by 1D distribution. Distributions of flux and MID prediction are also collected for display.

    :param result_list: List of all raw calculated results.
    :param processed_result_list: List of all proccessed results which includes glucose contribution ratio.
    :param const_parameter_dict: Dict of all constant parameters which are same in all samples.
    :param var_parameter_list: List of all variable parameters which are different in each sample.
    :return: None
    """

    # Construct some parameters for result processing.

    output_direct = const_parameter_dict['output_direct']
    free_fluxes_name_list = const_parameter_dict['free_fluxes_name_list']
    ternary_sigma = const_parameter_dict['ternary_sigma']
    ternary_resolution = const_parameter_dict['ternary_resolution']
    obj_tolerance = const_parameter_dict['obj_tolerance']
    model_name = const_parameter_dict['model_name']
    parallel_num = const_parameter_dict['parallel_num']
    raw_constant_flux_dict = const_parameter_dict['raw_constant_flux_dict']
    mid_constraint_list_dict = const_parameter_dict['mid_constraint_list_dict']
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
    predicted_mid_collection_dict = {tissue_name: {} for tissue_name in tissue_name_list}

    target_vector_dict, mid_size_dict = mid_prediction_preparation(tissue_name_list, mid_constraint_list_dict)

    # Iterate to process and collect results.

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
                append_flux_distribution(result_dict, feasible_flux_distribution_dict)
                one_case_mid_prediction(
                    result_dict, mid_constraint_list_dict[tissue_name], mid_size_dict,
                    predicted_mid_collection_dict[tissue_name])
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

    # Save and plotting results. Commented codes are unnecessary figures but could be restored.

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
            'predicted_mid_collection_dict': predicted_mid_collection_dict[tissue_name],
            'target_vector_dict': target_vector_dict[tissue_name]
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
            project_list.append((plot_violin_distribution, kwargs))
        else:
            for contribution_type, contribution_matrix in well_fit_contri_list_dict.items():
                kwargs = {
                    'tri_data_matrix': contribution_matrix, 'sigma': ternary_sigma, 'bin_num': ternary_resolution,
                    'title': "{}_{}".format(contribution_type, tissue_name),
                    'mean': ternary_mean,
                    'save_path': "{}/test_glucose_contribution_heatmap_{}_{}.png".format(
                        output_direct, contribution_type, tissue_name)}
                project_list.append((plot_ternary_density, kwargs))
                # plot_ternary_density(
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
        project_list.append((plot_violin_distribution, kwargs))
        kwargs = {
            'data_dict': {'normal': filtered_obj_array},
            'title': tissue_name,
            'save_path': "{}/filtered_objective_distribution_{}.png".format(output_direct, tissue_name)}
        project_list.append((plot_box_distribution, kwargs))
        kwargs = {
            'data_dict': feasible_flux_distribution_dict,
            'title': tissue_name,
            'save_path': "{}/flux_distribution_boxplot_{}.png".format(output_direct, tissue_name)}
        project_list.append((plot_box_distribution, kwargs))

    with mp.Pool(parallel_num) as pool:
        pool.map(func_parallel_wrap, project_list)

    with gzip.open("{}/raw_output_data_dict.gz".format(output_direct), 'wb') as f_out:
        pickle.dump(raw_output_data_dict, f_out)
    with gzip.open("{}/output_data_dict.gz".format(output_direct), 'wb') as f_out:
        pickle.dump(output_data_dict, f_out)


def final_processing_parameter_sensitivity_model1(
        result_list, processed_result_list, const_parameter_dict, var_parameter_list):
    """
    Final processing function for parameter sensitivity. MID and fluxes are perturbed and sampled, and their medians
    are collected together to reflect sensitivity of this parameter. Distribution of medians are displayed by violin
    plots.

    :param result_list: List of all raw calculated results.
    :param processed_result_list: List of all proccessed results which includes glucose contribution ratio.
    :param const_parameter_dict: Dict of all constant parameters which are same in all samples.
    :param var_parameter_list: List of all variable parameters which are different in each sample.
    :return: None
    """

    # Construct some parameters for result processing.

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

    # Iterate to process and collect results.

    for solver_result, processed_dict in zip(result_list, processed_result_list):
        sample_type = solver_result.label['sample_type']
        sample_index = solver_result.label['sample_index']
        obj_diff = processed_dict['obj_diff']
        contribution_dict = processed_dict['contribution_dict']
        valid = processed_dict['valid']
        glucose_contri_list_dict = well_fit_glucose_contri_dict[sample_type]
        objective_function_list = objective_function_list_dict[sample_type]

        if valid:
            while sample_index >= len(objective_function_list):
                objective_function_list.append([])
            objective_function_list[sample_index].append(obj_diff)
        for contribution_type, contribution_vector in contribution_dict.items():
            if contribution_type not in glucose_contri_list_dict:
                glucose_contri_list_dict[contribution_type] = []
            current_glucose_contribution_dict = glucose_contri_list_dict[contribution_type]
            if valid:
                if obj_diff < obj_tolerance:
                    while sample_index >= len(current_glucose_contribution_dict):
                        current_glucose_contribution_dict.append([])
                    current_glucose_contribution_dict[sample_index].append(contribution_vector[0])

    for sample_type, sample_obj_list in objective_function_list_dict.items():
        for sample_index, obj_list in enumerate(sample_obj_list):
            objective_function_median_dict[sample_type].append(np.median(obj_list))

    # Save and plotting results.

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
        plot_violin_distribution(
            {sample_type: objective_function_median_dict[sample_type]},
            color_set.purple, cutoff=obj_tolerance,
            save_path="{}/objective_function_violin_parameter_sensitivity_{}.png".format(
                output_direct, sample_type))
        current_sample_type_median_dict = well_fit_median_contri_dict[sample_type]
        plot_violin_distribution(
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
