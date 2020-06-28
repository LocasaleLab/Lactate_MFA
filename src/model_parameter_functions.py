#!/usr/bin/env python
#  -*- coding: utf-8 -*-
# (C) Shiyu Liu, Locasale Lab, 2019
# Contact: liushiyu1994@gmail.com
# All rights reserved
# Licensed under MIT License (see LICENSE-MIT)

"""
    Functions that return parameter dicts for analysis process.
"""

from src import model_specific_functions, config

constant_set = config.Constants()
color_set = config.Color()

global_output_direct = constant_set.output_direct

# Import common functions from model_specific_functions:

# Data loader functions
data_loader_rabinowitz = model_specific_functions.data_loader_rabinowitz
data_loader_dan = model_specific_functions.data_loader_dan
mid_data_collection_model1234 = model_specific_functions.mid_data_collection_model1234
mid_data_collection_model5 = model_specific_functions.mid_data_collection_model5
mid_data_collection_all_tissue = model_specific_functions.mid_data_collection_all_tissue

# Model construction functions
model1_construction = model_specific_functions.model1_construction
model2_construction = model_specific_functions.model2_construction
model3_construction = model_specific_functions.model3_construction
model4_construction = model_specific_functions.model4_construction
model5_construction = model_specific_functions.model5_construction
model6_construction = model_specific_functions.model6_construction
model7_construction = model_specific_functions.model7_construction

# Hook functions each iteration
metabolite_contribution_model12 = model_specific_functions.metabolite_contribution_model12
metabolite_contribution_model34 = model_specific_functions.metabolite_contribution_model34
metabolite_contribution_model5 = model_specific_functions.metabolite_contribution_model5

# Hook functions final
final_processing_dynamic_range_model12 = model_specific_functions.final_processing_dynamic_range_model12
final_processing_dynamic_range_model345 = model_specific_functions.final_processing_dynamic_range_model345
final_processing_parameter_sensitivity_model1 = model_specific_functions.final_processing_parameter_sensitivity_model1

# Preparation functions
dynamic_range_model12 = model_specific_functions.dynamic_range_model12
dynamic_range_model345 = model_specific_functions.dynamic_range_model345
all_tissue_model1 = model_specific_functions.all_tissue_model1
all_tissue_model3 = model_specific_functions.all_tissue_model3
all_tissue_hypoxia_correction = model_specific_functions.all_tissue_hypoxia_correction
parameter_sensitivity_model1 = model_specific_functions.parameter_sensitivity_model1


def model1_parameters(test=False):
    """
    Parameter dict for model1. It contains one source tissue liver, one sink tissue heart
    and two circulatory metabolites glucose and lactate, and is fitted with glucose-infused M1 in low-infusion data.

    :param test: If the parameter is for test mode. Sample size in test mode is much smaller, and therefore has
        shorter running time.
    :return: Parameter dict for model1.
    """

    model_name = "model1"
    output_direct = "{}/{}".format(global_output_direct, model_name)

    data_collection_kwargs = {
        'label_list': ["glucose"], 'mouse_id_list': ['M1'],
        'source_tissue_marker': constant_set.liver_marker, 'sink_tissue_marker': constant_set.heart_marker}
    data_loader_func = data_loader_rabinowitz
    data_collection_func = mid_data_collection_model1234

    hook_in_each_iteration = metabolite_contribution_model12
    hook_after_all_iterations = final_processing_dynamic_range_model12
    model_construction_func = model1_construction
    parameter_construction_func = dynamic_range_model12

    complete_flux_list = ['F{}'.format(i + 1) for i in range(10)] + ['G{}'.format(i + 1) for i in range(9)] + \
                         ['Fcirc_glc', 'Fcirc_lac']
    complete_flux_dict = {var: i for i, var in enumerate(complete_flux_list)}
    # constant_flux_dict = {'Fcirc_glc': 150.9, 'Fcirc_lac': 374.4, 'F10': 100}
    constant_flux_dict = {'Fcirc_glc': 150.9, 'Fcirc_lac': 374.4, 'F10': 35}

    min_flux_value = 1
    max_flux_value = 500  # 600
    # max_flux_value = 1000
    optimization_repeat_time = 10
    obj_tolerance = 0.1
    f1_range = [1, 150]
    g2_range = [1, 150]
    if test:
        f1_num = 31
        f1_display_interv = 30
        g2_num = 31
        g2_display_interv = 30
    else:
        f1_num = 1000
        f1_display_interv = 250
        g2_num = 1000
        g2_display_interv = 250
    bounds = config.bound_pair_generator(min_flux_value, max_flux_value, complete_flux_list)
    return locals()


def model1_m5_parameters(test=False):
    """
    Parameter dict for model1_m5. It contains one source tissue liver, one sink tissue heart
    and two circulatory metabolites glucose and lactate, and is fitted with glucose-infused M5 in low-infusion data.
    Most parameters are inherited from model1_parameters.

    :param test: If the parameter is for test mode. Sample size in test mode is much smaller, and therefore has
        shorter running time.
    :return: Parameter dict for model1_m5.
    """

    model_name = "model1_m5"
    output_direct = "{}/{}".format(global_output_direct, model_name)

    data_collection_kwargs = {
        'label_list': ["glucose"], 'mouse_id_list': ['M5'],
        'source_tissue_marker': constant_set.liver_marker, 'sink_tissue_marker': constant_set.heart_marker}
    model1_parameter_dict = model1_parameters(test)
    model1_parameter_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'data_collection_kwargs': data_collection_kwargs
    })
    return model1_parameter_dict


def model1_m9_parameters(test=False):
    """
    Parameter dict for model1_m9. It contains one source tissue liver, one sink tissue heart
    and two circulatory metabolites glucose and lactate, and is fitted with glucose-infused M9 in low-infusion data.
    Most parameters are inherited from model1_parameters.

    :param test: If the parameter is for test mode. Sample size in test mode is much smaller, and therefore has
        shorter running time.
    :return: Parameter dict for model1_m9.
    """

    model_name = "model1_m9"
    output_direct = "{}/{}".format(global_output_direct, model_name)

    data_collection_kwargs = {
        'label_list': ["glucose"], 'mouse_id_list': ['M9'],
        'source_tissue_marker': constant_set.liver_marker, 'sink_tissue_marker': constant_set.heart_marker}
    model1_parameter_dict = model1_parameters(test)
    model1_parameter_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'data_collection_kwargs': data_collection_kwargs
    })
    return model1_parameter_dict


def model1_lactate_parameters(test=False):
    """
    Parameter dict for model1_lactate. It contains one source tissue liver,
    one sink tissue heart and two circulatory metabolites glucose and lactate,
    and is fitted with lactate-infused M1 in low-infusion data.
    Most parameters are inherited from model1_parameters.

    :param test: If the parameter is for test mode. Sample size in test mode is much smaller, and therefore has
        shorter running time.
    :return: Parameter dict for model1_lactate.
    """

    model_name = "model1_lactate"
    output_direct = "{}/{}".format(global_output_direct, model_name)

    data_collection_kwargs = {
        'label_list': ["lactate"], 'mouse_id_list': ['M3'],
        'source_tissue_marker': constant_set.liver_marker, 'sink_tissue_marker': constant_set.heart_marker}
    model1_parameter_dict = model1_parameters(test)
    model1_parameter_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'data_collection_kwargs': data_collection_kwargs
    })
    return model1_parameter_dict


def model1_lactate_m4_parameters(test=False):
    """
    Parameter dict for model1_lactate_m4. It contains one source tissue liver,
    one sink tissue heart and two circulatory metabolites glucose and lactate,
    and is fitted with lactate-infused M4 in low-infusion data.
    Most parameters are inherited from model1_parameters.

    :param test: If the parameter is for test mode. Sample size in test mode is much smaller, and therefore has
        shorter running time.
    :return: Parameter dict for model1_lactate_m4.
    """

    model_name = "model1_lactate_m4"
    output_direct = "{}/{}".format(global_output_direct, model_name)

    data_collection_kwargs = {
        'label_list': ["lactate"], 'mouse_id_list': ['M4'],
        'source_tissue_marker': constant_set.liver_marker, 'sink_tissue_marker': constant_set.heart_marker}
    model1_parameter_dict = model1_parameters(test)
    model1_parameter_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'data_collection_kwargs': data_collection_kwargs
    })
    return model1_parameter_dict


def model1_lactate_m10_parameters(test=False):
    """
    Parameter dict for model1_lactate_m10. It contains one source tissue liver,
    one sink tissue heart and two circulatory metabolites glucose and lactate,
    and is fitted with lactate-infused M10 in low-infusion data.
    Most parameters are inherited from model1_parameters.

    :param test: If the parameter is for test mode. Sample size in test mode is much smaller, and therefore has
        shorter running time.
    :return: Parameter dict for model1_lactate_m10.
    """

    model_name = "model1_lactate_m10"
    output_direct = "{}/{}".format(global_output_direct, model_name)

    data_collection_kwargs = {
        'label_list': ["lactate"], 'mouse_id_list': ['M10'],
        'source_tissue_marker': constant_set.liver_marker, 'sink_tissue_marker': constant_set.heart_marker}
    model1_parameter_dict = model1_parameters(test)
    model1_parameter_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'data_collection_kwargs': data_collection_kwargs
    })
    return model1_parameter_dict


def model1_lactate_m11_parameters(test=False):
    """
    Parameter dict for model1_lactate_m11. It contains one source tissue liver,
    one sink tissue heart and two circulatory metabolites glucose and lactate,
    and is fitted with lactate-infused M10 in low-infusion data.
    Most parameters are inherited from model1_parameters.

    :param test: If the parameter is for test mode. Sample size in test mode is much smaller, and therefore has
        shorter running time.
    :return: Parameter dict for model1_lactate_m11.
    """

    model_name = "model1_lactate_m11"
    output_direct = "{}/{}".format(global_output_direct, model_name)

    data_collection_kwargs = {
        'label_list': ["lactate"], 'mouse_id_list': ['M11'],
        'source_tissue_marker': constant_set.liver_marker, 'sink_tissue_marker': constant_set.heart_marker}
    model1_parameter_dict = model1_parameters(test)
    model1_parameter_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'data_collection_kwargs': data_collection_kwargs
    })
    return model1_parameter_dict


def model1_all_tissue(test=False):
    """
    Parameter dict for model1_all_tissue. It contains one source tissue liver, one sink tissue from 8 kinds of tissue
    and two circulatory metabolites glucose and lactate, and is fitted with glucose-infused M1 in low-infusion data.
    Most parameters are inherited from model1_parameters.

    :param test: If the parameter is for test mode. Sample size in test mode is much smaller, and therefore has
        shorter running time.
    :return: Parameter dict for model1_all_tissue.
    """

    model_name = "model1_all_tissue"
    output_direct = "{}/{}".format(global_output_direct, model_name)

    model1_parameter_dict = model1_parameters(test)
    data_collection_kwargs = {
        'label_list': ["glucose"], 'mouse_id_list': ['M1'],
        'source_tissue_marker': constant_set.liver_marker,
        'sink_tissue_marker_list': [
            constant_set.heart_marker, constant_set.brain_marker, constant_set.muscle_marker,
            constant_set.kidney_marker, constant_set.lung_marker, constant_set.pancreas_marker,
            constant_set.intestine_marker, constant_set.spleen_marker]}
    data_collection_func = mid_data_collection_all_tissue
    parameter_construction_func = all_tissue_model1

    model1_parameter_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'data_collection_kwargs': data_collection_kwargs,
        'data_collection_func': data_collection_func,
        'parameter_construction_func': parameter_construction_func,
    })

    return model1_parameter_dict


def model1_all_tissue_m5(test=False):
    """
    Parameter dict for model1_all_tissue_m5. It contains one source tissue liver, one sink tissue from 8 kinds of tissue
    and two circulatory metabolites glucose and lactate, and is fitted with glucose-infused M5 in low-infusion data.
    Most parameters are inherited from model1_all_tissue.

    :param test: If the parameter is for test mode. Sample size in test mode is much smaller, and therefore has
        shorter running time.
    :return: Parameter dict for model1_all_tissue_m5.
    """

    model_name = "model1_all_tissue_m5"
    output_direct = "{}/{}".format(global_output_direct, model_name)

    model1_all_tissue_dict = model1_all_tissue(test)
    data_collection_kwargs = model1_all_tissue_dict['data_collection_kwargs']
    data_collection_kwargs.update({
        'mouse_id_list': ['M5']})
    model1_all_tissue_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'data_collection_kwargs': data_collection_kwargs
    })
    return model1_all_tissue_dict


def model1_all_tissue_m9(test=False):
    """
    Parameter dict for model1_all_tissue_m9. It contains one source tissue liver, one sink tissue from 8 kinds of tissue
    and two circulatory metabolites glucose and lactate, and is fitted with glucose-infused M9 in low-infusion data.
    Most parameters are inherited from model1_all_tissue.

    :param test: If the parameter is for test mode. Sample size in test mode is much smaller, and therefore has
        shorter running time.
    :return: Parameter dict for model1_all_tissue_m9.
    """

    model_name = "model1_all_tissue_m9"
    output_direct = "{}/{}".format(global_output_direct, model_name)

    model1_all_tissue_dict = model1_all_tissue(test)
    data_collection_kwargs = model1_all_tissue_dict['data_collection_kwargs']
    data_collection_kwargs.update({
        'mouse_id_list': ['M9']})
    model1_all_tissue_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'data_collection_kwargs': data_collection_kwargs
    })
    return model1_all_tissue_dict


def model1_all_tissue_lactate(test=False):
    """
    Parameter dict for model1_all_tissue_lactate. It contains one source tissue liver,
    one sink tissue from 8 kinds of tissue and two circulatory metabolites glucose and lactate,
    and is fitted with lactate-infused M1 in low-infusion data.
    Most parameters are inherited from model1_all_tissue.

    :param test: If the parameter is for test mode. Sample size in test mode is much smaller, and therefore has
        shorter running time.
    :return: Parameter dict for model1_all_tissue_lactate.
    """

    model_name = "model1_all_tissue_lactate"
    output_direct = "{}/{}".format(global_output_direct, model_name)

    model1_all_tissue_dict = model1_all_tissue(test)
    data_collection_kwargs = model1_all_tissue_dict['data_collection_kwargs']
    data_collection_kwargs.update({
        'label_list': ["lactate"], 'mouse_id_list': ['M3']})
    model1_all_tissue_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'data_collection_kwargs': data_collection_kwargs
    })
    return model1_all_tissue_dict


def model1_all_tissue_lactate_m4(test=False):
    """
    Parameter dict for model1_all_tissue_lactate_m4. It contains one source tissue liver,
    one sink tissue from 8 kinds of tissue and two circulatory metabolites glucose and lactate,
    and is fitted with lactate-infused M4 in low-infusion data.
    Most parameters are inherited from model1_all_tissue.

    :param test: If the parameter is for test mode. Sample size in test mode is much smaller, and therefore has
        shorter running time.
    :return: Parameter dict for model1_all_tissue_m4.
    """

    model_name = "model1_all_tissue_lactate_m4"
    output_direct = "{}/{}".format(global_output_direct, model_name)

    model1_all_tissue_dict = model1_all_tissue(test)
    data_collection_kwargs = model1_all_tissue_dict['data_collection_kwargs']
    data_collection_kwargs.update({
        'label_list': ["lactate"], 'mouse_id_list': ['M4']})
    model1_all_tissue_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'data_collection_kwargs': data_collection_kwargs
    })
    return model1_all_tissue_dict


def model1_all_tissue_lactate_m10(test=False):
    """
    Parameter dict for model1_all_tissue_lactate_m10. It contains one source tissue liver,
    one sink tissue from 8 kinds of tissue and two circulatory metabolites glucose and lactate,
    and is fitted with lactate-infused M10 in low-infusion data.
    Most parameters are inherited from model1_all_tissue.

    :param test: If the parameter is for test mode. Sample size in test mode is much smaller, and therefore has
        shorter running time.
    :return: Parameter dict for model1_all_tissue_m10.
    """

    model_name = "model1_all_tissue_lactate_m10"
    output_direct = "{}/{}".format(global_output_direct, model_name)

    model1_all_tissue_dict = model1_all_tissue(test)
    data_collection_kwargs = model1_all_tissue_dict['data_collection_kwargs']
    data_collection_kwargs.update({
        'label_list': ["lactate"], 'mouse_id_list': ['M10']})
    model1_all_tissue_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'data_collection_kwargs': data_collection_kwargs
    })
    return model1_all_tissue_dict


def model1_all_tissue_lactate_m11(test=False):
    """
    Parameter dict for model1_all_tissue_lactate_m11. It contains one source tissue liver,
    one sink tissue from 8 kinds of tissue and two circulatory metabolites glucose and lactate,
    and is fitted with lactate-infused M11 in low-infusion data.
    Most parameters are inherited from model1_all_tissue.

    :param test: If the parameter is for test mode. Sample size in test mode is much smaller, and therefore has
        shorter running time.
    :return: Parameter dict for model1_all_tissue_m11.
    """

    model_name = "model1_all_tissue_lactate_m11"
    output_direct = "{}/{}".format(global_output_direct, model_name)

    model1_all_tissue_dict = model1_all_tissue(test)
    data_collection_kwargs = model1_all_tissue_dict['data_collection_kwargs']
    data_collection_kwargs.update({
        'label_list': ["lactate"], 'mouse_id_list': ['M11']})
    model1_all_tissue_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'data_collection_kwargs': data_collection_kwargs
    })
    return model1_all_tissue_dict


def model1_parameter_sensitivity(test=False):
    """
    Parameter dict for model1_parameter_sensitivity. The model is same as model1, but fluxes and MID data are
    randomly perturbed to mimic random error and analyzed.
    The model is fitted with glucose-infused M1 in low-infusion data.
    Most parameters are inherited from model1_parameters.

    :param test: If the parameter is for test mode. Sample size in test mode is much smaller, and therefore has
        shorter running time.
    :return: Parameter dict for model1_parameter_sensitivity.
    """

    model_name = "model1_parameter_sensitivity"
    output_direct = "{}/{}".format(global_output_direct, model_name)

    model1_parameter_dict = model1_parameters(test)
    parameter_construction_func = parameter_sensitivity_model1
    hook_after_all_iterations = final_processing_parameter_sensitivity_model1

    deviation_factor_dict = {'mid': [0.1, 0.9], 'flux': [0.05, 0.6]}
    sigma_dict = {'mid': 0.5, 'flux': 0.2}  # 0.1

    if test:
        f1_num = 21
        f1_display_interv = 20
        g2_num = 21
        g2_display_interv = 20
        parameter_sampling_num = 5
    else:
        f1_num = 100
        f1_display_interv = 20
        g2_num = 100
        g2_display_interv = 20
        parameter_sampling_num = 100

    model1_parameter_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'parameter_construction_func': parameter_construction_func,
        'deviation_factor_dict': deviation_factor_dict,
        'sigma_dict': sigma_dict,
        'f1_num': f1_num, 'g2_num': g2_num,
        'f1_display_interv': f1_display_interv,
        'g2_display_interv': g2_display_interv,
        'parameter_sampling_num': parameter_sampling_num,
        'hook_after_all_iterations': hook_after_all_iterations
    })

    return model1_parameter_dict


def model1_hypoxia_correction(test=False):
    """
    Parameter dict for model1_all_tissue_hypoxia. The model is same as model1_all_tissue, but MID data are perturbed
    to mimic hypoxia condition and analyzed. The model is fitted with glucose-infused M1 in low-infusion data.
    Most parameters are inherited from model1_all_tissue.

    :param test: If the parameter is for test mode. Sample size in test mode is much smaller, and therefore has
        shorter running time.
    :return: Parameter dict for model1_all_tissue_hypoxia.
    """

    model_name = "model1_all_tissue_hypoxia"
    output_direct = "{}/{}".format(global_output_direct, model_name)

    # Ratio means mixture ratio of newly generated metabolites under hypoxia condition.
    # For example, "glc_source" is 0.2 means that in the mixture of final MID, 20% of glucose in source tissue
    # is from glycogenolysis, and therefore is unlabeled.
    hypoxia_correction_parameter_dict = {
        'glc_source': 0.2,
        'pyr_sink': 0,
        'lac_sink': 0.2,
    }
    model1_all_tissue_parameter_dict = model1_all_tissue(test)
    model1_all_tissue_parameter_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'parameter_construction_func': all_tissue_hypoxia_correction,
        'hypoxia_correction_parameter_dict': hypoxia_correction_parameter_dict
    })
    return model1_all_tissue_parameter_dict


def model1_hypoxia_correction_m5(test=False):
    """
    Parameter dict for model1_all_tissue_hypoxia_m5. The model is same as model1_all_tissue, but MID data are perturbed
    to mimic hypoxia condition and analyzed. The model is fitted with glucose-infused M5 in low-infusion data.
    Most parameters are inherited from model1_hypoxia_correction.

    :param test: If the parameter is for test mode. Sample size in test mode is much smaller, and therefore has
        shorter running time.
    :return: Parameter dict for model1_all_tissue_hypoxia_m5.
    """

    model_name = "model1_all_tissue_lactate_m11"
    output_direct = "{}/{}".format(global_output_direct, model_name)

    model1_all_tissue_dict = model1_all_tissue(test)
    data_collection_kwargs = model1_all_tissue_dict['data_collection_kwargs']
    data_collection_kwargs.update({
        'label_list': ["lactate"], 'mouse_id_list': ['M11']})
    model1_all_tissue_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'data_collection_kwargs': data_collection_kwargs
    })
    return model1_all_tissue_dict


def model1_unfitted_parameters(test=False):
    """
    Parameter dict for model1_unfitted. The model is same as model1, but fluxes are randomly generated and not
    optimized for data. The objective value is calculated with glucose-infused M1 in low-infusion data.
    Most parameters are inherited from model1_parameters.

    :param test: If the parameter is for test mode. Sample size in test mode is much smaller, and therefore has
        shorter running time.
    :return: Parameter dict for model1_unfitted.
    """

    model_name = "model1_unfitted"
    output_direct = "{}/{}".format(global_output_direct, model_name)

    model1_parameter_dict = model1_parameters(test)
    model1_parameter_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'fitted': False,
        'obj_tolerance': 999999
    })

    return model1_parameter_dict


def model3_parameters(test=False):
    """
    Parameter dict for model1. It contains one source tissue liver, one sink tissue heart
    and three circulatory metabolites glucose, lactate and pyruvate, and is fitted with
    glucose-infused M1 in low-infusion data.

    :param test: If the parameter is for test mode. Sample size in test mode is much smaller, and therefore has
        shorter running time.
    :return: Parameter dict for model3.
    """

    model_name = "model3"
    output_direct = "{}/{}".format(global_output_direct, model_name)

    data_collection_kwargs = {
        'label_list': ["glucose"], 'mouse_id_list': ['M1'],
        'source_tissue_marker': constant_set.liver_marker, 'sink_tissue_marker': constant_set.heart_marker}
    data_loader_func = data_loader_rabinowitz
    data_collection_func = mid_data_collection_model1234

    hook_in_each_iteration = metabolite_contribution_model34
    hook_after_all_iterations = final_processing_dynamic_range_model345
    model_construction_func = model3_construction
    parameter_construction_func = dynamic_range_model345

    complete_flux_list = ['F{}'.format(i + 1) for i in range(12)] + ['G{}'.format(i + 1) for i in range(11)] + \
                         ['J{}'.format(i + 1) for i in range(3)] + ['Fcirc_glc', 'Fcirc_lac', 'Fcirc_pyr']
    complete_flux_dict = {var: i for i, var in enumerate(complete_flux_list)}
    constant_flux_dict = {'Fcirc_glc': 150.9, 'Fcirc_lac': 374.4, 'Fcirc_pyr': 57.3, 'F12': 60}
    # constant_flux_dict = {'Fcirc_glc': 150.9, 'Fcirc_lac': 374.4, 'Fcirc_pyr': 57.3, 'F12': 200}

    min_flux_value = 1
    # max_flux_value = 1000
    max_flux_value = 800
    optimization_repeat_time = 10
    obj_tolerance = 0.1
    ternary_sigma = 0.15

    free_fluxes_name_list = ['F1', 'G2', 'F9', 'G10', 'F3']
    free_fluxes_range_list = [
        [min_flux_value, constant_flux_dict['Fcirc_glc']],
        [min_flux_value, constant_flux_dict['Fcirc_glc']],
        [min_flux_value, constant_flux_dict['Fcirc_pyr']],
        [min_flux_value, constant_flux_dict['Fcirc_pyr']],
        [min_flux_value, constant_flux_dict['Fcirc_lac']],
    ]

    if test:
        total_point_num = int(1e3)
        ternary_resolution = int(2 ** 7)
    else:
        total_point_num = int(1e6)
        ternary_resolution = int(2 ** 8)
    bounds = config.bound_pair_generator(min_flux_value, max_flux_value, complete_flux_list)
    return locals()


def model3_all_tissue(test=False):
    """
    Parameter dict for model3_all_tissue. It contains one source tissue liver, one sink tissue from 8 kinds of tissue
    and three circulatory metabolites glucose and lactate, and is fitted with glucose-infused M1 in low-infusion data.
    Most parameters are inherited from model3_parameters.

    :param test: If the parameter is for test mode. Sample size in test mode is much smaller, and therefore has
        shorter running time.
    :return: Parameter dict for model3_all_tissue.
    """

    model_name = "model3_all_tissue"
    output_direct = "{}/{}".format(global_output_direct, model_name)

    model3_parameter_dict = model3_parameters(test)

    data_collection_kwargs = {
        'label_list': ["glucose"], 'mouse_id_list': ['M1'],
        'source_tissue_marker': constant_set.liver_marker,
        'sink_tissue_marker_list': [
            constant_set.heart_marker, constant_set.brain_marker, constant_set.muscle_marker,
            constant_set.kidney_marker, constant_set.lung_marker, constant_set.pancreas_marker,
            constant_set.intestine_marker, constant_set.spleen_marker]}
    data_collection_func = mid_data_collection_all_tissue
    parameter_construction_func = all_tissue_model3

    model3_parameter_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'data_collection_kwargs': data_collection_kwargs,
        'data_collection_func': data_collection_func,
        'parameter_construction_func': parameter_construction_func,
    })
    return model3_parameter_dict


def model3_all_tissue_m5(test=False):
    """
    Parameter dict for model3_all_tissue_m5. It contains one source tissue liver, one sink tissue from 8 kinds of tissue
    and three circulatory metabolites glucose and lactate, and is fitted with glucose-infused M5 in low-infusion data.
    Most parameters are inherited from model3_all_tissue.

    :param test: If the parameter is for test mode. Sample size in test mode is much smaller, and therefore has
        shorter running time.
    :return: Parameter dict for model3_all_tissue_m5.
    """

    model_name = "model3_all_tissue_m5"
    output_direct = "{}/{}".format(global_output_direct, model_name)

    model3_all_tissue_dict = model3_all_tissue(test)
    data_collection_kwargs = model3_all_tissue_dict['data_collection_kwargs']
    data_collection_kwargs.update({
        'mouse_id_list': ['M5']})
    model3_all_tissue_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'data_collection_kwargs': data_collection_kwargs
    })
    return model3_all_tissue_dict


def model3_all_tissue_m9(test=False):
    """
    Parameter dict for model3_all_tissue_m9. It contains one source tissue liver, one sink tissue from 8 kinds of tissue
    and three circulatory metabolites glucose and lactate, and is fitted with glucose-infused M9 in low-infusion data.
    Most parameters are inherited from model3_all_tissue.

    :param test: If the parameter is for test mode. Sample size in test mode is much smaller, and therefore has
        shorter running time.
    :return: Parameter dict for model3_all_tissue_m9.
    """

    model_name = "model3_all_tissue_m9"
    output_direct = "{}/{}".format(global_output_direct, model_name)

    model3_all_tissue_dict = model3_all_tissue(test)
    data_collection_kwargs = model3_all_tissue_dict['data_collection_kwargs']
    data_collection_kwargs.update({
        'mouse_id_list': ['M9']})
    model3_all_tissue_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'data_collection_kwargs': data_collection_kwargs
    })
    return model3_all_tissue_dict


def model3_all_tissue_lactate(test=False):
    """
    Parameter dict for model3_all_tissue_lactate. It contains one source tissue liver,
    one sink tissue from 8 kinds of tissue and three circulatory metabolites glucose and lactate,
    and is fitted with lactate-infused M1 in low-infusion data.
    Most parameters are inherited from model3_all_tissue.

    :param test: If the parameter is for test mode. Sample size in test mode is much smaller, and therefore has
        shorter running time.
    :return: Parameter dict for model3_all_tissue_lactate.
    """

    model_name = "model3_all_tissue_lactate"
    output_direct = "{}/{}".format(global_output_direct, model_name)

    model3_all_tissue_dict = model3_all_tissue(test)
    data_collection_kwargs = model3_all_tissue_dict['data_collection_kwargs']
    data_collection_kwargs.update({
        'label_list': ["lactate"], 'mouse_id_list': ['M3']})
    model3_all_tissue_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'data_collection_kwargs': data_collection_kwargs
    })
    return model3_all_tissue_dict


def model3_all_tissue_lactate_m4(test=False):
    """
    Parameter dict for model3_all_tissue_lactate_m4. It contains one source tissue liver,
    one sink tissue from 8 kinds of tissue and three circulatory metabolites glucose and lactate,
    and is fitted with lactate-infused M4 in low-infusion data.
    Most parameters are inherited from model3_all_tissue.

    :param test: If the parameter is for test mode. Sample size in test mode is much smaller, and therefore has
        shorter running time.
    :return: Parameter dict for model3_all_tissue_lactate_m4.
    """

    model_name = "model3_all_tissue_lactate_m4"
    output_direct = "{}/{}".format(global_output_direct, model_name)

    model3_all_tissue_dict = model3_all_tissue(test)
    data_collection_kwargs = model3_all_tissue_dict['data_collection_kwargs']
    data_collection_kwargs.update({
        'label_list': ["lactate"], 'mouse_id_list': ['M4']})
    model3_all_tissue_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'data_collection_kwargs': data_collection_kwargs
    })
    return model3_all_tissue_dict


def model3_all_tissue_lactate_m10(test=False):
    """
    Parameter dict for model3_all_tissue_lactate_m10. It contains one source tissue liver,
    one sink tissue from 8 kinds of tissue and three circulatory metabolites glucose and lactate,
    and is fitted with lactate-infused M10 in low-infusion data.
    Most parameters are inherited from model3_all_tissue.

    :param test: If the parameter is for test mode. Sample size in test mode is much smaller, and therefore has
        shorter running time.
    :return: Parameter dict for model3_all_tissue_lactate_m10.
    """

    model_name = "model3_all_tissue_lactate_m10"
    output_direct = "{}/{}".format(global_output_direct, model_name)

    model3_all_tissue_dict = model3_all_tissue(test)
    data_collection_kwargs = model3_all_tissue_dict['data_collection_kwargs']
    data_collection_kwargs.update({
        'label_list': ["lactate"], 'mouse_id_list': ['M10']})
    model3_all_tissue_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'data_collection_kwargs': data_collection_kwargs
    })
    return model3_all_tissue_dict


def model3_all_tissue_lactate_m11(test=False):
    """
    Parameter dict for model3_all_tissue_lactate_m11. It contains one source tissue liver,
    one sink tissue from 8 kinds of tissue and three circulatory metabolites glucose and lactate,
    and is fitted with lactate-infused M11 in low-infusion data.
    Most parameters are inherited from model3_all_tissue.

    :param test: If the parameter is for test mode. Sample size in test mode is much smaller, and therefore has
        shorter running time.
    :return: Parameter dict for model3_all_tissue_lactate_m11.
    """

    model_name = "model3_all_tissue_lactate_m11"
    output_direct = "{}/{}".format(global_output_direct, model_name)

    model3_all_tissue_dict = model3_all_tissue(test)
    data_collection_kwargs = model3_all_tissue_dict['data_collection_kwargs']
    data_collection_kwargs.update({
        'label_list': ["lactate"], 'mouse_id_list': ['M11']})
    model3_all_tissue_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'data_collection_kwargs': data_collection_kwargs
    })
    return model3_all_tissue_dict


def model3_unfitted_parameters(test=False):
    """
    Parameter dict for model3_unfitted. The model is same as model3, but fluxes are randomly generated and not
    optimized for data. The objective value is calculated with glucose-infused M1 in low-infusion data.
    Most parameters are inherited from model3_parameters.

    :param test: If the parameter is for test mode. Sample size in test mode is much smaller, and therefore has
        shorter running time.
    :return: Parameter dict for model3_unfitted.
    """

    model_name = "model3_unfitted"
    output_direct = "{}/{}".format(global_output_direct, model_name)

    model3_parameter_dict = model3_parameters(test)
    model3_parameter_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'fitted': False,
        'obj_tolerance': 999999
    })

    return model3_parameter_dict


def model5_parameters(test=False):
    """
    Parameter dict for model5. It contains one source tissue liver, two sink tissue heart and muscle
    and two circulatory metabolites glucose and lactate, and is fitted with glucose-infused M1 in low-infusion data.

    :param test: If the parameter is for test mode. Sample size in test mode is much smaller, and therefore has
        shorter running time.
    :return: Parameter dict for model5.
    """

    model_name = "model5"
    output_direct = "{}/{}".format(global_output_direct, model_name)

    data_collection_kwargs = {
        'label_list': ["glucose"], 'mouse_id_list': ['M1'],
        'source_tissue_marker': constant_set.liver_marker,
        'sink1_tissue_marker': constant_set.heart_marker,
        'sink2_tissue_marker': constant_set.muscle_marker}
    data_loader_func = data_loader_rabinowitz
    data_collection_func = mid_data_collection_model5

    hook_in_each_iteration = metabolite_contribution_model5
    # hook_after_all_iterations = final_processing_dynamic_range_model5
    hook_after_all_iterations = final_processing_dynamic_range_model345
    model_construction_func = model5_construction
    parameter_construction_func = dynamic_range_model345

    complete_flux_list = ['F{}'.format(i + 1) for i in range(10)] + ['G{}'.format(i + 1) for i in range(9)] + \
                         ['H{}'.format(i + 1) for i in range(9)] + ['Fcirc_lac', 'Fcirc_glc']
    complete_flux_dict = {var: i for i, var in enumerate(complete_flux_list)}
    # constant_flux_dict = {'F10': 100, 'Fcirc_lac': 374.4, 'Fcirc_glc': 150.9}
    constant_flux_dict = {'F10': 40, 'Fcirc_lac': 374.4, 'Fcirc_glc': 150.9}

    min_flux_value = 1
    # max_flux_value = 1000
    max_flux_value = 700
    optimization_repeat_time = 10
    obj_tolerance = 0.15
    ternary_sigma = 0.15
    special_bound_dict = {}
    # special_bound_dict = {
    #     'F9': (10, 30),
    #     'G9': (10, 30),
    #     'H9': (10, 30)
    # }

    free_fluxes_name_list = ['F1', 'G2', 'H1', 'F3', 'G4']
    free_fluxes_range_list = [
        [min_flux_value, constant_flux_dict['Fcirc_glc']],
        [min_flux_value, constant_flux_dict['Fcirc_glc']],
        [min_flux_value, constant_flux_dict['Fcirc_glc']],
        [min_flux_value, constant_flux_dict['Fcirc_lac']],
        [min_flux_value, constant_flux_dict['Fcirc_lac']],
    ]

    if test:
        total_point_num = int(6e3)
        # point_interval_list = [50, 50, 20, 20, 100]
        ternary_resolution = int(2 ** 7)
    else:
        total_point_num = int(3e6)
        # point_interval_list = [25, 25, 5, 5, 25]
        ternary_resolution = int(2 ** 8)

    bounds = config.bound_pair_generator(min_flux_value, max_flux_value, complete_flux_list, special_bound_dict)
    return locals()


def model5_comb2_parameters(test=False):
    """
    Parameter dict for model5_comb2. It contains one source tissue liver, two sink tissue brain and muscle
    and two circulatory metabolites glucose and lactate, and is fitted with glucose-infused M1 in low-infusion data.
    Most parameters are inherited from model5_parameters.

    :param test: If the parameter is for test mode. Sample size in test mode is much smaller, and therefore has
        shorter running time.
    :return: Parameter dict for model5_comb2.
    """

    model_name = "model5_comb2"
    output_direct = "{}/{}".format(global_output_direct, model_name)

    model5_parameter_dict = model5_parameters(test)
    data_collection_kwargs = model5_parameter_dict['data_collection_kwargs']
    data_collection_kwargs.update({
        'sink1_tissue_marker': constant_set.brain_marker,
        'sink2_tissue_marker': constant_set.muscle_marker})
    model5_parameter_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'data_collection_kwargs': data_collection_kwargs
    })
    return model5_parameter_dict


def model5_comb3_parameters(test=False):
    """
    Parameter dict for model5_comb3. It contains one source tissue liver, two sink tissue heart and brain
    and two circulatory metabolites glucose and lactate, and is fitted with glucose-infused M1 in low-infusion data.
    Most parameters are inherited from model5_parameters.

    :param test: If the parameter is for test mode. Sample size in test mode is much smaller, and therefore has
        shorter running time.
    :return: Parameter dict for model5_comb3.
    """

    model_name = "model5_comb3"
    output_direct = "{}/{}".format(global_output_direct, model_name)

    model5_parameter_dict = model5_parameters(test)
    data_collection_kwargs = model5_parameter_dict['data_collection_kwargs']
    data_collection_kwargs.update({
        'sink1_tissue_marker': constant_set.heart_marker,
        'sink2_tissue_marker': constant_set.brain_marker})
    model5_parameter_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'data_collection_kwargs': data_collection_kwargs
    })
    return model5_parameter_dict


def model5_unfitted_parameters(test=False):
    """
    Parameter dict for model5_unfitted. The model is same as model5, but fluxes are randomly generated and not
    optimized for data. The objective value is calculated with glucose-infused M1 in low-infusion data.
    Most parameters are inherited from model5_parameters.

    :param test: If the parameter is for test mode. Sample size in test mode is much smaller, and therefore has
        shorter running time.
    :return: Parameter dict for model5_unfitted.
    """

    model_name = "model5_unfitted"
    output_direct = "{}/{}".format(global_output_direct, model_name)

    model5_parameter_dict = model5_parameters(test)
    model5_parameter_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'fitted': False,
        'obj_tolerance': 999999
    })

    return model5_parameter_dict


def model6_parameters(test=False):
    """
    Parameter dict for model6. It contains one source tissue liver, one sink tissue skeletal muscle
    and two circulatory metabolites glucose and lactate, and is fitted with glucose-infused M1 in high-infusion data.

    :param test: If the parameter is for test mode. Sample size in test mode is much smaller, and therefore has
        shorter running time.
    :return: Parameter dict for model6.
    """

    model_name = "model6"
    output_direct = "{}/{}".format(global_output_direct, model_name)

    data_collection_kwargs = {
        'label_list': ["glucose"], 'mouse_id_list': ['M1'],
        'source_tissue_marker': constant_set.liver_marker, 'sink_tissue_marker': constant_set.muscle_marker}
    data_loader_func = data_loader_dan
    data_collection_func = mid_data_collection_model1234

    hook_in_each_iteration = metabolite_contribution_model12
    hook_after_all_iterations = final_processing_dynamic_range_model12
    model_construction_func = model6_construction
    parameter_construction_func = dynamic_range_model12

    complete_flux_list = ['F{}'.format(i + 1) for i in range(10)] + ['G{}'.format(i + 1) for i in range(9)] + \
                         ['Jin', 'Fcirc_lac']
    complete_flux_dict = {var: i for i, var in enumerate(complete_flux_list)}
    constant_flux_dict = {'Jin': 111.1, 'F10': 80, 'Fcirc_lac': 400}

    min_flux_value = 1
    max_flux_value = 1000
    max_free_flux_value = 300
    optimization_repeat_time = 10
    obj_tolerance = 0.25
    f1_range = [min_flux_value, max_free_flux_value]
    g2_range = [min_flux_value, max_free_flux_value]

    if test:
        f1_num = 51
        f1_display_interv = 50
        g2_num = 51
        g2_display_interv = 50
    else:
        f1_num = 1500
        f1_display_interv = 250
        g2_num = 1500
        g2_display_interv = 250
    bounds = config.bound_pair_generator(min_flux_value, max_flux_value, complete_flux_list)
    return locals()


def model6_m2_parameters(test=False):
    """
    Parameter dict for model6_m2. It contains one source tissue liver, one sink tissue skeletal muscle
    and two circulatory metabolites glucose and lactate, and is fitted with glucose-infused M2 in high-infusion data.
    Most parameters are inherited from model6_parameters.

    :param test: If the parameter is for test mode. Sample size in test mode is much smaller, and therefore has
        shorter running time.
    :return: Parameter dict for model6_m2.
    """

    model_name = "model6_m2"
    output_direct = "{}/{}".format(global_output_direct, model_name)

    data_collection_kwargs = {
        'label_list': ["glucose"], 'mouse_id_list': ['M2'],
        'source_tissue_marker': constant_set.liver_marker, 'sink_tissue_marker': constant_set.muscle_marker}
    model6_parameter_dict = model6_parameters(test)
    model6_parameter_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'data_collection_kwargs': data_collection_kwargs,
        'obj_tolerance': 0.4
    })

    return model6_parameter_dict


def model6_m3_parameters(test=False):
    """
    Parameter dict for model6_m3. It contains one source tissue liver, one sink tissue skeletal muscle
    and two circulatory metabolites glucose and lactate, and is fitted with glucose-infused M3 in high-infusion data.
    Most parameters are inherited from model6_parameters.

    :param test: If the parameter is for test mode. Sample size in test mode is much smaller, and therefore has
        shorter running time.
    :return: Parameter dict for model6_m3.
    """

    model_name = "model6_m3"
    output_direct = "{}/{}".format(global_output_direct, model_name)

    data_collection_kwargs = {
        'label_list': ["glucose"], 'mouse_id_list': ['M3'],
        'source_tissue_marker': constant_set.liver_marker, 'sink_tissue_marker': constant_set.muscle_marker}
    model6_parameter_dict = model6_parameters(test)
    model6_parameter_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'data_collection_kwargs': data_collection_kwargs,
        'obj_tolerance': 0.4
    })

    return model6_parameter_dict


def model6_m4_parameters(test=False):
    """
    Parameter dict for model6_m4. It contains one source tissue liver, one sink tissue skeletal muscle
    and two circulatory metabolites glucose and lactate, and is fitted with glucose-infused M4 in high-infusion data.
    Most parameters are inherited from model6_parameters.

    :param test: If the parameter is for test mode. Sample size in test mode is much smaller, and therefore has
        shorter running time.
    :return: Parameter dict for model6_m4.
    """

    model_name = "model6_m4"
    output_direct = "{}/{}".format(global_output_direct, model_name)

    data_collection_kwargs = {
        'label_list': ["glucose"], 'mouse_id_list': ['M4'],
        'source_tissue_marker': constant_set.liver_marker, 'sink_tissue_marker': constant_set.muscle_marker}
    model6_parameter_dict = model6_parameters(test)
    model6_parameter_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'data_collection_kwargs': data_collection_kwargs,
        'obj_tolerance': 0.25
    })

    return model6_parameter_dict


def model6_unfitted_parameters(test=False):
    """
    Parameter dict for model6_unfitted. The model is same as model6, but fluxes are randomly generated and not
    optimized for data. The objective value is calculated with glucose-infused M1 in high-infusion data.
    Most parameters are inherited from model6_parameters.

    :param test: If the parameter is for test mode. Sample size in test mode is much smaller, and therefore has
        shorter running time.
    :return: Parameter dict for model6_unfitted.
    """

    model_name = "model6_unfitted"
    output_direct = "{}/{}".format(global_output_direct, model_name)

    model6_parameter_dict = model6_parameters(test)
    model6_parameter_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'fitted': False,
        'obj_tolerance': 999999
    })

    return model6_parameter_dict


def model7_parameters(test=False):
    """
    Parameter dict for model7. It contains one source tissue liver, one sink tissue heart
    and three circulatory metabolites glucose, lactate and pyruvate, and is fitted with
    glucose-infused M1 in high-infusion data.

    :param test: If the parameter is for test mode. Sample size in test mode is much smaller, and therefore has
        shorter running time.
    :return: Parameter dict for model7.
    """

    model_name = "model7"
    output_direct = "{}/{}".format(global_output_direct, model_name)

    data_collection_kwargs = {
        'label_list': ["glucose"], 'mouse_id_list': ['M1'],
        'source_tissue_marker': constant_set.liver_marker, 'sink_tissue_marker': constant_set.muscle_marker}
    data_loader_func = data_loader_dan
    data_collection_func = mid_data_collection_model1234
    # model_mid_data_dict = data_loader_dan(mid_data_collection_model1234, data_collection_kwargs)

    hook_in_each_iteration = metabolite_contribution_model34
    hook_after_all_iterations = final_processing_dynamic_range_model345
    model_construction_func = model7_construction
    parameter_construction_func = dynamic_range_model345

    complete_flux_list = ['F{}'.format(i + 1) for i in range(12)] + ['G{}'.format(i + 1) for i in range(11)] + \
                         ['J{}'.format(i + 1) for i in range(3)] + ['Fcirc_lac', 'Fcirc_pyr', 'Jin']
    complete_flux_dict = {var: i for i, var in enumerate(complete_flux_list)}
    constant_flux_dict = {'Jin': 111.1, 'F12': 150, 'Fcirc_lac': 400, 'Fcirc_pyr': 70}
    fcirc_glc_max = 200

    min_flux_value = 1
    max_flux_value = 1000  # 2000
    optimization_repeat_time = 10
    obj_tolerance = 0.4
    ternary_sigma = 0.15

    free_fluxes_name_list = ['F1', 'G2', 'F9', 'G10', 'F3']
    free_fluxes_range_list = [
        [min_flux_value, fcirc_glc_max],
        [min_flux_value, fcirc_glc_max],
        [min_flux_value, constant_flux_dict['Fcirc_pyr']],
        [min_flux_value, constant_flux_dict['Fcirc_pyr']],
        [min_flux_value, constant_flux_dict['Fcirc_lac']],
    ]

    if test:
        total_point_num = int(3e3)
        ternary_resolution = int(2 ** 7)
    else:
        total_point_num = int(3e6)
        ternary_resolution = int(2 ** 8)
    bounds = config.bound_pair_generator(min_flux_value, max_flux_value, complete_flux_list)
    return locals()


def model7_m2_parameters(test=False):
    """
    Parameter dict for model7_m2. It contains one source tissue liver, one sink tissue heart
    and three circulatory metabolites glucose, lactate and pyruvate, and is fitted with
    glucose-infused M2 in high-infusion data.
    Most parameters are inherited from model7_parameters.

    :param test: If the parameter is for test mode. Sample size in test mode is much smaller, and therefore has
        shorter running time.
    :return: Parameter dict for model7_m2.
    """

    model_name = "model7_m2"
    output_direct = "{}/{}".format(global_output_direct, model_name)

    model7_dict = model7_parameters(test)
    data_collection_kwargs = model7_dict['data_collection_kwargs']
    data_collection_kwargs.update({
        'mouse_id_list': ['M2']})
    model7_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'data_collection_kwargs': data_collection_kwargs
    })
    return model7_dict


def model7_m3_parameters(test=False):
    """
    Parameter dict for model7_m3. It contains one source tissue liver, one sink tissue heart
    and three circulatory metabolites glucose, lactate and pyruvate, and is fitted with
    glucose-infused M3 in high-infusion data.
    Most parameters are inherited from model7_parameters.

    :param test: If the parameter is for test mode. Sample size in test mode is much smaller, and therefore has
        shorter running time.
    :return: Parameter dict for model7_m3.
    """

    model_name = "model7_m3"
    output_direct = "{}/{}".format(global_output_direct, model_name)

    model7_dict = model7_parameters(test)
    data_collection_kwargs = model7_dict['data_collection_kwargs']
    data_collection_kwargs.update({
        'mouse_id_list': ['M3']})
    model7_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'data_collection_kwargs': data_collection_kwargs
    })
    return model7_dict


def model7_m4_parameters(test=False):
    """
    Parameter dict for model7_m4. It contains one source tissue liver, one sink tissue heart
    and three circulatory metabolites glucose, lactate and pyruvate, and is fitted with
    glucose-infused M4 in high-infusion data.
    Most parameters are inherited from model7_parameters.

    :param test: If the parameter is for test mode. Sample size in test mode is much smaller, and therefore has
        shorter running time.
    :return: Parameter dict for model7_m4.
    """

    model_name = "model7_m4"
    output_direct = "{}/{}".format(global_output_direct, model_name)

    model7_dict = model7_parameters(test)
    data_collection_kwargs = model7_dict['data_collection_kwargs']
    data_collection_kwargs.update({
        'mouse_id_list': ['M4']})
    model7_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'data_collection_kwargs': data_collection_kwargs
    })
    return model7_dict


def model7_unfitted_parameters(test=False):
    """
    Parameter dict for model7_unfitted. The model is same as model7, but fluxes are randomly generated and not
    optimized for data. The objective value is calculated with glucose-infused M1 in high-infusion data.
    Most parameters are inherited from model7_parameters.

    :param test: If the parameter is for test mode. Sample size in test mode is much smaller, and therefore has
        shorter running time.
    :return: Parameter dict for model7_unfitted.
    """

    model_name = "model7_unfitted"
    output_direct = "{}/{}".format(global_output_direct, model_name)

    model7_parameter_dict = model7_parameters(test)
    model7_parameter_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'fitted': False,
        'obj_tolerance': 999999
    })

    return model7_parameter_dict
