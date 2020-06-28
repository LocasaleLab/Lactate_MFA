#!/usr/bin/env python
#  -*- coding: utf-8 -*-
# (C) Shiyu Liu, Locasale Lab, 2019
# Contact: liushiyu1994@gmail.com
# All rights reserved
# Licensed under MIT License (see LICENSE-MIT)

"""
    Configurations for some constants and common classes.
"""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap


class Result(object):
    """
    Class to contain results, which include necessary components as a result.
    """

    def __init__(
            self, result_dict: dict, obj_value: float, success: bool, minimal_obj_value: float, label: dict):
        self.result_dict = result_dict
        self.obj_value = obj_value
        self.success = success
        self.minimal_obj_value = minimal_obj_value
        self.label = label

    def __repr__(self):
        """
        Overloading to provide a user-friendly display for result objects.
        """
        return "Result: {}\nObjective value: {}\nSuccess: {}\nMinimal objective value: {}".format(
            self.result_dict, self.obj_value, self.success, self.minimal_obj_value)


class FreeVariable(object):
    """
    Class that contains necessary component for a free variable, such as F1 and G2.
    """

    def __init__(self, name, total_num, var_range, display_interv):
        self.flux_name = name
        self.total_num = total_num + 1
        self.range = var_range
        self.display_interv = display_interv
        self.value_array = np.linspace(*self.range, self.total_num)
        self.tick_in_range = np.arange(0, self.total_num, self.display_interv, dtype='int')
        self.tick_labels = np.around(self.value_array[self.tick_in_range])

    def __iter__(self):
        """
        Overload the __iter__ method for iterating operation.
        """
        return self.value_array.__iter__()


def rgba_to_rgb(raw_rgb, alpha, background):
    """
    Convert color in RGBA to RGB.
    :param raw_rgb: RGB components in RGBA form.
    :param alpha: Transparency of RGBA color.
    :param background: Background color.
    :return: Transformed RGB color.
    """
    return raw_rgb * alpha + background * (1 - alpha)


def empty_vector(n):
    """
    Construct an vector full of NaN with size n.
    :param n: Size of the NaN vector.
    :return: The NaN vector.
    """
    a = np.empty(n)
    a.fill(np.nan)
    return a


def bound_pair_generator(min_flux_value, max_flux_value, flux_list, special_bound_dict=None):
    """
    Function that convert uniform min and max flux value to two arrays which contain min and max value of all fluxes.
    Some special bounds can also be inserted into those bounds.

    :param min_flux_value: Uniform min flux value.
    :param max_flux_value: Uniform max flux value.
    :param flux_list: List of all flux name.
    :param special_bound_dict: Special bounds for certain fluxes.
    :return: Generated min and max bound pair.
    """

    if special_bound_dict is None:
        bound_pair_list = ([min_flux_value] * len(flux_list), [max_flux_value] * len(flux_list))
    else:
        bound_pair_list = ([], [])
        for flux_name in flux_list:
            if flux_name in special_bound_dict:
                special_min, special_max = special_bound_dict[flux_name]
                bound_pair_list[0].append(special_min)
                bound_pair_list[1].append(special_max)
            else:
                bound_pair_list[0].append(min_flux_value)
                bound_pair_list[1].append(max_flux_value)
    return bound_pair_list


class Color(object):
    """
    Some constants related to color.
    """
    white = np.array([1, 1, 1])
    blue = np.array([21, 113, 177]) / 255
    orange = np.array([251, 138, 68]) / 255
    purple = np.array([112, 48, 160]) / 255
    light_blue = np.array([221, 241, 255]) / 255

    alpha_value = 0.3
    alpha_for_bar_plot = alpha_value + 0.1
    alpha_for_heatmap = alpha_value + 0.2

    color_list = [
        rgba_to_rgb(blue, alpha_for_heatmap, white), white,
        rgba_to_rgb(orange, alpha_for_heatmap, white)]
    blue_orange_cmap = LinearSegmentedColormap.from_list('BlOr', color_list, N=200)


class Constants(object):
    """
    Some constants in model.
    """
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
    eps_of_mid = 1e-5

    data_direct = "data"
    output_direct = "new_models"
    default_tissue_name = ''
