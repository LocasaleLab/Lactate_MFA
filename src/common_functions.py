#!/usr/bin/env python
#  -*- coding: utf-8 -*-
# (C) Shiyu Liu, Locasale Lab, 2019
# Contact: liushiyu1994@gmail.com
# All rights reserved
# Licensed under MIT License (see LICENSE-MIT)

"""
    Functions that provide common operations in fitting process and plotting.
"""

import numpy as np
from scipy.special import comb as scipy_comb
import scipy.interpolate
import scipy.signal
import matplotlib.pyplot as plt
import ternary
from ternary.helpers import simplex_iterator

from src import config

constant_set = config.Constants()
color_set = config.Color()

eps_for_log = constant_set.eps_for_log
target_label = constant_set.target_label


def natural_dist(c13_ratio, carbon_num):
    """
    Generate natural MID distribution of a metabolite.

    :param c13_ratio: Abundance of 13C.
    :param carbon_num: Carbon num of this metabolite.
    :return: NumPy array for MID distribution.
    """

    c12_ratio = 1 - c13_ratio
    total_num = carbon_num + 1
    output = []
    for index in range(total_num):
        output.append(
            scipy_comb(carbon_num, index) * c13_ratio ** index * c12_ratio ** (carbon_num - index))
    return np.array(output)


def split_equal_dist(source_mid, target_carbon_num):
    """
    Split MID of a large metabolite to MID of two small metabolites. Usually used to approximate
    MID of pyruvate generated from glucose.

    :param source_mid: MID of source metabolite, usually glucose.
    :param target_carbon_num: Carbon number of target metabolite, usually 3 (pyruvate).
    :return: MID of target metabolite generated from source metabolite.
    """

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

    return final_output_vector


def collect_all_data(
        data_dict, metabolite_name, label_list, tissue, mouse_id_list=None, convolve=False,
        split=0, mean=True):
    """
    This function will select data based on metabolite, label, tissue and mouse ID, and do some primary process,
    such as convolution of small metabolites or split of large metabolites.

    :param data_dict: Complete raw data dict.
    :param metabolite_name: Required metabolite name.
    :param label_list: Required list of experimental labels.
    :param tissue: Required tissue name.
    :param mouse_id_list: Required list of mouse ID.
    :param convolve: Whether convolve MID of source metabolites.
    :param split: Whether split MID of source metabolites.
    :param mean: Whether return mean of all collected data.
    :return: MID of target metabolites in target data set.
    """

    matrix = []
    for label in label_list:
        if mouse_id_list is None:
            mouse_id_list = data_dict[label].keys()
        for mouse_label in mouse_id_list:
            data_for_mouse = data_dict[label][mouse_label]
            data_vector = data_for_mouse[tissue][metabolite_name]
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


def flux_balance_constraint_constructor(balance_list, complete_flux_dict):
    """
    Function to construct flux balance matrix. List of flux balance dict is import and converted to flux balance
    matrix A and constant b, which let A @ x = b, where x is flux vector.

    :param balance_list: List of flux balance dict, which includes input and output fluxes that need to balance.
    :param complete_flux_dict: Flux name to index dict to ensure all fluxes have same order.
    :return: The matrix A and constant vector b that makes A @ x = b is the flux balance requirement.
    """

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
    """
    Function to construct MID calculation matrix. List of MID relationship is import and converted to two matrix:
    substrate MID matrix S and flux sum matrix F, which let predicted MID vector M = (S @ x) / (F @ x), where x is
    flux vector and division is element-wise.

    :param mid_constraint_list: List of MID constraint dict.
    :param complete_flux_dict: Flux name to index dict to ensure all fluxes have same order.
    :return: Substrate MID matrix S and flux sum matrix F, which let predicted MID vector M = (S @ x) / (F @ x).
        Target MID vector and optimal objective value are also returned to compare with predicted MID vector.
    """

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
    """
    Function to construct constraint matrix of constant flux. Requirements such as F1 = a1, F2 = a2 will be
    converted to matrix form like A @ x = b, where x is flux vector.

    :param constant_flux_dict: A dict to provide requirements for constant fluxes.
    :param complete_flux_dict: Flux name to index dict to ensure all fluxes have same order.
    :return: Constant matrix A and vector b that A @ x = b is the constant requirement.
    """

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


def calculate_one_tissue_tca_contribution(input_net_flux_list):
    """
    Calculate TCA contribution ratio in one tissue based on all net fluxes connected to pyruvate node.
    Contribution fluxes are non-negative, and its number is equal to number of input net fluxes.
    An contribution array is returned with the same length as input_net_flux_list, and each number corresponds
    to contribution ratio of one net flux in that list.

    :param input_net_flux_list: List of all net fluxes connected to pyruvate node.
    :return: An array of contribution ratio. Its length is equal to length of input_net_flux_list.
    """

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
    real_flux_array = np.array(real_flux_list)
    return real_flux_array


def one_time_prediction(predicted_vector_dim, mid_constraint_dict, flux_value_dict):
    """
    Predict one target MID based on given MID constraints and value of all fluxes.

    :param predicted_vector_dim: Dimension of predicted vector.
    :param mid_constraint_dict: Dict of components used to calculate predicted MID.
    :param flux_value_dict: Dict from flux name to value of all fluxes.
    :return: Array of predicted MID.
    """

    predicted_vector = np.zeros(predicted_vector_dim)
    total_flux_value = 0
    for flux_name, mid_vector in mid_constraint_dict.items():
        if flux_name == target_label:
            continue
        else:
            flux_value = flux_value_dict[flux_name]
            total_flux_value += flux_value
            predicted_vector += flux_value * mid_vector
    predicted_vector /= total_flux_value
    return predicted_vector


def evaluation_for_one_flux(result_dict, constant_dict, mid_constraint_list, mid_size_dict):
    """
    Predict MID of all target metabolites in a complete model based on given MID constraints list,
    result flux dict and constant flux dict.

    :param result_dict: Dict of result fluxes from a solution.
    :param constant_dict: Dict of all constant fluxes.
    :param mid_constraint_list: List of MID constraints for all target metabolites.
    :param mid_size_dict: Dict of MID size of all target metabolites.
    :return: Dict of all predicted MID.
    """

    flux_value_dict = dict(result_dict)
    flux_value_dict.update(constant_dict)
    predicted_mid_dict = {}
    for mid_constraint_dict in mid_constraint_list:
        name = "_".join([name for name in mid_constraint_dict.keys() if name != 'target'])
        predicted_vector = one_time_prediction(mid_size_dict[name], mid_constraint_dict, flux_value_dict)
        predicted_mid_dict[name] = predicted_vector
    return predicted_mid_dict


def one_case_mid_prediction(result_dict, mid_constraint_list, mid_size_dict, predicted_mid_collection_dict):
    """
    Function to predict MID given a optimization result.
    :param result_dict: Optimization result which includes flux value.
    :param mid_constraint_list: List of an
    :param mid_size_dict:
    :param predicted_mid_collection_dict:
    :return:
    """
    predicted_mid_dict = evaluation_for_one_flux(
        result_dict, {}, mid_constraint_list, mid_size_dict)
    for mid_name, mid_vector in predicted_mid_dict.items():
        if mid_name not in predicted_mid_collection_dict:
            predicted_mid_collection_dict[mid_name] = []
        predicted_mid_collection_dict[mid_name].append(mid_vector)


def append_flux_distribution(result_dict, feasible_flux_distribution_dict):
    """
    Function to append fluxes in result dict to feasible_flux_distribution_dict
    :param result_dict: Dict of optimization result.
    :param feasible_flux_distribution_dict: Dict to collect all feasible fluxes. Each element contains a list.
    :return:
    """
    for flux_name, flux_value in result_dict.items():
        if flux_name not in feasible_flux_distribution_dict:
            feasible_flux_distribution_dict[flux_name] = []
        feasible_flux_distribution_dict[flux_name].append(flux_value)


def mid_prediction_preparation(tissue_name_list, mid_constraint_list_dict):
    """
    Function to collect target vector in each tissue and size of all MID.
    :param tissue_name_list: List of all tissue names.
    :param mid_constraint_list_dict: List of all MID constraints with 8 kinds of sink tissue.
    :return: Dict of target MID of metabolites and dict of MID size.
    """
    target_vector_dict = {tissue_name: {} for tissue_name in tissue_name_list}
    mid_size_dict = {}
    for tissue_name, mid_constraint_list in mid_constraint_list_dict.items():
        for mid_constraint_dict in mid_constraint_list:
            target_vector = mid_constraint_dict[constant_set.target_label]
            name = "_".join([name for name in mid_constraint_dict.keys() if name != 'target'])
            target_vector_dict[tissue_name][name] = target_vector
            if name not in mid_size_dict:
                mid_size_dict[name] = len(target_vector)
    return target_vector_dict, mid_size_dict


def plot_heat_map(
        data_matrix, x_free_variable, y_free_variable, matrix_range=(None, None),
        cmap=None, cbar_name=None, save_path=None):
    """
    Plot heat map from a matrix.

    :param data_matrix: Data matrix for heat map.
    :param x_free_variable: FreeVariable object of x-axis.
    :param y_free_variable: FreeVariable object of y-axis.
    :param matrix_range: Artificial range of value in matrix. Default is None for min and max value.
    :param cmap: Colormap for heat map.
    :param cbar_name: Title for colorbar in figure.
    :param save_path: Save path for the whole figure.
    :return: None
    """

    fig, ax = plt.subplots()
    im = ax.imshow(data_matrix, cmap=cmap, vmin=matrix_range[0], vmax=matrix_range[1])
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


def plot_violin_distribution(data_dict, color_dict=None, cutoff=0.5, title=None, save_path=None):
    """
    Plot violin graph for distributions of a set of data.

    :param data_dict: Dict of the data set, in which key is their name and value is data array.
    :param color_dict: Dict of color used in violin plot.
    :param cutoff: A dash line introduced to evaluate median of distribution.
    :param title: Figure title.
    :param save_path: Save path for the whole figure.
    :return: None
    """

    fig, ax = plt.subplots()
    data_list_for_violin = data_dict.values()
    tissue_label_list = data_dict.keys()
    x_axis_position = np.arange(1, len(tissue_label_list) + 1)

    parts = ax.violinplot(data_list_for_violin, showmedians=True, showextrema=True)
    if color_dict is not None:
        if isinstance(color_dict, np.ndarray):
            new_color_dict = {key: color_dict for key in data_dict}
            color_dict = new_color_dict
        color_list = [color_dict[key] for key in tissue_label_list]
        parts['cmaxes'].set_edgecolor(color_list)
        parts['cmins'].set_edgecolor(color_list)
        parts['cbars'].set_edgecolor(color_list)
        parts['cmedians'].set_edgecolor(color_set.orange)
        for pc, color in zip(parts['bodies'], color_list):
            pc.set_facecolor(color)
            pc.set_alpha(color_set.alpha_value)
    if cutoff is not None:
        ax.axhline(cutoff, linestyle='--', color=color_set.orange)
    ax.set_ylim([-0.1, 1.1])
    ax.set_xticks(x_axis_position)
    ax.set_xticklabels(tissue_label_list)
    if title:
        ax.set_title(title)
    if save_path:
        # print(save_path)
        fig.savefig(save_path, dpi=fig.dpi)


def plot_box_distribution(data_dict, save_path=None, title=None, broken_yaxis=None):
    """
    Plot box graph for distributions of a set of data.

    :param data_dict: Dict of the data set, in which key is their name and value is data array.
    :param save_path: Save path for the whole figure.
    :param title: Figure title.
    :param broken_yaxis: Deprecated. Whether y-axis is broken.
    :return: None
    """

    def color_edges(box_parts):
        for part_name, part_list in box_parts.items():
            if part_name == 'medians':
                current_color = color_set.orange
            else:
                current_color = color_set.blue
            for part in part_list:
                part.set_color(current_color)

    data_list_for_box = data_dict.values()
    tissue_label_list = data_dict.keys()
    x_axis_position = np.arange(1, len(tissue_label_list) + 1)

    if broken_yaxis is None:
        fig, ax = plt.subplots()
        parts = ax.boxplot(data_list_for_box, whis='range')
        color_edges(parts)
        ax.set_xticks(x_axis_position)
        ax.set_xticklabels(tissue_label_list)
        if title:
            ax.set_title(title)
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        parts1 = ax1.boxplot(data_list_for_box, whis='range')
        parts2 = ax2.boxplot(data_list_for_box, whis='range')
        color_edges(parts1)
        color_edges(parts2)
        ax1.set_ylim([broken_yaxis[1], None])
        ax2.set_ylim([-50, broken_yaxis[0]])
        ax1.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax1.xaxis.tick_top()
        ax2.set_xticks(x_axis_position)
        ax2.set_xticklabels(tissue_label_list)

    if save_path:
        fig.savefig(save_path, dpi=fig.dpi)


def plot_ternary_density(
        tri_data_matrix, sigma: float = 1, bin_num: int = 2 ** 8, mean=False, title=None, save_path=None):
    """
    Plot ternary graph to display density of points in tri_data_matrix.

    In cartesian cor, the left bottom corner of triangle is the origin.

    :param tri_data_matrix: Data matrix of points in triangle space. Each row of data matrix is a point
        in triple tuple. The scale of all triangle points is 1.
        Order of ternary cor: x1: bottom (to right) x2: right (to left) x3: left (to bottom)
    :param sigma: Parameter for Gaussian kernal to generate density.
    :param bin_num: Bin number of ternary plot.
    :param mean: Whether mean of all points are displayed.
    :param title: Figure title.
    :param save_path: Save path for the whole figure.
    :return: None
    """

    sqrt_3 = np.sqrt(3)

    def standard_2dnormal(x, y, _sigma):
        """
        Calculate 2D Gaussian distribution based on x, y, and _sigma.
        Center is origin and correlation coefficient is 0.
        :param x: Input variable, could be matrix.
        :param y: Input variable, coule be matrix.
        :param _sigma: Parameter for Gaussian distribution.
        :return: Value of 2D Gaussian distribution on x and y.
        """
        return np.exp(-0.5 / _sigma ** 2 * (x ** 2 + y ** 2)) / (2 * np.pi * _sigma ** 2)

    def tri_to_car(input_data_matrix):
        """
        Convert points in triangle space to those in Cartesian space.
        Each row is one point in both triangle and Cartesian space.
        :param input_data_matrix: Coordinates of points in triangle space.
        :return: Coordinates of points in Cartesian space.
        """
        y_value = input_data_matrix[:, 1] * sqrt_3 / 2
        x_value = input_data_matrix[:, 0] + y_value / sqrt_3
        return np.vstack([x_value, y_value]).T

    def car_to_tri(input_data_matrix):
        """
        Convert points in Cartesian space to those in triangle space.
        Each row is one point in both triangle and Cartesian space.
        :param input_data_matrix: Coordinates of points in Cartesian space.
        :return: Coordinates of points in triangle space.
        """
        y_value = input_data_matrix[:, 1]
        x2_value = y_value / (sqrt_3 / 2)
        x1_value = input_data_matrix[:, 0] - y_value / sqrt_3
        return np.vstack([x1_value, x2_value]).T

    def gaussian_kernel_generator(_bin_num, _sigma):
        """
        Generate kernel matrix of Gaussian distribution given current resolution. Range is [0, 1] on both axes.
        :param _bin_num: Resolution of the kernel, equal to size of final kernel matrix.
        :param _sigma: Parameter for Gaussian distribution.
        :return: Kernel matrix of Gaussian distribution.
        """
        x = np.linspace(0, 1, _bin_num) - 0.5
        y = np.linspace(0, 1, _bin_num) - 0.5
        X, Y = np.meshgrid(x, y)
        gaussian_kernel = standard_2dnormal(X, Y, _sigma)
        return np.rot90(gaussian_kernel)

    def bin_car_data_points(_car_data_matrix, _bin_num):
        """
        Use 2D lattice to discretize and bin points in Cartesian space to generate frequency matrix.
        Range is [0, 1] on both axes. Each row is one point in Cartesian space.
        :param _car_data_matrix: Coordinates of points in Cartesian space.
        :param _bin_num: Number of bins used to discretize.
        :return: Frequency matrix of each bin.
        """

        histogram, _, _ = np.histogram2d(
            _car_data_matrix[:, 0], _car_data_matrix[:, 1], bins=np.linspace(0, 1, _bin_num + 1))
        return histogram

    def complete_tri_set_interpolation(_location_list, _value_list, _scale):
        """
        Because the color function in ternary package only accept coloring based on certain hexagonal points,
        color on those points need to be determined by interpolation based on density matrix in Cartesian space.
        Therefore, hexagonal points are first converted to Cartesian space, and their value are interpolated
        based on density matrix to generate value for plotting.
        :param _location_list: List of locations of density matrix in Cartesian space.
        :param _value_list: List of density value of density matrix in Cartesian space.
        :param _scale: Scale of ternary plot, which determined number of hexagonal points in triangle space.
        :return: Dict that map location in triangle space to its value (also color) in final ternary plot.
        """
        result_tri_array = np.array(list(simplex_iterator(_scale))) / _scale
        result_car_array = tri_to_car(result_tri_array)
        # The set of locations of hexagonal points result_car_array is generated. Its density value is determined by
        # interpolation with location and value of density matrix.
        result_value_array = scipy.interpolate.griddata(
            np.array(_location_list), np.array(_value_list), result_car_array, method='cubic')
        # Generate Dict to store the location-value mapping of hexagonal points
        target_dict = {}
        for (i, j, k), result_value in zip(simplex_iterator(bin_num), result_value_array):
            target_dict[(i, j)] = result_value
        return target_dict

    # Input points in triangle space are transformed to those in Cartesian space, and binned to
    # generate density matrix. Density matrix is than convolved with Gaussian kernel to be smoothed.
    car_data_matrix = tri_to_car(tri_data_matrix)
    data_bin_matrix = bin_car_data_points(car_data_matrix, bin_num)
    gaussian_kernel_matrix = gaussian_kernel_generator(bin_num, sigma)
    car_blurred_matrix = scipy.signal.convolve2d(data_bin_matrix, gaussian_kernel_matrix, mode='same')

    # Generate location and value list to calculate density dict of standard hexagonal points in ternary plot.
    x_axis = y_axis = np.linspace(0, 1, bin_num)
    location_list = []
    value_list = []
    for x_index, x_value in enumerate(x_axis):
        for y_index, y_value in enumerate(y_axis):
            location_list.append([x_value, y_value])
            value_list.append(car_blurred_matrix[x_index, y_index])
    complete_density_dict = complete_tri_set_interpolation(location_list, value_list, bin_num)

    # Plot the final density dict to ternary figure.
    fig, tax = ternary.figure(scale=bin_num)
    tax.heatmap(complete_density_dict, cmap='Blues', style="h")
    tax.boundary(linewidth=1.0)
    tick_labels = list(np.linspace(0, bin_num, 6) / bin_num)
    tax.ticks(axis='lbr', ticks=tick_labels, linewidth=1, tick_formats="")
    tax.clear_matplotlib_ticks()
    tax.set_title(title)
    plt.tight_layout()
    if mean:
        mean_value = tri_data_matrix.mean(axis=0).reshape([1, -1]) * bin_num
        tax.scatter(mean_value, marker='o', color=color_set.orange, zorder=100)
    if save_path:
        print(save_path)
        fig.savefig(save_path, dpi=fig.dpi)
    # tax.show()
