import warnings
import pickle

import numpy as np
from scipy.misc import comb
from scipy.stats import t
from scipy.stats import rv_discrete
import matplotlib.pyplot as plt

import data_parser as data_parser
import config

color_set = config.Color()


class ArbitraryDist(object):
    def __init__(self, segment_point_array, probability_array):
        self.segment_point_array = segment_point_array
        if probability_array.min() < -1e-5:
            raise ValueError("Negative probability!")
        self.probability_array = probability_array
        self.probability_array[probability_array < 0] = 0
        self.probability_array /= np.sum(self.probability_array)
        if len(self.segment_point_array) != len(self.probability_array) + 1:
            raise ValueError("Inconsistent length! Segment point: {}, Probability: {}".format(
                len(self.segment_point_array), len(self.probability_array)))

    def rvs(self, size):
        bin_result = np.random.choice(len(self.probability_array), p=self.probability_array, size=size)
        result_list = []
        for bin_index in bin_result:
            bin_start = self.segment_point_array[bin_index]
            bin_end = self.segment_point_array[bin_index + 1]
            bin_width = bin_end - bin_start
            result_list.append(np.random.random() * bin_width + bin_start)
        return np.array(result_list)


class NormalDist(object):
    def __init__(self, miu, sigma):
        self.miu = miu
        self.sigma = sigma

    def rvs(self, size):
        return np.random.normal(self.miu, self.sigma, size=size)


def solve_two_ratios(source1, source2, target):
    if not (len(source1) == len(source2) == len(target)):
        raise ValueError("Length of 3 vectors are not equal !!!")
    a = (source1 - source2).reshape([-1, 1])
    b = target - source2
    result = np.linalg.lstsq(a, b)
    residual_sum = np.sum(result[1] ** 2)
    total_sum = np.sum((b - np.mean(b)) ** 2)
    degree_deter = 1 - residual_sum / total_sum
    print(degree_deter)
    if degree_deter < 0.8:
        warnings.warn("Degree of determination is too low: {:.2f}".format(degree_deter))
    coeff = result[0][0]
    point_num = len(source1)
    var_num = 1
    df = point_num - var_num - 1
    residual_stderr = np.sqrt(residual_sum / df)
    a_stderr = a.std()
    coeff_stderr = residual_stderr / a_stderr

    segment_num = 1000
    pdf_low_limit = 1e-10
    lb = 0.1
    ub = 0.9
    x_linespace = np.linspace(lb, ub, segment_num)
    x_cdf_linespace = t.cdf(x_linespace, df=df, loc=coeff, scale=coeff_stderr)
    x_pdf_linespace = x_cdf_linespace[1:] - x_cdf_linespace[:-1]
    x_pdf_linespace[x_pdf_linespace < pdf_low_limit] = pdf_low_limit
    x_prob = x_pdf_linespace / np.sum(x_pdf_linespace)

    coeff_dist_obj = ArbitraryDist(x_linespace, x_prob)

    modified_coeff = min(max(lb, coeff), ub)

    return modified_coeff, 1 - modified_coeff, coeff_dist_obj


def natural_dist(c13_ratio, carbon_num):
    c12_ratio = 1 - c13_ratio
    total_num = carbon_num + 1
    output = []
    for index in range(total_num):
        output.append(comb(carbon_num, index) * c13_ratio ** index * c12_ratio ** (carbon_num - index))
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
    return final_output_vector


def construct_model_primary(data_collect_dict):
    for experimental_label in ["glucose", "lactate"]:
        current_data_collect = data_collect_dict[experimental_label]
        glucose_in_serum = current_data_collect.serum_mids["glucose"]
        pyruvate_in_tissue = current_data_collect.tissue_mids["pyruvate"]
        glucose_in_tissue = current_data_collect.tissue_mids["glucose"]
        glucose_from_pyruvate = np.convolve(pyruvate_in_tissue, pyruvate_in_tissue)
        f1, f6, *_ = solve_two_ratios(glucose_in_serum, glucose_from_pyruvate, glucose_in_tissue)
        lactate_in_serum = current_data_collect.serum_mids["lactate"]
        lactate_in_tissue = current_data_collect.tissue_mids["lactate"]
        f3, f8, *_ = solve_two_ratios(lactate_in_serum, pyruvate_in_tissue, lactate_in_tissue)
        pyruvate_from_glucose = split_equal_dist(glucose_in_tissue, 3)
        f5, f7, *_ = solve_two_ratios(pyruvate_from_glucose, lactate_in_tissue, pyruvate_in_tissue)
        print(experimental_label)
        f_str = "{:.10f}"
        print(f_str.format(f1 / f6))
        print(f_str.format(f3 / f8))
        print(f_str.format(f5 / f7))


def solve_single_mid(data_collect_dict):
    def collect_all_data(_label_list, tissue_or_serum, metabolite_name, data_dict, convolve=False, split=False):
        matrix = []
        for label in _label_list:
            current_data_collect = data_dict[label]
            if tissue_or_serum == "serum":
                data_vector = current_data_collect.serum_mids[metabolite_name]
            else:
                data_vector = current_data_collect.tissue_mids[metabolite_name]
            if convolve:
                data_vector = np.convolve(data_vector, data_vector)
            elif split:
                data_vector = split_equal_dist(data_vector, 3)
            matrix.append(data_vector)
        result_matrix = np.array(matrix).transpose()
        return result_matrix

    label_list = ["glucose", "lactate"]
    glucose_in_serum = collect_all_data(label_list, "serum", "glucose", data_collect_dict)
    pyruvate_in_tissue = collect_all_data(label_list, "tissue", "pyruvate", data_collect_dict)
    glucose_from_pyruvate = collect_all_data(label_list, "tissue", "pyruvate", data_collect_dict, convolve=True)
    glucose_in_tissue = collect_all_data(label_list, "tissue", "glucose", data_collect_dict)
    f1, f6, *_ = solve_two_ratios(
        glucose_in_serum.reshape(-1, order='F'), glucose_from_pyruvate.reshape(-1, order='F'),
        glucose_in_tissue.reshape(-1, order='F'))
    lactate_in_serum = collect_all_data(label_list, "serum", "lactate", data_collect_dict)
    lactate_in_tissue = collect_all_data(label_list, "tissue", "lactate", data_collect_dict)
    f3, f8, *_ = solve_two_ratios(
        lactate_in_serum.reshape(-1, order='F'), pyruvate_in_tissue.reshape(-1, order='F'),
        lactate_in_tissue.reshape(-1, order='F'))
    pyruvate_from_glucose = split_equal_dist(glucose_in_tissue, 3)
    f5, f7, *_ = solve_two_ratios(
        pyruvate_from_glucose.reshape(-1, order='F'), lactate_in_tissue.reshape(-1, order='F'),
        pyruvate_in_tissue.reshape(-1, order='F'))

    f_str = "{:.10f}"
    print(f_str.format(f1 / f6))
    print(f_str.format(f3 / f8))
    print(f_str.format(f5 / f7))
    # print(f1, f6)
    # print(f3, f8)
    # print(f5, f7)
    return f1 / f6, f3 / f8, 0.01


def collect_all_data(_label_list, _tissue, _metabolite_name, data_dict, convolve=False, split=0):
    matrix = []
    for label in _label_list:
        for data_for_mouse in data_dict[label].values():
            data_vector = data_for_mouse[_tissue][_metabolite_name]
            if convolve:
                data_vector = np.convolve(data_vector, data_vector)
            elif split != 0:
                data_vector = split_equal_dist(data_vector, split)
            matrix.append(data_vector)
    result_matrix = np.array(matrix).transpose()
    return result_matrix


def solve_mid_distribution(data_collect_dict, label_list, dist_or_mean="dist", tissue_marker='Lv'):

    serum_marker = 'Sr'
    glucose_in_serum = collect_all_data(label_list, serum_marker, "glucose", data_collect_dict)
    pyruvate_in_tissue = collect_all_data(label_list, tissue_marker, "pyruvate", data_collect_dict)
    glucose_from_pyruvate = collect_all_data(label_list, tissue_marker, "pyruvate", data_collect_dict, convolve=True)
    glucose_in_tissue = collect_all_data(label_list, tissue_marker, "glucose", data_collect_dict)
    pyruvate_from_glucose = collect_all_data(label_list, tissue_marker, "glucose", data_collect_dict, split=3)

    f1, f6, f1_ratio_dist = solve_two_ratios(
        glucose_in_serum.reshape(-1, order='F'), glucose_from_pyruvate.reshape(-1, order='F'),
        glucose_in_tissue.reshape(-1, order='F'))
    lactate_in_serum = collect_all_data(label_list, serum_marker, "lactate", data_collect_dict)
    lactate_in_tissue = collect_all_data(label_list, tissue_marker, "lactate", data_collect_dict)
    f3, f8, f3_ratio_dist = solve_two_ratios(
        lactate_in_serum.reshape(-1, order='F'), pyruvate_in_tissue.reshape(-1, order='F'),
        lactate_in_tissue.reshape(-1, order='F'))
    f5, f7, f5_ratio_dist = solve_two_ratios(
        pyruvate_from_glucose.reshape(-1, order='F'), lactate_in_tissue.reshape(-1, order='F'),
        pyruvate_in_tissue.reshape(-1, order='F'))
    if dist_or_mean == "dist":
        return f1_ratio_dist, f3_ratio_dist, f5_ratio_dist
    elif dist_or_mean == "mean":
        return f1, f3, f5
    else:
        raise ValueError("Error in dist_or_mean: {}".format(dist_or_mean))


def solve_infuse_ratio(data_collect_dict, tissue_marker_list):
    serum_marker = 'Sr'
    glucose_infused = np.array([0, 0, 0, 0, 0, 0, 1])
    result_ratio_list = []
    for data_for_mouse in data_collect_dict["glucose"].values():
        try:
            glucose_in_serum = data_for_mouse[serum_marker]["glucose"]
            for tissue_marker in tissue_marker_list:
                glucose_in_tissue = data_for_mouse[tissue_marker]["glucose"]
                f_infuse, f_circ, f_infuse_ratio_dist = solve_two_ratios(
                    glucose_infused, glucose_in_tissue, glucose_in_serum)
                result_ratio_list.append(f_infuse)
        except KeyError:
            continue
    mean_infuse_ratio = np.array(result_ratio_list).mean()
    stderr_infuse_ratio = np.array(result_ratio_list).std()
    return mean_infuse_ratio, stderr_infuse_ratio


def solve_one_case_model1(a, b, c, g1, g2, x):
    y = - b * g2 + b / c * g1 + b / (a * c) * x
    f2 = x - g1
    f3 = y
    f4 = y - g2
    f6 = x / a
    f5 = g1 + f6
    f8 = y / b
    f7 = g2 + f8
    f1 = x
    return [f1, f2, f3, f4, f5, f6, f7, f8]


def solve_one_case_model2(a, b, c, const1, const2, x):
    f10 = x
    f1 = const1
    f3 = const2
    f6 = f1 / a
    f8 = f3 / b
    y = (x + f6) / c - f8
    f11 = y
    f2 = f1 - x
    f4 = f3 - y
    f5 = x + f6
    f7 = y + f8
    f9 = f10 + f11
    return [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11]


# Model for Rabinowitz data, including supplement flux in source tissue.
def solve_one_case_model3(
        a_liver, a_tissue, b_liver, b_tissue, c_liver, c_tissue, f_circ_gluc, f_circ_lac, x1, x2, f_input,
        print=lambda *x: None):
    print("a_liver={}, a_tissue={}, b_liver={}, b_tissue={}, c_liver={}, c_tissue={}".format(
        a_liver, a_tissue, b_liver, b_tissue, c_liver, c_tissue))
    f1 = x1
    g2 = x2
    g1 = f_circ_gluc - f1
    g6 = g1 / a_tissue
    g5 = g6 + g1 - g2
    g7 = g5 / c_tissue
    print("g1={}, g2={}, g5={}, g6={}, g7={}".format(g1, g2, g5, g6, g7))

    f10 = f_input
    f2 = f_circ_gluc - g2
    f6 = f1 / a_liver - f10
    f5 = f6 + f1 - f2 + f10
    f7 = f5 / c_liver
    print("f1={}, f2={}, f5={}, f6={}, f7={}".format(f1, f2, f5, f6, f7))
    A = np.array([[1, 1 + 1 / b_tissue], [1, 1 + 1 / b_liver]])
    B = np.array([f_circ_lac + g7, (1 + 1 / b_liver) * f_circ_lac - f7])
    print(A, B)
    [y1, y2] = np.linalg.solve(A, B)
    print("f4={}, g3={}".format(y1, y2))
    g3 = y2
    f4 = y1
    g4 = f_circ_lac - f4
    f3 = f_circ_lac - g3
    g8 = g3 / b_tissue
    f8 = f3 / b_liver
    g9 = g5 - g6 - (g8 - g7)
    f9 = f7 - f8 - (f6 - f5)
    return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, g1, g2, g3, g4, g5, g6, g7, g8, g9])


# Model for our mouse infusion data, including direct supplement to glucose pool in plasma
def solve_one_case_model4(
        a_liver, a_tissue, b_liver, b_tissue, c_liver, c_tissue, f_circ_gluc, f_circ_lac,
        x1, x2, f_infuse_gluc, print=lambda *x: None):
    print("a_liver={}, a_tissue={}, b_liver={}, b_tissue={}, c_liver={}, c_tissue={}".format(
        a_liver, a_tissue, b_liver, b_tissue, c_liver, c_tissue))
    f1 = x1
    g2 = x2
    g1 = f_circ_gluc + f_infuse_gluc - f1
    g6 = g1 / a_tissue
    g5 = g6 + g1 - g2
    g7 = g5 / c_tissue
    print("g1={}, g2={}, g5={}, g6={}, g7={}".format(g1, g2, g5, g6, g7))

    f10 = f_infuse_gluc
    f2 = f_circ_gluc - g2
    f6 = f1 / a_liver
    f5 = f6 + f1 - f2
    f7 = f5 / c_liver
    print("f1={}, f2={}, f5={}, f6={}, f7={}".format(f1, f2, f5, f6, f7))
    A = np.array([[1, 1 + 1 / b_tissue], [1, 1 + 1 / b_liver]])
    B = np.array([f_circ_lac + g7, (1 + 1 / b_liver) * f_circ_lac - f7])
    print(A, B)
    [y1, y2] = np.linalg.solve(A, B)
    print("f4={}, g3={}".format(y1, y2))
    g3 = y2
    f4 = y1
    g4 = f_circ_lac - f4
    f3 = f_circ_lac - g3
    g8 = g3 / b_tissue
    f8 = f3 / b_liver
    g9 = g5 - g6 - (g8 - g7)
    f9 = f7 - f8 - (f6 - f5)
    return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, g1, g2, g3, g4, g5, g6, g7, g8, g9])


# Model with more circulatory metabolite (pyruvate) in Rabinowitz' data
def solve_one_case_model5(
        a_liver, a_tissue, b_liver, b_tissue, c_liver, c_tissue, f_circ_gluc, f_circ_lac,
        x1, x2, f_infuse_gluc, print=lambda *x: None):
    print("a_liver={}, a_tissue={}, b_liver={}, b_tissue={}, c_liver={}, c_tissue={}".format(
        a_liver, a_tissue, b_liver, b_tissue, c_liver, c_tissue))
    f1 = x1
    g2 = x2
    g1 = f_circ_gluc + f_infuse_gluc - f1
    g6 = g1 / a_tissue
    g5 = g6 + g1 - g2
    g7 = g5 / c_tissue
    print("g1={}, g2={}, g5={}, g6={}, g7={}".format(g1, g2, g5, g6, g7))

    f10 = f_infuse_gluc
    f2 = f_circ_gluc - g2
    f6 = f1 / a_liver
    f5 = f6 + f1 - f2
    f7 = f5 / c_liver
    print("f1={}, f2={}, f5={}, f6={}, f7={}".format(f1, f2, f5, f6, f7))
    A = np.array([[1, 1 + 1 / b_tissue], [1, 1 + 1 / b_liver]])
    B = np.array([f_circ_lac + g7, (1 + 1 / b_liver) * f_circ_lac - f7])
    print(A, B)
    [y1, y2] = np.linalg.solve(A, B)
    print("f4={}, g3={}".format(y1, y2))
    g3 = y2
    f4 = y1
    g4 = f_circ_lac - f4
    f3 = f_circ_lac - g3
    g8 = g3 / b_tissue
    f8 = f3 / b_liver
    g9 = g5 - g6 - (g8 - g7)
    f9 = f7 - f8 - (f6 - f5)
    return np.array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, g1, g2, g3, g4, g5, g6, g7, g8, g9])


def solve_dist(
        x_array, f1_ratio_dist, f3_ratio_dist, f5_ratio_dist, g1_dist, g2_dist, sample_size,
        solve_one_case):
    f1_ratio_array = f1_ratio_dist.rvs(sample_size)
    a_ratio_array = f1_ratio_array / (1 - f1_ratio_array)
    f3_ratio_array = f3_ratio_dist.rvs(sample_size)
    b_ratio_array = f3_ratio_array / (1 - f3_ratio_array)
    f5_ratio_array = f5_ratio_dist.rvs(sample_size)
    c_ratio_array = f5_ratio_array / (1 - f5_ratio_array)
    g1_array = g1_dist.rvs(sample_size)
    g2_array = g2_dist.rvs(sample_size)

    # f_result = [x_array.reshape([1, -1]), [], [], [], [], [], [], []]
    f_result = None
    for x in x_array:
        f_list = solve_one_case(a_ratio_array, b_ratio_array, c_ratio_array, g1_array, g2_array, x)
        if f_result is None:
            f_result = [[] for _ in range(len(f_list))]
        for index, value in enumerate(f_list):
            f_result[index].append(value)
    for index, value in enumerate(f_result):
        f_result[index] = np.array(value).transpose()
    return f_result


def plot_dist(f_result, x_value_index, leave_out_set):
    def calculate_confidential(var_matrix, percentage):
        lower_bound = np.percentile(var_matrix, 50 - 0.5 * percentage, axis=0)
        upper_bound = np.percentile(var_matrix, 50 + 0.5 * percentage, axis=0)
        mean_value = var_matrix.mean(axis=0)
        return mean_value, lower_bound, upper_bound

    def plot_ci(_ax, x, y, upper, lower, label):
        _ax.plot(x, y, label=label)
        _ax.fill_between(x, lower, upper, alpha=color_set.alpha_value)

    fig, ax = plt.subplots(ncols=1, nrows=1)
    percent = 95
    x_array = f_result[x_value_index].reshape([-1])
    for index, flux_value_matrix in enumerate(f_result):
        if index == x_value_index or index + 1 in leave_out_set:
            continue
        y_array, lower_array, upper_array = calculate_confidential(flux_value_matrix, percent)
        plot_ci(ax, x_array, y_array, lower_array, upper_array, label="f{}".format(index + 1))
    ax.legend()
    plt.show()


def solve_all_fluxes(a, b, c, g1, g2, solve_one_case):
    item_num = 100
    f1_array = np.linspace(g1, 1000, item_num)
    f2_array = np.zeros(item_num)
    f3_array = np.zeros(item_num)
    f4_array = np.zeros(item_num)
    f5_array = np.zeros(item_num)
    f6_array = np.zeros(item_num)
    f7_array = np.zeros(item_num)
    f8_array = np.zeros(item_num)
    for index, x in enumerate(f1_array):
        (
            _, f2_array[index], f3_array[index], f4_array[index], f5_array[index],
            f6_array[index], f7_array[index], f8_array[index], *_) = solve_one_case(a, b, c, g1, g2, x)
    fig, ax = plt.subplots(ncols=1, nrows=1)
    ax.plot(f1_array, f2_array, label="f2")
    # ax.plot(f1_array, f3_array, label="f3")
    # ax.plot(f1_array, f4_array, label="f4")
    ax.plot(f1_array, f5_array, label="f5")
    ax.plot(f1_array, f6_array, label="f6")
    # ax.plot(f1_array, f7_array, label="f7")
    # ax.plot(f1_array, f8_array, label="f8")
    ax.legend()
    plt.show()


def solve_distribution_model1(data_collection, label_list):
    f1_dist, f3_dist, f5_dist = solve_mid_distribution(data_collection.mid_data, label_list, dist_or_mean="dist")
    g1_dist = NormalDist(650, 70)
    g2_dist = NormalDist(400, 40)
    sample_size = 100
    solve_one_case = solve_one_case_model1
    x_array = np.linspace(700, 1000, 100)
    f_result = solve_dist(
        x_array, f1_dist, f3_dist, f5_dist, g1_dist, g2_dist, sample_size, solve_one_case)
    leave_out_set = {3, 4, 7, 8}
    plot_dist(f_result, 0, leave_out_set)


def solve_distribution_model2(data_collection, label_list):
    f1_f6_ratio_dist, f3_f8_ratio_dist, f5_f7_ratio_dist = solve_mid_distribution(
        data_collection.mid_data, label_list, dist_or_mean="dist")
    f1_dist = NormalDist(150.9, 46.7)
    f3_dist = NormalDist(374.4, 112.4)
    sample_size = 100
    solve_one_case = solve_one_case_model2
    x_array = np.linspace(0, 150, 100)
    f_result = solve_dist(
        x_array, f1_f6_ratio_dist, f3_f8_ratio_dist, f5_f7_ratio_dist, f1_dist, f3_dist, sample_size,
        solve_one_case)
    plot_dist(f_result, 9, {})


def solve_single_result_model2(data_collection, label_list):
    f1_f6_ratio, f3_f8_ratio, f5_f7_ratio = solve_mid_distribution(
        data_collection.mid_data, label_list, dist_or_mean="mean")
    f1 = 150.9
    f3 = 374.4
    a = f1_f6_ratio / (1 - f1_f6_ratio)
    b = f3_f8_ratio / (1 - f3_f8_ratio)
    c = f5_f7_ratio / (1 - f5_f7_ratio)
    solve_one_case = solve_one_case_model2
    x_value = 150
    f_result = solve_one_case(a, b, c, f1, f3, x_value)
    print(f_result)


def solve_net_glucose_lactate_flux(flux_vector):
    glucose_flux = 0
    lactate_flux = 0
    f56 = flux_vector[4] - flux_vector[5]
    f78 = flux_vector[6] - flux_vector[7]
    f_flux_num = 10
    g56 = flux_vector[f_flux_num + 4] - flux_vector[f_flux_num + 5]
    g78 = flux_vector[f_flux_num + 6] - flux_vector[f_flux_num + 7]
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


def model3_calculator(
        _a_liver, _a_tissue, _b_liver, _b_tissue, _c_liver, _c_tissue, _f_circ_gluc, _f_circ_lac, _f_input,
        _x1_range, _x2_range):
    x1_num = len(_x1_range)
    x2_num = len(_x2_range)
    _cycle_flux_matrix = np.zeros([x1_num, x2_num])
    min_cycle_flux = max_cycle_flux = 0
    min_glucose_contri = max_glucose_contri = 0.5
    _valid_matrix = np.zeros([x1_num, x2_num])
    _glucose_contri_matrix = np.zeros([x1_num, x2_num])
    for index1, x1 in enumerate(_x1_range):
        for index2, x2 in enumerate(_x2_range):
            f_result = solve_one_case_model3(
                _a_liver, _a_tissue, _b_liver, _b_tissue, _c_liver, _c_tissue, _f_circ_gluc, _f_circ_lac, x1, x2,
                _f_input)
            if np.all(f_result > -1e-5):
                _valid_matrix[index1, index2] = 1
                cycle_flux = ((f_result[1] - f_result[0]) + (f_result[2] - f_result[3])) / 2
                _cycle_flux_matrix[index1, index2] = cycle_flux
                if cycle_flux < min_cycle_flux:
                    min_cycle_flux = cycle_flux
                    min_cycle_solution = [x1, x2]
                elif cycle_flux > max_cycle_flux:
                    max_cycle_flux = cycle_flux
                    max_cycle_solution = [x1, x2]
                glucose_contri = solve_net_glucose_lactate_flux(f_result)
                _glucose_contri_matrix[index1, index2] = glucose_contri
                if glucose_contri < min_glucose_contri:
                    min_glucose_contri = glucose_contri
                    min_glucose_solution = [x1, x2]
                elif glucose_contri > max_glucose_contri:
                    max_glucose_contri = glucose_contri
                    max_glucose_solution = [x1, x2]
                # print(x1, x2)
            else:
                _cycle_flux_matrix[index1, index2] = np.nan
                _glucose_contri_matrix[index1, index2] = np.nan
    return _valid_matrix, _glucose_contri_matrix, _cycle_flux_matrix


def solve_net_contribution_fluxes_model3(data_collection, label_list):
    brain_marker = 'Br'
    heart_marker = 'Ht'
    muscle_marker = 'SkM'
    liver_marker = 'Lv'
    tissue_marker = heart_marker

    dynamic_range_heatmap = True
    contribution_heatmap = True
    contribution_histgram = False
    cycle_flux_heatmap = False
    contribution_violin_plot = True

    f1_f6_ratio_liver, f3_f8_ratio_liver, f5_f7_ratio_liver = solve_mid_distribution(
        data_collection.mid_data, label_list, dist_or_mean="mean", tissue_marker=liver_marker)
    f1_f6_ratio_tissue, f3_f8_ratio_tissue, f5_f7_ratio_tissue = solve_mid_distribution(
        data_collection.mid_data, label_list, dist_or_mean="mean", tissue_marker=tissue_marker)
    a_liver = f1_f6_ratio_liver / (1 - f1_f6_ratio_liver)
    b_liver = f3_f8_ratio_liver / (1 - f3_f8_ratio_liver)
    c_liver = f5_f7_ratio_liver / (1 - f5_f7_ratio_liver)
    a_tissue = f1_f6_ratio_tissue / (1 - f1_f6_ratio_tissue)
    b_tissue = f3_f8_ratio_tissue / (1 - f3_f8_ratio_tissue)
    c_tissue = f5_f7_ratio_tissue / (1 - f5_f7_ratio_tissue)
    f_circ_gluc = 150.9
    f_circ_lac = 374.4
    f_input = 100
    x1_num = 1000
    x1_interv = 100
    x1_limit = [0, 150]
    x1_range = np.linspace(*x1_limit, x1_num + 1)
    x2_num = 1000
    x2_interv = 100
    x2_limit = [0, 150]
    x2_range = np.linspace(*x2_limit, x2_num + 1)

    valid_matrix, glucose_contri_matrix, cycle_flux_matrix = model3_calculator(
        a_liver, a_tissue, b_liver, b_tissue, c_liver, c_tissue, f_circ_gluc, f_circ_lac, f_input,
        x1_range, x2_range)

    with open("./Figures/data/valid_matrix_{}_{}".format(liver_marker, tissue_marker), 'wb') as f_out:
        pickle.dump(valid_matrix, f_out)
    with open("./Figures/data/glucose_contri_matrix_{}_{}".format(liver_marker, tissue_marker), 'wb') as f_out:
        pickle.dump(glucose_contri_matrix, f_out)

    if dynamic_range_heatmap:
        plot_size = (20, 7)
        dpi = 150
        fig, ax = plt.subplots(figsize=plot_size, dpi=dpi)
        im = ax.imshow(valid_matrix)
        ax.set_xlim([0, x2_num])
        ax.set_ylim([0, x1_num])
        # x_tick = ax.get_xticks()
        # y_tick = ax.get_yticks()
        # x_tick_in_range = x_tick[x_tick <= x2_num]
        # y_tick_in_range = y_tick[y_tick <= x1_num]
        x_tick_in_range = np.arange(0, x2_num, x2_interv)
        y_tick_in_range = np.arange(0, x1_num, x1_interv)
        ax.set_xticks(x_tick_in_range)
        ax.set_yticks(y_tick_in_range)
        ax.set_xticklabels(np.around(x2_range[np.array(x_tick_in_range, dtype='int')]))
        ax.set_yticklabels(np.around(x1_range[np.array(y_tick_in_range, dtype='int')]))
        fig.savefig("./Figures/model3/dynamic_range_{}_{}_100_hf.png".format(liver_marker, tissue_marker), dpi=fig.dpi)

    if cycle_flux_heatmap:
        fig, ax = plt.subplots()
        im = ax.imshow(cycle_flux_matrix, cmap='cool')
        ax.set_xlim([0, x2_num])
        ax.set_ylim([0, x1_num])
        x_tick = ax.get_xticks()
        y_tick = ax.get_yticks()
        ax.set_xticks(x_tick)
        ax.set_yticks(y_tick)
        x_tick_label = np.around(x2_range[np.array(x_tick, dtype='int')])
        y_tick_label = np.around(x1_range[np.array(y_tick, dtype='int')])
        ax.set_xticklabels(x_tick_label)
        ax.set_yticklabels(y_tick_label)
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Cycle flux value', rotation=-90, va="bottom")

    if contribution_histgram:
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

    if contribution_heatmap:
        fig, ax = plt.subplots(figsize=plot_size, dpi=dpi)
        im = ax.imshow(glucose_contri_matrix, cmap='cool')
        ax.set_xlim([0, x2_num])
        ax.set_ylim([0, x1_num])
        # x_tick = ax.get_xticks()
        # y_tick = ax.get_yticks()
        # ax.set_xticks(x_tick)
        # ax.set_yticks(y_tick)
        # x_tick_in_range = x_tick[x_tick <= x2_num]
        # y_tick_in_range = y_tick[y_tick <= x1_num]
        x_tick_in_range = np.arange(0, x2_num, x2_interv)
        y_tick_in_range = np.arange(0, x1_num, x1_interv)
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

    if contribution_violin_plot:
        glucose_contri_vector = glucose_contri_matrix.reshape([-1])
        glucose_contri_vector = glucose_contri_vector[~np.isnan(glucose_contri_vector)]
        contribution_variation_dict = {
            "normal": glucose_contri_vector,
        }
        fig, ax = violin_plot(contribution_variation_dict)
        # fig.savefig("./Figures/model4/glucose_contribution_violinplot_normal_{}_{}.png".format(
        #     liver_marker, tissue_marker))

    plt.show()
    print()
    # print("Min cycle flux: {} at point {}".format(min_cycle_flux, min_cycle_solution))
    # print("Max cycle flux: {} at point {}".format(max_cycle_flux, max_cycle_solution))
    # print("Min glucose contribution: {} at point {}".format(min_glucose_contri, min_glucose_solution))
    # print("Max glucose contribution: {} at point {}".format(max_glucose_contri, max_glucose_solution))


def glucose_contribution_violin_model3(data_collection, label_list):
    new_data = False
    contribution_data_file_path = "./Figures/data/glucose_contribution_data_model_3_finput_100"

    brain_marker = 'Br'
    heart_marker = 'Ht'
    muscle_marker = 'SkM'
    kidney_marker = 'Kd'
    lung_marker = 'Lg'
    pancreas_marker = 'Pc'
    intestine_marker = 'SI'
    spleen_marker = 'Sp'
    liver_marker = 'Lv'
    tissue_marker_list = [
        heart_marker, brain_marker, muscle_marker, kidney_marker, lung_marker, pancreas_marker,
        intestine_marker, spleen_marker]

    if new_data:
        final_data_dict = {}
        for tissue_marker in tissue_marker_list:
            f1_f6_ratio_liver, f3_f8_ratio_liver, f5_f7_ratio_liver = solve_mid_distribution(
                data_collection.mid_data, label_list, dist_or_mean="mean", tissue_marker=liver_marker)
            f1_f6_ratio_tissue, f3_f8_ratio_tissue, f5_f7_ratio_tissue = solve_mid_distribution(
                data_collection.mid_data, label_list, dist_or_mean="mean", tissue_marker=tissue_marker)
            a_liver = f1_f6_ratio_liver / (1 - f1_f6_ratio_liver)
            b_liver = f3_f8_ratio_liver / (1 - f3_f8_ratio_liver)
            c_liver = f5_f7_ratio_liver / (1 - f5_f7_ratio_liver)
            a_tissue = f1_f6_ratio_tissue / (1 - f1_f6_ratio_tissue)
            b_tissue = f3_f8_ratio_tissue / (1 - f3_f8_ratio_tissue)
            c_tissue = f5_f7_ratio_tissue / (1 - f5_f7_ratio_tissue)
            f_circ_gluc = 150.9
            f_circ_lac = 374.4
            f_input = 100
            x1_num = 1000
            x1_interv = 100
            x1_limit = [0, 150]
            x1_range = np.linspace(*x1_limit, x1_num + 1)
            x2_num = 1000
            x2_interv = 300
            x2_limit = [0, 150]
            x2_range = np.linspace(*x2_limit, x2_num + 1)

            valid_matrix, glucose_contri_matrix, cycle_flux_matrix = model3_calculator(
                a_liver, a_tissue, b_liver, b_tissue, c_liver, c_tissue, f_circ_gluc, f_circ_lac, f_input,
                x1_range, x2_range)
            sample = glucose_contri_matrix.reshape([-1])
            sample = sample[~np.isnan(sample)]
            final_data_dict[tissue_marker] = sample

        with open(contribution_data_file_path, 'wb') as f_out:
            pickle.dump(final_data_dict, f_out)

    else:
        with open(contribution_data_file_path, 'rb') as f_in:
            final_data_dict = pickle.load(f_in)

    violin_plot(final_data_dict)


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
            pc.set_alpha(color_set.alpha_value)
    dash_color = color_set.orange
    ax.axhline(0.5, linestyle='--', color=dash_color)
    ax.set_ylim([-0.1, 1.1])
    ax.set_xticks(x_axis_position)
    ax.set_xticklabels(tissue_label_list)
    # plt.show()
    return fig, ax


compare_color_dict = {
    'low': color_set.purple,
    'original': color_set.blue,
    'high': color_set.orange
}


def violin_test():
    data_dict = {
        'low': [0, 1, 0.2, 0.1, 0.3, 0.2, 0.4],
        'original': [0, 1, 0.4, 0.5, 0.5, 0.5, 0.5],
        'high': [0, 1, 0.7, 0.8, 0.9, 0.8, 0.7]
    }
    violin_plot(data_dict, compare_color_dict)
    plt.show()


def variation_analysis_model3(data_collection, label_list):
    brain_marker = 'Br'
    heart_marker = 'Ht'
    muscle_marker = 'SkM'
    liver_marker = 'Lv'
    tissue_marker = heart_marker

    dynamic_range_heatmap = False
    contribution_plot = False
    contribution_histgram = False
    contribution_violin_plot = True

    x1_num = 1000
    x1_interv = 100
    x1_limit = [0, 200]
    x1_range = np.linspace(*x1_limit, x1_num + 1)
    x2_num = 1000
    x2_interv = 100
    x2_limit = [0, 200]
    x2_range = np.linspace(*x2_limit, x2_num + 1)

    f1_f6_ratio_liver, f3_f8_ratio_liver, f5_f7_ratio_liver = solve_mid_distribution(
        data_collection.mid_data, label_list, dist_or_mean="mean", tissue_marker=liver_marker)
    f1_f6_ratio_tissue, f3_f8_ratio_tissue, f5_f7_ratio_tissue = solve_mid_distribution(
        data_collection.mid_data, label_list, dist_or_mean="mean", tissue_marker=tissue_marker)
    a_liver = f1_f6_ratio_liver / (1 - f1_f6_ratio_liver)
    b_liver = f3_f8_ratio_liver / (1 - f3_f8_ratio_liver)
    c_liver = f5_f7_ratio_liver / (1 - f5_f7_ratio_liver)
    a_tissue = f1_f6_ratio_tissue / (1 - f1_f6_ratio_tissue)
    b_tissue = f3_f8_ratio_tissue / (1 - f3_f8_ratio_tissue)
    c_tissue = f5_f7_ratio_tissue / (1 - f5_f7_ratio_tissue)
    f_circ_gluc = 150.9
    f_circ_lac = 374.4
    f_input = 100

    variation_ratio = 0.25
    param_dict = {
        '_a_liver': a_liver, '_a_tissue': a_tissue,
        '_b_liver': b_liver, '_b_tissue': b_tissue,
        '_c_liver': c_liver, '_c_tissue': c_tissue,
        '_f_circ_gluc': f_circ_gluc, '_f_circ_lac': f_circ_lac,
        '_f_input': f_input}

    normal_valid_matrix, normal_glucose_contri_matrix, _ = model3_calculator(
        **param_dict, _x1_range=x1_range, _x2_range=x2_range)
    normal_glucose_contri_vector = normal_glucose_contri_matrix.reshape([-1])
    normal_glucose_contri_vector = normal_glucose_contri_vector[~np.isnan(normal_glucose_contri_vector)]
    more_glucose_ratio_normal = np.count_nonzero(normal_glucose_contri_vector > 0.5) / len(normal_glucose_contri_vector)
    print("More glucose ratio in normal case: {}".format(more_glucose_ratio_normal))

    for var_name in param_dict.keys():
        # for var_name in ['_a_liver']:
        high_param_dict = dict(param_dict)
        high_param_dict[var_name] *= (1 + variation_ratio)
        high_valid_matrix, high_glucose_contri_matrix, _ = model3_calculator(
            **high_param_dict, _x1_range=x1_range, _x2_range=x2_range)
        high_glucose_contri_vector = high_glucose_contri_matrix.reshape([-1])
        high_glucose_contri_vector = high_glucose_contri_vector[~np.isnan(high_glucose_contri_vector)]
        more_glucose_ratio_high = np.count_nonzero(high_glucose_contri_vector > 0.5) / len(
            high_glucose_contri_vector)
        print("More glucose ratio in high case with var {}: {}".format(var_name, more_glucose_ratio_high))

        low_param_dict = dict(param_dict)
        low_param_dict[var_name] *= (1 - variation_ratio)
        low_valid_matrix, low_glucose_contri_matrix, _ = model3_calculator(
            **low_param_dict, _x1_range=x1_range, _x2_range=x2_range)
        low_glucose_contri_vector = low_glucose_contri_matrix.reshape([-1])
        low_glucose_contri_vector = low_glucose_contri_vector[~np.isnan(low_glucose_contri_vector)]
        more_glucose_ratio_low = np.count_nonzero(low_glucose_contri_vector > 0.5) / len(
            low_glucose_contri_vector)
        print("More glucose ratio in low case with var {}: {}".format(var_name, more_glucose_ratio_low))
        high_valid_matrix[normal_valid_matrix == 1.0] = 3
        high_valid_matrix[low_valid_matrix == 1.0] = 2

        if dynamic_range_heatmap:
            plot_size = (20, 7)
            dpi = 150
            fig, ax = plt.subplots(figsize=plot_size, dpi=dpi)
            im = ax.imshow(high_valid_matrix)
            ax.set_xlim([0, x2_num])
            ax.set_ylim([0, x1_num])
            x_tick_in_range = np.arange(0, x2_num + 1, x2_interv)
            y_tick_in_range = np.arange(0, x1_num + 1, x1_interv)
            ax.set_xticks(x_tick_in_range)
            ax.set_yticks(y_tick_in_range)
            ax.set_xticklabels(np.around(x2_range[np.array(x_tick_in_range, dtype='int')]))
            ax.set_yticklabels(np.around(x1_range[np.array(y_tick_in_range, dtype='int')]))
            fig.savefig("./Figures/model3/dynamic_range_variation_{}_{}_{}.png".format(
                liver_marker, tissue_marker, var_name), dpi=fig.dpi)

        if contribution_histgram:
            fig, ax = plt.subplots()
            bin_num = 200
            n_normal, bins_normal, _ = ax.hist(
                normal_glucose_contri_vector, bins=bin_num, density=True, label="original",
                color=(0.992, 0.906, 0.141), alpha=color_set.alpha_value)
            n_high, bins_high, _ = ax.hist(
                high_glucose_contri_vector, bins=bin_num, density=True, label="high",
                color=(0.188, 0.404, 0.553), alpha=color_set.alpha_value)
            n_low, bins_low, _ = ax.hist(
                low_glucose_contri_vector, bins=bin_num, density=True, label="low",
                color=(0.208, 0.718, 0.471), alpha=color_set.alpha_value)
            ax.set_xlim([0, 1])
            # ax.legend()
            fig.savefig("./Figures/model3/glucose_contribution_variation_{}_{}_{}.png".format(
                liver_marker, tissue_marker, var_name))

            if contribution_plot:
                fig, ax = plt.subplots()
                x_normal = (bins_normal[1:] + bins_normal[:-1]) / 2
                x_high = (bins_high[1:] + bins_high[:-1]) / 2
                x_low = (bins_low[1:] + bins_low[:-1]) / 2
                ax.plot(x_normal, n_normal, color=(0.992, 0.906, 0.141), linewidth=2)
                ax.plot(x_high, n_high, color=(0.188, 0.404, 0.553), linewidth=2)
                ax.plot(x_low, n_low, color=(0.208, 0.718, 0.471), linewidth=2)
                fig.savefig("./Figures/model3/glucose_contribution_plot_variation_{}_{}_{}.png".format(
                    liver_marker, tissue_marker, var_name))

        if contribution_violin_plot:
            contribution_variation_dict = {
                "low": low_glucose_contri_vector,
                "original": normal_glucose_contri_vector,
                "high": high_glucose_contri_vector
            }
            fig, ax = violin_plot(contribution_variation_dict, compare_color_dict)
            fig.savefig("./Figures/model3/glucose_contribution_variation_violinplot_{}_{}_{}.png".format(
                liver_marker, tissue_marker, var_name))


def model4_calculator(
        a_liver, a_tissue, b_liver, b_tissue, c_liver, c_tissue, f_circ_gluc, f_circ_lac,
        x1_range, x2_range, f_infuse_gluc):
    x1_num = len(x1_range)
    x2_num = len(x2_range)

    cycle_flux_matrix = np.zeros([x1_num, x2_num])
    min_cycle_flux = max_cycle_flux = 0
    min_cycle_solution = []
    max_cycle_solution = []
    min_glucose_contri = max_glucose_contri = 0.5
    min_glucose_solution = []
    max_glucose_solution = []

    cycle_flux_matrix = np.zeros([x1_num, x2_num])
    glucose_contri_matrix = np.zeros([x1_num, x2_num])
    valid_matrix = np.zeros([x1_num, x2_num])
    for index1, x1 in enumerate(x1_range):
        for index2, x2 in enumerate(x2_range):
            f_result = solve_one_case_model4(
                a_liver, a_tissue, b_liver, b_tissue, c_liver, c_tissue, f_circ_gluc, f_circ_lac,
                x1, x2, f_infuse_gluc)
            if np.all(f_result > -1e-5):
                valid_matrix[index1, index2] = 1
                cycle_flux = ((f_result[1] - f_result[0]) + (f_result[2] - f_result[3])) / 2
                cycle_flux_matrix[index1, index2] = cycle_flux
                if cycle_flux < min_cycle_flux:
                    min_cycle_flux = cycle_flux
                    min_cycle_solution = [x1, x2]
                elif cycle_flux > max_cycle_flux:
                    max_cycle_flux = cycle_flux
                    max_cycle_solution = [x1, x2]
                glucose_contri = solve_net_glucose_lactate_flux(f_result)
                glucose_contri_matrix[index1, index2] = glucose_contri
                if glucose_contri < min_glucose_contri:
                    min_glucose_contri = glucose_contri
                    min_glucose_solution = [x1, x2]
                elif glucose_contri > max_glucose_contri:
                    max_glucose_contri = glucose_contri
                    max_glucose_solution = [x1, x2]
                # print(x1, x2)
            else:
                valid_matrix[index1, index2] = 0
                cycle_flux_matrix[index1, index2] = np.nan
                glucose_contri_matrix[index1, index2] = np.nan

    return valid_matrix, glucose_contri_matrix, cycle_flux_matrix


def solve_net_contribution_fluxes_model4(data_collection, label_list):
    # heart_marker = 'Ht'
    muscle_marker = 'SkM'
    liver_marker = 'Lv'
    tissue_marker = muscle_marker

    dynamic_range_heatmap = True
    contribution_heatmap = True
    contribution_histgram = False
    contribution_violin_plot = True

    f1_f6_ratio_liver, f3_f8_ratio_liver, f5_f7_ratio_liver = solve_mid_distribution(
        data_collection.mid_data, label_list, dist_or_mean="mean", tissue_marker=liver_marker)
    f1_f6_ratio_tissue, f3_f8_ratio_tissue, f5_f7_ratio_tissue = solve_mid_distribution(
        data_collection.mid_data, label_list, dist_or_mean="mean", tissue_marker=tissue_marker)
    f_infuse_gluc = 111.1
    infuse_gluc_ratio, _ = solve_infuse_ratio(data_collection.mid_data, [liver_marker, muscle_marker])
    f_circ_gluc = f_infuse_gluc / infuse_gluc_ratio - f_infuse_gluc
    a_liver = f1_f6_ratio_liver / (1 - f1_f6_ratio_liver)
    b_liver = f3_f8_ratio_liver / (1 - f3_f8_ratio_liver)
    c_liver = f5_f7_ratio_liver / (1 - f5_f7_ratio_liver)
    a_tissue = f1_f6_ratio_tissue / (1 - f1_f6_ratio_tissue)
    b_tissue = f3_f8_ratio_tissue / (1 - f3_f8_ratio_tissue)
    c_tissue = f5_f7_ratio_tissue / (1 - f5_f7_ratio_tissue)
    f_circ_lac = 500
    x1_num = 500
    x1_limit = [0, 50]
    x1_range = np.linspace(*x1_limit, x1_num + 1)
    x2_num = 800
    x2_limit = [120, 210]
    x2_range = np.linspace(*x2_limit, x2_num + 1)

    valid_matrix, glucose_contri_matrix, cycle_flux_matrix = model4_calculator(
        a_liver, a_tissue, b_liver, b_tissue, c_liver, c_tissue, f_circ_gluc, f_circ_lac,
        x1_range, x2_range, f_infuse_gluc)

    if dynamic_range_heatmap:
        fig, ax = plt.subplots()
        im = ax.imshow(valid_matrix)
        ax.set_xlim([0, x2_num])
        ax.set_ylim([0, x1_num])
        x_tick = ax.get_xticks()
        y_tick = ax.get_yticks()
        ax.set_xticklabels(np.around(x2_range[np.array(x_tick, dtype='int')]))
        ax.set_yticklabels(np.around(x1_range[np.array(y_tick, dtype='int')]))

    if contribution_heatmap:
        fig, ax = plt.subplots()
        im = ax.imshow(glucose_contri_matrix, cmap='cool')
        ax.set_xlim([0, x2_num])
        ax.set_ylim([0, x1_num])
        x_tick = ax.get_xticks()
        y_tick = ax.get_yticks()
        ax.set_xticks(x_tick)
        ax.set_yticks(y_tick)
        x_tick_label = np.around(x2_range[np.array(x_tick, dtype='int')])
        y_tick_label = np.around(x1_range[np.array(y_tick, dtype='int')])
        ax.set_xticklabels(x_tick_label)
        ax.set_yticklabels(y_tick_label)
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Glucose Contribution', rotation=-90, va="bottom")

    if contribution_histgram:
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

    if contribution_violin_plot:
        glucose_contri_vector = glucose_contri_matrix.reshape([-1])
        glucose_contri_vector = glucose_contri_vector[~np.isnan(glucose_contri_vector)]
        if len(glucose_contri_vector) == 0:
            raise ValueError('No point fit the constraint for contribution of carbon sources!')
        contribution_variation_dict = {
            "normal": glucose_contri_vector,
        }
        fig, ax = violin_plot(contribution_variation_dict)
        # fig.savefig("./Figures/model4/glucose_contribution_violinplot_normal_{}_{}.png".format(
        #     liver_marker, tissue_marker))

    plt.show()
    # print("Min cycle flux: {} at point {}".format(min_cycle_flux, min_cycle_solution))
    # print("Max cycle flux: {} at point {}".format(max_cycle_flux, max_cycle_solution))
    # print("Min glucose contribution: {} at point {}".format(min_glucose_contri, min_glucose_solution))
    # print("Max glucose contribution: {} at point {}".format(max_glucose_contri, max_glucose_solution))


def variation_analysis_model4(data_collection, label_list):
    brain_marker = 'Br'
    heart_marker = 'Ht'
    muscle_marker = 'SkM'
    liver_marker = 'Lv'
    tissue_marker = muscle_marker

    dynamic_range_heatmap = True
    contribution_plot = False
    contribution_histgram = False
    contribution_violin_plot = True

    f1_f6_ratio_liver, f3_f8_ratio_liver, f5_f7_ratio_liver = solve_mid_distribution(
        data_collection.mid_data, label_list, dist_or_mean="mean", tissue_marker=liver_marker)
    f1_f6_ratio_tissue, f3_f8_ratio_tissue, f5_f7_ratio_tissue = solve_mid_distribution(
        data_collection.mid_data, label_list, dist_or_mean="mean", tissue_marker=tissue_marker)
    f_infuse_gluc = 111.1
    infuse_gluc_ratio, _ = solve_infuse_ratio(data_collection.mid_data, [liver_marker, muscle_marker])
    f_circ_gluc = f_infuse_gluc / infuse_gluc_ratio - f_infuse_gluc
    a_liver = f1_f6_ratio_liver / (1 - f1_f6_ratio_liver)
    b_liver = f3_f8_ratio_liver / (1 - f3_f8_ratio_liver)
    c_liver = f5_f7_ratio_liver / (1 - f5_f7_ratio_liver)
    a_tissue = f1_f6_ratio_tissue / (1 - f1_f6_ratio_tissue)
    b_tissue = f3_f8_ratio_tissue / (1 - f3_f8_ratio_tissue)
    c_tissue = f5_f7_ratio_tissue / (1 - f5_f7_ratio_tissue)
    f_circ_lac = 500
    x1_num = 1000
    x1_interv = 100
    x1_limit = [0, 100]
    x1_range = np.linspace(*x1_limit, x1_num + 1)
    x2_num = 1500
    x2_interv = 300
    x2_limit = [60, 210]
    x2_range = np.linspace(*x2_limit, x2_num + 1)

    variation_ratio = 0.15
    param_dict = {
        'a_liver': a_liver, 'a_tissue': a_tissue,
        'b_liver': b_liver, 'b_tissue': b_tissue,
        'c_liver': c_liver, 'c_tissue': c_tissue,
        'f_circ_gluc': f_circ_gluc, 'f_circ_lac': f_circ_lac,
        'f_infuse_gluc': f_infuse_gluc}

    normal_valid_matrix, normal_glucose_contri_matrix, _ = model4_calculator(
        **param_dict, x1_range=x1_range, x2_range=x2_range)
    normal_glucose_contri_vector = normal_glucose_contri_matrix.reshape([-1])
    normal_glucose_contri_vector = normal_glucose_contri_vector[~np.isnan(normal_glucose_contri_vector)]
    more_glucose_ratio_normal = np.count_nonzero(normal_glucose_contri_vector > 0.5) / len(normal_glucose_contri_vector)
    print("More glucose ratio in normal case: {}".format(more_glucose_ratio_normal))

    # for var_name in param_dict.keys():
    for var_name in ['f_infuse_gluc']:
        high_param_dict = dict(param_dict)
        high_param_dict[var_name] *= (1 + variation_ratio)
        high_valid_matrix, high_glucose_contri_matrix, _ = model4_calculator(
            **high_param_dict, x1_range=x1_range, x2_range=x2_range)
        high_glucose_contri_vector = high_glucose_contri_matrix.reshape([-1])
        high_glucose_contri_vector = high_glucose_contri_vector[~np.isnan(high_glucose_contri_vector)]
        more_glucose_ratio_high = np.count_nonzero(high_glucose_contri_vector > 0.5) / len(
            high_glucose_contri_vector)
        print("More glucose ratio in high case with var {}: {}".format(var_name, more_glucose_ratio_high))

        low_param_dict = dict(param_dict)
        low_param_dict[var_name] *= (1 - variation_ratio)
        low_valid_matrix, low_glucose_contri_matrix, _ = model4_calculator(
            **low_param_dict, x1_range=x1_range, x2_range=x2_range)
        low_glucose_contri_vector = low_glucose_contri_matrix.reshape([-1])
        low_glucose_contri_vector = low_glucose_contri_vector[~np.isnan(low_glucose_contri_vector)]
        more_glucose_ratio_low = np.count_nonzero(low_glucose_contri_vector > 0.5) / len(
            low_glucose_contri_vector)
        print("More glucose ratio in low case with var {}: {}".format(var_name, more_glucose_ratio_low))
        high_valid_matrix[normal_valid_matrix == 1.0] = 3
        high_valid_matrix[low_valid_matrix == 1.0] = 2

        if dynamic_range_heatmap:
            high_valid_matrix[normal_valid_matrix == 1.0] = 3
            high_valid_matrix[low_valid_matrix == 1.0] = 2

            fig, ax = plt.subplots()
            im = ax.imshow(high_valid_matrix)
            ax.set_xlim([0, x2_num])
            ax.set_ylim([0, x1_num])
            x_tick_in_range = np.arange(0, x2_num + 1, x2_interv)
            y_tick_in_range = np.arange(0, x1_num + 1, x1_interv)
            ax.set_xticklabels(x_tick_in_range)
            ax.set_yticklabels(y_tick_in_range)
            ax.set_xticklabels(np.around(x2_range[np.array(x_tick_in_range, dtype='int')]))
            ax.set_yticklabels(np.around(x1_range[np.array(y_tick_in_range, dtype='int')]))

        if contribution_violin_plot:
            contribution_variation_dict = {
                "low": low_glucose_contri_vector,
                "original": normal_glucose_contri_vector,
                "high": high_glucose_contri_vector
            }
            fig, ax = violin_plot(contribution_variation_dict, compare_color_dict)
            fig.savefig("./Figures/model4/glucose_contribution_variation_violinplot_{}_{}_{}.png".format(
                liver_marker, tissue_marker, var_name))
    plt.show()


def raw_data_plotting(data_collection, label_list):
    def mean_std(np_array):
        return np.mean(np_array, axis=0), np.std(np_array, axis=0)

    heart_marker = 'Ht'
    muscle_marker = 'SkM'
    liver_marker = 'Lv'
    serum_marker = 'Sr'
    tissue_marker = muscle_marker
    label = label_list[0]
    tissue_marker_list = [tissue_marker, serum_marker, liver_marker]
    metabolite_list = ['glucose', 'pyruvate', 'lactate']

    data_dict = data_collection.mid_data[label]
    collected_data_dict = {}
    final_average_dict = {}
    final_std_dict = {}
    for dict_each_mouse in data_dict.values():
        for tissue_name in tissue_marker_list:
            for metabolite_name in metabolite_list:
                try:
                    current_data = dict_each_mouse[tissue_name][metabolite_name]
                except KeyError:
                    continue
                else:
                    current_key = "{}_{}".format(tissue_name, metabolite_name)
                    try:
                        collected_data_dict[current_key].append(current_data)
                    except KeyError:
                        collected_data_dict[current_key] = [current_data]
    for mixed_key, mid_array in collected_data_dict.items():
        final_average_dict[mixed_key], final_std_dict[mixed_key] = mean_std(mid_array)

    base_color = color_set.blue
    base_alpha_value = color_set.alpha_value + 0.1
    for mixed_key, mid_array in final_average_dict.items():
        edge = 0.2
        array_len = len(mid_array)
        fig_size = (array_len + edge * 2, 4)
        fig, ax = plt.subplots(figsize=fig_size)
        x_loc = np.arange(array_len) + 0.5
        ax.bar(x_loc, mid_array, width=0.6, color=base_color, alpha=color_set.alpha_for_bar_plot)
        ax.errorbar(
            x_loc, mid_array, yerr=final_std_dict[mixed_key], capsize=5, fmt='none',
            color=base_color)
        ax.set_xlabel(mixed_key)
        ax.set_ylim([0, 1])
        ax.set_xlim(-edge, array_len + edge)
        ax.set_xticks(x_loc)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.set_yticklabels([])
        fig.savefig("./Figures/data/raw_data_{}_{}.png".format(data_collection.experiment_name, mixed_key))
    plt.show()


def main():
    # file_path = "data_collection.xlsx"
    # experiment_name_prefix = "Sup_Fig_5_fasted"
    # label_list = ["glucose", "lactate"]
    file_path = "data_collection_from_Dan.xlsx"
    experiment_name_prefix = "no_tumor"
    label_list = ["glucose"]
    data_collection = data_parser.data_parser(file_path, experiment_name_prefix, label_list)
    data_collection = data_parser.data_checker(
        data_collection, ["glucose", "lactate"], ["glucose", "pyruvate", "lactate"])
    # solve_distribution_model2(data_collection, label_list)
    # solve_single_result_model2(data_collection, label_list)
    # solve_single_result_model3(data_collection, label_list)
    # solve_single_result_model4(data_collection, label_list)
    # solve_parameter_sensitivity_model3(data_collection, label_list)
    solve_net_contribution_fluxes_model3(data_collection, label_list)
    # glucose_contribution_violin_model3(data_collection, label_list)
    # solve_net_contribution_fluxes_model4(data_collection, label_list)
    # solve_param_sensitivity(data_collection, label_list, solve_one_case_model3)
    variation_analysis_model3(data_collection, label_list)
    # variation_analysis_model4(data_collection, label_list)
    # raw_data_plotting(data_collection, label_list)


if __name__ == '__main__':
    main()
