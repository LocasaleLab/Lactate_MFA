import warnings

import numpy as np
from scipy.misc import comb
from scipy.stats import t
from scipy.stats import rv_discrete
import matplotlib.pyplot as plt

import data_parser as data_parser


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
    k = (source1 - source2).reshape([-1, 1])
    b = target - source2
    result = np.linalg.lstsq(k, b)
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
    k_stderr = k.std()
    coeff_stderr = residual_stderr / k_stderr

    segment_num = 1000
    lb = 0.1
    ub = 0.9
    x_linespace = np.linspace(lb, ub, segment_num)
    x_cdf_linespace = t.cdf(x_linespace, df=df, loc=coeff, scale=coeff_stderr)
    x_prob = (x_cdf_linespace[1:] - x_cdf_linespace[:-1]) / (x_cdf_linespace[-1] - x_cdf_linespace[0])

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
    c13_ratio = np.power(source_mid[0], (1 / carbon_num))

    final_output_vector = natural_dist(c13_ratio, target_carbon_num)
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


def solve_mid_distribution(data_collect_dict, dist_or_mean="dist", tissue_marker='Lv'):
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

    label_list = ["glucose", "lactate"]
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
        _ax.fill_between(x, lower, upper, alpha=0.4)

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


def old_main():
    data_file_name = "data_collection.xlsx"
    data_collect_dict = data_parser.data_loader(data_file_name, "Sheet2")
    # print(data_collect_dict)
    # construct_model_primary(data_collect_dict)
    a, b, c = solve_single_mid(data_collect_dict)
    g1 = 650
    sigma_g1 = 70
    g2 = 400
    sigma_g2 = 40
    x_max = 1000
    # solve_all_fluxes(a, b, c, g1, g2)
    solve_one_case = solve_one_case_model1
    print(solve_one_case(a, b, c, g1, g2, x_max))


def solve_distribution_model1(data_collection):
    f1_dist, f3_dist, f5_dist = solve_mid_distribution(data_collection.mid_data, dist_or_mean="dist")
    g1_dist = NormalDist(650, 70)
    g2_dist = NormalDist(400, 40)
    sample_size = 100
    solve_one_case = solve_one_case_model1
    x_array = np.linspace(700, 1000, 100)
    f_result = solve_dist(
        x_array, f1_dist, f3_dist, f5_dist, g1_dist, g2_dist, sample_size, solve_one_case)
    leave_out_set = {3, 4, 7, 8}
    plot_dist(f_result, 0, leave_out_set)


def solve_distribution_model2(data_collection):
    f1_f6_ratio_dist, f3_f8_ratio_dist, f5_f7_ratio_dist = solve_mid_distribution(
        data_collection.mid_data, dist_or_mean="dist")
    f1_dist = NormalDist(150.9, 46.7)
    f3_dist = NormalDist(374.4, 112.4)
    sample_size = 100
    solve_one_case = solve_one_case_model2
    x_array = np.linspace(0, 150, 100)
    f_result = solve_dist(
        x_array, f1_f6_ratio_dist, f3_f8_ratio_dist, f5_f7_ratio_dist, f1_dist, f3_dist, sample_size,
        solve_one_case)
    plot_dist(f_result, 9, {})


def solve_single_result_model2(data_collection):
    f1_f6_ratio, f3_f8_ratio, f5_f7_ratio = solve_mid_distribution(
        data_collection.mid_data, dist_or_mean="mean")
    f1 = 150.9
    f3 = 374.4
    a = f1_f6_ratio / (1 - f1_f6_ratio)
    b = f3_f8_ratio / (1 - f3_f8_ratio)
    c = f5_f7_ratio / (1 - f5_f7_ratio)
    solve_one_case = solve_one_case_model2
    x_value = 150
    f_result = solve_one_case(a, b, c, f1, f3, x_value)
    print(f_result)


def solve_single_result_model3(data_collection):
    brain_marker = 'Br'
    heart_marker = 'Ht'
    muscle_marker = 'SkM'
    liver_marker = 'Lv'
    tissue_marker = heart_marker

    f1_f6_ratio_liver, f3_f8_ratio_liver, f5_f7_ratio_liver = solve_mid_distribution(
        data_collection.mid_data, dist_or_mean="mean", tissue_marker=liver_marker)
    f1_f6_ratio_tissue, f3_f8_ratio_tissue, f5_f7_ratio_tissue = solve_mid_distribution(
        data_collection.mid_data, dist_or_mean="mean", tissue_marker=tissue_marker)
    a_liver = f1_f6_ratio_liver / (1 - f1_f6_ratio_liver)
    b_liver = f3_f8_ratio_liver / (1 - f3_f8_ratio_liver)
    c_liver = f5_f7_ratio_liver / (1 - f5_f7_ratio_liver)
    a_tissue = f1_f6_ratio_tissue / (1 - f1_f6_ratio_tissue)
    b_tissue = f3_f8_ratio_tissue / (1 - f3_f8_ratio_tissue)
    c_tissue = f5_f7_ratio_tissue / (1 - f5_f7_ratio_tissue)
    f_circ_gluc = 150.9
    f_circ_lac = 374.4
    f_10 = 100
    x1 = 37
    x2 = 94.5
    f_result = solve_one_case_model3(
        a_liver, a_tissue, b_liver, b_tissue, c_liver, c_tissue, f_circ_gluc, f_circ_lac, x1, x2, f_10, print)
    print(" ".join(
        ["f{}={{min_result[{}]}}".format(i + 1, i) for i in range(10)] +
        ["g{}={{min_result[{}]}}".format(i + 1, i + 10) for i in range(9)]).format(min_result=f_result))

    # min_result = None
    # min_min_value = -np.inf
    # min_x = None
    # for x1 in range(151):
    #     for x2 in range(151):
    #         f_result = solve_one_case_model3(
    #             a_liver, a_tissue, b_liver, b_tissue, c_liver, c_tissue, f_circ_gluc, f_circ_lac, x1, x2, f_10)
    #         if np.min(f_result) > min_min_value:
    #             min_result = f_result
    #             min_min_value = np.min(f_result)
    #             min_x = [x1, x2]
    # print(" ".join(
    #     ["f{}={{min_result[{}]}}".format(i + 1, i) for i in range(10)] +
    #     ["g{}={{min_result[{}]}}".format(i + 1, i + 10) for i in range(9)]).format(min_result=list(min_result)))
    # print(min_x)


def solve_dynamic_range_model3(data_collection):
    brain_marker = 'Br'
    heart_marker = 'Ht'
    muscle_marker = 'SkM'
    liver_marker = 'Lv'
    tissue_marker = heart_marker

    f1_f6_ratio_liver, f3_f8_ratio_liver, f5_f7_ratio_liver = solve_mid_distribution(
        data_collection.mid_data, dist_or_mean="mean", tissue_marker=liver_marker)
    f1_f6_ratio_tissue, f3_f8_ratio_tissue, f5_f7_ratio_tissue = solve_mid_distribution(
        data_collection.mid_data, dist_or_mean="mean", tissue_marker=tissue_marker)
    a_liver = f1_f6_ratio_liver / (1 - f1_f6_ratio_liver)
    b_liver = f3_f8_ratio_liver / (1 - f3_f8_ratio_liver)
    c_liver = f5_f7_ratio_liver / (1 - f5_f7_ratio_liver)
    a_tissue = f1_f6_ratio_tissue / (1 - f1_f6_ratio_tissue)
    b_tissue = f3_f8_ratio_tissue / (1 - f3_f8_ratio_tissue)
    c_tissue = f5_f7_ratio_tissue / (1 - f5_f7_ratio_tissue)
    f_circ_gluc = 150.9
    f_circ_lac = 374.4
    f_10 = 100
    x1_num = 400
    x1_limit = 180
    x1_range = np.linspace(0, x1_limit, x1_num + 1)
    x2_num = x1_num
    x2_limit = x1_limit
    x2_range = np.linspace(0, x2_limit, x2_num + 1)
    valid_matrix = np.zeros([x1_num + 1, x2_num + 1])
    for index1, x1 in enumerate(x1_range):
        for index2, x2 in enumerate(x2_range):
            f_result = solve_one_case_model3(
                a_liver, a_tissue, b_liver, b_tissue, c_liver, c_tissue, f_circ_gluc, f_circ_lac, x1, x2, f_10)
            if np.all(f_result > -1e-5):
                valid_matrix[index1, index2] = 1
                # print(x1, x2)
            else:
                valid_matrix[index1, index2] = 0
    fig, ax = plt.subplots()
    im = ax.imshow(valid_matrix)
    ax.set_xlim([0, x1_num])
    ax.set_ylim([0, x2_num])
    x_tick = ax.get_xticks()
    y_tick = ax.get_yticks()
    ax.set_xticklabels(np.around(x1_range[np.array(x_tick, dtype='int')]))
    ax.set_yticklabels(np.around(x2_range[np.array(y_tick, dtype='int')]))
    plt.show()


def solve_param_sensitivity(data_collection, solve_one_case):
    f1, f3, f5 = solve_mid_distribution(data_collection.mid_data, dist_or_mean="mean")
    g1 = 650
    g2 = 400
    variation_ratio = 0.5

    x = 800
    a = f1 / (1 - f1)
    b = f3 / (1 - f3)
    c = f5 / (1 - f5)
    param_dict = {'a': a, 'b': b, 'c': c, 'g1': g1, 'g2': g2, 'x': x}
    mean_flux_vector = np.array(solve_one_case(**param_dict))
    flux_num = len(mean_flux_vector)
    y_lim = [0.9, 1.1]
    x_ticks = list(range(flux_num))
    x_ticklabel = ['f{}'.format(i + 2) for i in range(flux_num)]
    for var_name, _ in param_dict.items():
        if var_name == 'x':
            continue
        fig, ax = plt.subplots(ncols=1, nrows=1)
        upper_param_dict = dict(param_dict)
        lower_param_dict = dict(param_dict)
        lower_param_dict[var_name] *= (1 + variation_ratio)
        upper_param_dict[var_name] /= (1 + variation_ratio)
        lower_flux_vector = np.array(solve_one_case(**lower_param_dict)) / mean_flux_vector
        upper_flux_vector = np.array(solve_one_case(**upper_param_dict)) / mean_flux_vector
        up_low_limit = np.zeros([2, flux_num])
        for index, (i, j) in enumerate(zip(lower_flux_vector, upper_flux_vector)):
            if i > j:
                up_low_limit[0, index] = j
                up_low_limit[1, index] = i
            else:
                up_low_limit[0, index] = i
                up_low_limit[1, index] = j
        relative_error = np.abs(up_low_limit - 1)
        ax.errorbar(
            range(flux_num), np.ones(flux_num),
            yerr=relative_error, marker='o', linestyle='', capsize=7)
        ax.set_ylim(y_lim)
        ax.set_title(var_name)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticklabel)
        plt.show()


def solve_parameter_sensitivity_model3(data_collection):
    brain_marker = 'Br'
    heart_marker = 'Ht'
    muscle_marker = 'SkM'
    liver_marker = 'Lv'
    tissue_marker = heart_marker

    f1_f6_ratio_liver, f3_f8_ratio_liver, f5_f7_ratio_liver = solve_mid_distribution(
        data_collection.mid_data, dist_or_mean="mean", tissue_marker=liver_marker)
    f1_f6_ratio_tissue, f3_f8_ratio_tissue, f5_f7_ratio_tissue = solve_mid_distribution(
        data_collection.mid_data, dist_or_mean="mean", tissue_marker=tissue_marker)
    a_liver = f1_f6_ratio_liver / (1 - f1_f6_ratio_liver)
    b_liver = f3_f8_ratio_liver / (1 - f3_f8_ratio_liver)
    c_liver = f5_f7_ratio_liver / (1 - f5_f7_ratio_liver)
    a_tissue = f1_f6_ratio_tissue / (1 - f1_f6_ratio_tissue)
    b_tissue = f3_f8_ratio_tissue / (1 - f3_f8_ratio_tissue)
    c_tissue = f5_f7_ratio_tissue / (1 - f5_f7_ratio_tissue)
    f_circ_gluc = 150.9
    f_circ_lac = 374.4
    f_10 = 100
    x1_num = 400
    x1_limit = 180
    x1_range = np.linspace(0, x1_limit, x1_num + 1)
    x2_num = x1_num
    x2_limit = x1_limit
    x2_range = np.linspace(0, x2_limit, x2_num + 1)
    valid_matrix = np.zeros([x1_num + 1, x2_num + 1])

    min_result = None
    min_min_value = -np.inf
    min_x = None
    for index1, x1 in enumerate(x1_range):
        for index2, x2 in enumerate(x2_range):
            f_result = solve_one_case_model3(
                a_liver, a_tissue, b_liver, b_tissue, c_liver, c_tissue, f_circ_gluc, f_circ_lac, x1, x2, f_10)
            if np.min(f_result) > min_min_value:
                min_result = f_result
                min_min_value = np.min(f_result)
                min_x = [x1, x2]

    delta_percent = 0.01
    min_x_dict = {'x1': min_x[0], 'x2': min_x[1], 'f_input': f_10}
    for key in min_x_dict.keys():
        delta_h = min_x_dict[key] * delta_percent
        x_high_dict = dict(min_x_dict)
        x_high_dict[key] += delta_h
        f_high = solve_one_case_model3(
            a_liver, a_tissue, b_liver, b_tissue, c_liver, c_tissue, f_circ_gluc, f_circ_lac, **x_high_dict)
        x_low_dict = dict(min_x_dict)
        x_low_dict[key] -= delta_h
        f_low = solve_one_case_model3(
            a_liver, a_tissue, b_liver, b_tissue, c_liver, c_tissue, f_circ_gluc, f_circ_lac, **x_low_dict)
        derivative = (f_high - f_low) / (2 * delta_h)
        relative_derivative = derivative / min_result
        fig, ax = plt.subplots()
        x = range(len(derivative))
        ax.bar(x, relative_derivative)
        x_label = ["f{}".format(i + 1, i) for i in range(10)] + ["g{}".format(i + 1, i + 10) for i in range(9)]
        ax.set_xticks(x)
        ax.set_xticklabels(x_label)
        ax.set_title(key)
        plt.show()


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


def solve_net_contribution_fluxes(data_collection):
    brain_marker = 'Br'
    heart_marker = 'Ht'
    muscle_marker = 'SkM'
    liver_marker = 'Lv'
    tissue_marker = heart_marker

    f1_f6_ratio_liver, f3_f8_ratio_liver, f5_f7_ratio_liver = solve_mid_distribution(
        data_collection.mid_data, dist_or_mean="mean", tissue_marker=liver_marker)
    f1_f6_ratio_tissue, f3_f8_ratio_tissue, f5_f7_ratio_tissue = solve_mid_distribution(
        data_collection.mid_data, dist_or_mean="mean", tissue_marker=tissue_marker)
    a_liver = f1_f6_ratio_liver / (1 - f1_f6_ratio_liver)
    b_liver = f3_f8_ratio_liver / (1 - f3_f8_ratio_liver)
    c_liver = f5_f7_ratio_liver / (1 - f5_f7_ratio_liver)
    a_tissue = f1_f6_ratio_tissue / (1 - f1_f6_ratio_tissue)
    b_tissue = f3_f8_ratio_tissue / (1 - f3_f8_ratio_tissue)
    c_tissue = f5_f7_ratio_tissue / (1 - f5_f7_ratio_tissue)
    f_circ_gluc = 150.9
    f_circ_lac = 374.4
    f_10 = 100
    x1_num = 100
    x1_limit = 50
    x1_range = np.linspace(0, x1_limit, x1_num + 1)
    x2_num = 300
    x2_limit = 150
    x2_range = np.linspace(0, x2_limit, x2_num + 1)
    cycle_flux_matrix = np.zeros([x1_num + 1, x2_num + 1])
    min_cycle_flux = max_cycle_flux = 0
    min_cycle_solution = []
    max_cycle_solution = []
    min_glucose_contri = max_glucose_contri = 0.5
    min_glucose_solution = []
    max_glucose_solution = []
    glucose_contri_matrix = np.zeros([x1_num + 1, x2_num + 1])
    for index1, x1 in enumerate(x1_range):
        for index2, x2 in enumerate(x2_range):
            f_result = solve_one_case_model3(
                a_liver, a_tissue, b_liver, b_tissue, c_liver, c_tissue, f_circ_gluc, f_circ_lac, x1, x2, f_10)
            if np.all(f_result > -1e-5):
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
                cycle_flux_matrix[index1, index2] = np.nan
                glucose_contri_matrix[index1, index2] = np.nan
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
    cbar.ax.set_ylabel('Contribution of glucose', rotation=-90, va="bottom")

    plt.show()
    print("Min cycle flux: {} at point {}".format(min_cycle_flux, min_cycle_solution))
    print("Max cycle flux: {} at point {}".format(max_cycle_flux, max_cycle_solution))
    print("Min glucose contribution: {} at point {}".format(min_glucose_contri, min_glucose_solution))
    print("Max glucose contribution: {} at point {}".format(max_glucose_contri, max_glucose_solution))


def main():
    file_path = "data_collection.xlsx"
    experiment_name_prefix = "Sup_Fig_5_fasted"
    label_list = ["glucose", "lactate"]
    data_collection = data_parser.data_parser(file_path, experiment_name_prefix, label_list)
    data_collection = data_parser.data_checker(
        data_collection, ["glucose", "lactate"], ["glucose", "pyruvate", "lactate"])
    # solve_distribution_model2(data_collection)
    # solve_single_result_model2(data_collection)
    solve_single_result_model3(data_collection)
    # solve_dynamic_range_model3(data_collection)
    # solve_parameter_sensitivity_model3(data_collection)
    # solve_net_contribution_fluxes(data_collection)
    # solve_param_sensitivity(data_collection)


if __name__ == '__main__':
    main()
