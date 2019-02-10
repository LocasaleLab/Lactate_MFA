import numpy as np
import cvxopt

from new_model_main import result_evaluation

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


def solve_flux_model_cvxopt(
        balance_matrix, balance_right_side, mid_matrix, mid_right_side, min_flux_value=1, max_flux_value=10000):
    var_num = balance_matrix.shape[1]
    cvx_matrix = cvxopt.matrix
    raw_matrix_a = mid_matrix
    raw_vector_b = mid_right_side.reshape([-1, 1])
    matrix_p = cvx_matrix(raw_matrix_a.T @ raw_matrix_a)
    vector_q = cvx_matrix(- raw_matrix_a.T @ raw_vector_b)

    matrix_g = cvx_matrix(np.vstack([-1 * np.identity(var_num), np.identity(var_num)]))
    matrix_h = cvx_matrix(np.vstack([-min_flux_value * np.ones([var_num, 1]), max_flux_value * np.ones([var_num, 1])]))
    matrix_a = cvx_matrix(balance_matrix)
    matrix_b = cvx_matrix(balance_right_side.reshape([-1, 1]))

    result = cvxopt.solvers.qp(matrix_p, vector_q, matrix_g, matrix_h, matrix_a, matrix_b)
    result_array = np.array(result['x'])
    print("Result: {}".format(result_array))
    obj_value = result['primal objective'] * 2 + raw_vector_b.T @ raw_vector_b
    print("Objective function: {}".format(obj_value))
    return result_array.reshape([-1])


def sentence_recognition_cvxopt(
        balance_list: list, mid_constraint_list: list, var_dict: dict, constant_flux_dict: dict):
    var_num = len(var_dict)
    balance_array_list = []
    balance_right_side_list = []
    for balance_dict in balance_list:
        constant = 0
        new_balance_array = np.zeros(var_num)
        flux_name_list = balance_dict['input'] + balance_dict['output']
        value_list = [-1 for _ in balance_dict['input']] + [1 for _ in balance_dict['output']]
        for flux_name, value in zip(flux_name_list, value_list):
            try:
                flux_index = var_dict[flux_name]
            except KeyError:
                constant -= value * constant_flux_dict[flux_name]
            else:
                new_balance_array[flux_index] = value
        balance_array_list.append(new_balance_array)
        balance_right_side_list.append(constant)

    mid_array_list = []
    mid_right_side_list = []
    for mid_constraint_dict in mid_constraint_list:
        target_vector = mid_constraint_dict[target_label]
        vector_dim = len(target_vector)
        new_mid_array_list = [np.zeros(var_num) for _ in range(vector_dim)]
        constant_array = np.zeros(vector_dim)
        for flux_name, vector in mid_constraint_dict.items():
            if flux_name == target_label:
                continue
            else:
                normalized_vector = vector - target_vector
            try:
                flux_index = var_dict[flux_name]
            except KeyError:
                constant_array -= constant_flux_dict[flux_name] * normalized_vector
            else:
                for index, vector_value in enumerate(normalized_vector):
                    new_mid_array_list[index][flux_index] = vector_value
        mid_array_list.extend(new_mid_array_list)
        mid_right_side_list.extend(constant_array)

    balance_matrix = np.array(balance_array_list)
    balance_right_side = np.array(balance_right_side_list)
    mid_matrix = np.array(mid_array_list)
    mid_right_side = np.array(mid_right_side_list)
    return balance_matrix, balance_right_side, mid_matrix, mid_right_side


def model_construction_test():
    var_list = ['F1', 'F2', 'F3', 'F4', 'Foutput']
    input_mid_vector = np.array([1.0, 0])
    node1_mid_vector = np.array([0.9, 0.1])
    node2_mid_vector = np.array([0.7, 0.3])
    node3_mid_vector = np.array([0.1, 0.9])
    var_dict = {var: index for index, var in enumerate(var_list)}
    constant_flux_dict = {'Finput': 100}
    node1_balance_eq = {'input': ['F2', 'Finput'], 'output': ['F1']}
    node1_mid_eq = {'Finput': input_mid_vector, 'F2': node2_mid_vector, target_label: node1_mid_vector}
    node2_balance_eq = {'input': ['F1', 'F4'], 'output': ['F2', 'F3']}
    node2_mid_eq = {'F1': node1_mid_vector, 'F4': node3_mid_vector, target_label: node2_mid_vector}
    node3_balance_eq = {'input': ['F3'], 'output': ['F4', 'Foutput']}
    balance_list = [node1_balance_eq, node2_balance_eq, node3_balance_eq]
    mid_constraint_list = [node1_mid_eq, node2_mid_eq]
    balance_matrix, balance_right_side, mid_matrix, mid_right_side = sentence_recognition_cvxopt(
        balance_list, mid_constraint_list, var_dict, constant_flux_dict)
    print(balance_matrix)
    print(balance_right_side)
    print(mid_matrix)
    print(mid_right_side)
    float_type = 'float64'
    result = solve_flux_model_cvxopt(
        balance_matrix.astype(dtype=float_type), balance_right_side.astype(dtype=float_type),
        mid_matrix.astype(dtype=float_type), mid_right_side.astype(dtype=float_type))
    print(result)


def model_solver_cvxopt(
        data_collection_obj, general_setting_func, general_setting_kwargs,
        data_collection_func, data_collection_kwargs, model_construction_func, print_result_func):
    var_dict, constant_flux_dict = general_setting_func(**general_setting_kwargs)
    raw_data_collection_dict = data_collection_obj.mid_data
    model_mid_data_dict = data_collection_func(raw_data_collection_dict, **data_collection_kwargs)
    balance_list, mid_constraint_list = model_construction_func(model_mid_data_dict)
    balance_matrix, balance_right_side, mid_matrix, mid_right_side = sentence_recognition_cvxopt(
        balance_list, mid_constraint_list, var_dict, constant_flux_dict)
    float_type = 'float64'
    result = solve_flux_model_cvxopt(
        balance_matrix.astype(dtype=float_type), balance_right_side.astype(dtype=float_type),
        mid_matrix.astype(dtype=float_type), mid_right_side.astype(dtype=float_type))
    result_dict = {flux_name: flux_value for flux_name, flux_value in zip(var_dict.keys(), result)}
    print_result_func(result_dict, constant_flux_dict)
    result_evaluation(result_dict, constant_flux_dict, mid_constraint_list)
    return result
