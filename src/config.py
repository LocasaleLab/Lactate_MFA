import platform
import numpy as np

if platform.node() == 'BaranLiu-PC':
    test_running = True
else:
    test_running = False


class Result(object):
    def __init__(
            self, result_dict: dict, obj_value: float, success: bool, minimal_obj_value: float, label: dict):
        self.result_dict = result_dict
        self.obj_value = obj_value
        self.success = success
        self.minimal_obj_value = minimal_obj_value
        self.label = label

    def __repr__(self):
        return "Result: {}\nObjective value: {}\nSuccess: {}\nMinimal objective value: {}".format(
            self.result_dict, self.obj_value, self.success, self.minimal_obj_value)


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


class Color(object):
    blue = np.array([21, 113, 177]) / 255
    orange = np.array([251, 138, 68]) / 255
    purple = np.array([112, 48, 160]) / 255
    light_blue = np.array([221, 241, 255]) / 255

    alpha_value = 0.3
    alpha_for_bar_plot = alpha_value + 0.1


class Constants(object):
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
