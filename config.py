import platform
import numpy as np

if platform.node() == 'BaranLiu-PC':
    test_running = True
else:
    test_running = False


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
