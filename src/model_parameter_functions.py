from src import model_specific_functions, config

constant_set = config.Constants()
color_set = config.Color()

# Data loader functions
data_loader_rabinowitz = model_specific_functions.data_loader_rabinowitz
data_loader_dan = model_specific_functions.data_loader_dan
mid_data_loader_model1234 = model_specific_functions.mid_data_loader_model1234
mid_data_loader_model5 = model_specific_functions.mid_data_loader_model5
mid_data_loader_all_tissue = model_specific_functions.mid_data_loader_all_tissue

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
# final_processing_dynamic_range_model5 = model_specific_functions.final_processing_dynamic_range_model5
# final_processing_all_tissue_model12 = model_specific_functions.final_processing_all_tissue_model12
# final_processing_all_tissue_model34 = model_specific_functions.final_processing_all_tissue_model34
final_processing_parameter_sensitivity_model1 = model_specific_functions.final_processing_parameter_sensitivity_model1

# Preparation functions
dynamic_range_model12 = model_specific_functions.dynamic_range_model12
dynamic_range_model345 = model_specific_functions.dynamic_range_model345
all_tissue_model1 = model_specific_functions.all_tissue_model1
all_tissue_model3 = model_specific_functions.all_tissue_model3
all_tissue_hypoxia_correction = model_specific_functions.all_tissue_hypoxia_correction
parameter_sensitivity_model1 = model_specific_functions.parameter_sensitivity_model1


def model1_parameters(test=False):
    model_name = "model1"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

    data_collection_kwargs = {
        'label_list': ["glucose"], 'mouse_id_list': ['M1'],
        'source_tissue_marker': constant_set.liver_marker, 'sink_tissue_marker': constant_set.heart_marker}
    data_loader_func = data_loader_rabinowitz
    data_collection_func = mid_data_loader_model1234
    # model_mid_data_dict = data_loader_rabinowitz(mid_data_loader_model1234, data_collection_kwargs)

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


# def model2_parameters(test=False):
#     model_name = "model2"
#     output_direct = "{}/{}".format(constant_set.output_direct, model_name)
#
#     data_collection_kwargs = {
#         'label_list': ["glucose"], 'mouse_id_list': ['M1'],
#         'source_tissue_marker': constant_set.liver_marker, 'sink_tissue_marker': constant_set.muscle_marker}
#     data_loader_func = data_loader_dan
#     data_collection_func = mid_data_loader_model1234
#     # model_mid_data_dict = data_loader_dan(mid_data_loader_model1234, data_collection_kwargs)
#
#     hook_in_each_iteration = metabolite_contribution_model12
#     hook_after_all_iterations = final_processing_dynamic_range_model12
#     model_construction_func = model2_construction
#     parameter_construction_func = dynamic_range_model12
#
#     complete_flux_list = ['F{}'.format(i + 1) for i in range(9)] + ['G{}'.format(i + 1) for i in range(9)] + \
#                          ['Jin', 'Fcirc_lac']
#     complete_flux_dict = {var: i for i, var in enumerate(complete_flux_list)}
#     constant_flux_dict = {'Jin': 111.1, 'Fcirc_lac': 500}
#
#     min_flux_value = 1
#     max_flux_value = 2000
#     max_free_flux_value = 250
#     optimization_repeat_time = 10
#     obj_tolerance = 0.25
#     f1_range = [min_flux_value, max_free_flux_value]
#     g2_range = [min_flux_value, max_free_flux_value]
#
#     if test:
#         f1_num = 101
#         f1_display_interv = 100
#         g2_num = 101
#         g2_display_interv = 100
#     else:
#         f1_num = 1500
#         f1_display_interv = 250
#         g2_num = 1500
#         g2_display_interv = 250
#
#     return locals()


def model3_parameters(test=False):
    model_name = "model3"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

    data_collection_kwargs = {
        'label_list': ["glucose"], 'mouse_id_list': ['M1'],
        'source_tissue_marker': constant_set.liver_marker, 'sink_tissue_marker': constant_set.heart_marker}
    data_loader_func = data_loader_rabinowitz
    data_collection_func = mid_data_loader_model1234

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


# def model4_parameters(test=False):
#     model_name = "model4"
#     output_direct = "{}/{}".format(constant_set.output_direct, model_name)
#
#     data_collection_kwargs = {
#         'label_list': ["glucose"], 'mouse_id_list': ['M1'],
#         'source_tissue_marker': constant_set.liver_marker, 'sink_tissue_marker': constant_set.muscle_marker}
#     data_loader_func = data_loader_dan
#     data_collection_func = mid_data_loader_model1234
#     # model_mid_data_dict = data_loader_dan(mid_data_loader_model1234, data_collection_kwargs)
#
#     hook_in_each_iteration = metabolite_contribution_model34
#     hook_after_all_iterations = final_processing_dynamic_range_model345
#     model_construction_func = model4_construction
#     parameter_construction_func = dynamic_range_model345
#
#     complete_flux_list = ['F{}'.format(i + 1) for i in range(11)] + ['G{}'.format(i + 1) for i in range(11)] + \
#                          ['H{}'.format(i + 1) for i in range(3)] + ['Fcirc_lac', 'Fcirc_pyr', 'Jin']
#     complete_flux_dict = {var: i for i, var in enumerate(complete_flux_list)}
#     constant_flux_dict = {'Jin': 111.1, 'Fcirc_lac': 500, 'Fcirc_pyr': 100}
#     fcirc_glc_max = 250
#
#     min_flux_value = 1
#     max_flux_value = 2000
#     optimization_repeat_time = 10
#     obj_tolerance = 0.25
#     ternary_sigma = 0.15
#
#     free_fluxes_name_list = ['F1', 'G2', 'F9', 'G10', 'F3']
#     free_fluxes_range_list = [
#         [min_flux_value, fcirc_glc_max],
#         [min_flux_value, fcirc_glc_max],
#         [min_flux_value, constant_flux_dict['Fcirc_pyr']],
#         [min_flux_value, constant_flux_dict['Fcirc_pyr']],
#         [min_flux_value, constant_flux_dict['Fcirc_lac']],
#     ]
#     if test:
#         total_point_num = int(3e3)
#         # point_interval_list = [50, 50, 20, 20, 100]
#         ternary_resolution = int(2 ** 7)
#     else:
#         total_point_num = int(3e6)
#         # point_interval_list = [25, 25, 5, 5, 25]
#         ternary_resolution = int(2 ** 8)
#
#     return locals()


def model5_parameters(test=False):
    model_name = "model5"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

    data_collection_kwargs = {
        'label_list': ["glucose"], 'mouse_id_list': ['M1'],
        'source_tissue_marker': constant_set.liver_marker, 'sink1_tissue_marker': constant_set.heart_marker,
        'sink2_tissue_marker': constant_set.muscle_marker}
    # data_collection_kwargs = {
    #     'label_list': ["glucose"], 'mouse_id_list': ['M1'],
    #     'source_tissue_marker': constant_set.liver_marker, 'sink1_tissue_marker': constant_set.brain_marker,
    #     'sink2_tissue_marker': constant_set.muscle_marker}
    data_loader_func = data_loader_rabinowitz
    data_collection_func = mid_data_loader_model5
    # model_mid_data_dict = data_loader_rabinowitz(mid_data_loader_model5, data_collection_kwargs)

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
    model_name = "model5_comb2"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

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
    model_name = "model5_comb3"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

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


def model6_parameters(test=False):
    model_name = "model6"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

    data_collection_kwargs = {
        'label_list': ["glucose"], 'mouse_id_list': ['M1'],
        'source_tissue_marker': constant_set.liver_marker, 'sink_tissue_marker': constant_set.muscle_marker}
    data_loader_func = data_loader_dan
    data_collection_func = mid_data_loader_model1234

    hook_in_each_iteration = metabolite_contribution_model12
    hook_after_all_iterations = final_processing_dynamic_range_model12
    model_construction_func = model6_construction
    parameter_construction_func = dynamic_range_model12

    complete_flux_list = ['F{}'.format(i + 1) for i in range(10)] + ['G{}'.format(i + 1) for i in range(9)] + \
                         ['Jin', 'Fcirc_lac']
    complete_flux_dict = {var: i for i, var in enumerate(complete_flux_list)}
    constant_flux_dict = {'Jin': 111.1, 'F10': 80, 'Fcirc_lac': 400}

    min_flux_value = 1
    max_flux_value = 1000  # 8000
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


def model7_parameters(test=False):
    model_name = "model7"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

    data_collection_kwargs = {
        'label_list': ["glucose"], 'mouse_id_list': ['M1'],
        'source_tissue_marker': constant_set.liver_marker, 'sink_tissue_marker': constant_set.muscle_marker}
    data_loader_func = data_loader_dan
    data_collection_func = mid_data_loader_model1234
    # model_mid_data_dict = data_loader_dan(mid_data_loader_model1234, data_collection_kwargs)

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
    max_flux_value = 2000
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


def model1_all_tissue(test=False):
    model_name = "model1_all_tissue"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

    model1_parameter_dict = model1_parameters(test)
    data_collection_kwargs = {
        'label_list': ["glucose"], 'mouse_id_list': ['M1'],
        'source_tissue_marker': constant_set.liver_marker,
        'sink_tissue_marker_list': [
            constant_set.heart_marker, constant_set.brain_marker, constant_set.muscle_marker,
            constant_set.kidney_marker, constant_set.lung_marker, constant_set.pancreas_marker,
            constant_set.intestine_marker, constant_set.spleen_marker]}
    data_collection_func = mid_data_loader_all_tissue
    parameter_construction_func = all_tissue_model1

    model1_parameter_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'data_collection_kwargs': data_collection_kwargs,
        'data_collection_func': data_collection_func,
        'parameter_construction_func': parameter_construction_func,
    })

    return model1_parameter_dict


def model1_all_tissue_old(test=False):
    model_name = "model1_all_tissue"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

    data_collection_kwargs = {
        'label_list': ["glucose"], 'mouse_id_list': ['M1'],
        'source_tissue_marker': constant_set.liver_marker,
        'sink_tissue_marker_list': [
            constant_set.heart_marker, constant_set.brain_marker, constant_set.muscle_marker,
            constant_set.kidney_marker, constant_set.lung_marker, constant_set.pancreas_marker,
            constant_set.intestine_marker, constant_set.spleen_marker]}
    data_loader_func = data_loader_rabinowitz
    data_collection_func = mid_data_loader_all_tissue

    hook_in_each_iteration = metabolite_contribution_model12
    # hook_after_all_iterations = final_processing_all_tissue_model12
    hook_after_all_iterations = final_processing_dynamic_range_model12
    model_construction_func = model1_construction
    parameter_construction_func = all_tissue_model1

    complete_flux_list = ['F{}'.format(i + 1) for i in range(10)] + ['G{}'.format(i + 1) for i in range(9)] + \
                         ['Fcirc_glc', 'Fcirc_lac']
    complete_flux_dict = {var: i for i, var in enumerate(complete_flux_list)}
    constant_flux_dict = {'Fcirc_glc': 150.9, 'Fcirc_lac': 374.4, 'F10': 40}

    min_flux_value = 1
    max_flux_value = 1000
    # max_flux_value = 500
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

    return locals()


def model1_parameter_sensitivity(test=False):
    model_name = "model1_parameter_sensitivity"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

    model1_parameter_dict = model1_parameters(test)
    parameter_construction_func = parameter_sensitivity_model1
    hook_after_all_iterations = final_processing_parameter_sensitivity_model1

    deviation_factor_dict = {'mid': [0.1, 0.9], 'flux': [0.1, 0.7]}
    sigma_dict = {'mid': 0.5, 'flux': 0.4}

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


def model1_parameter_sensitivity_old(test=False):
    model_name = "model1_parameter_sensitivity"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

    data_collection_kwargs = {
        'label_list': ["glucose"], 'mouse_id_list': ['M1'],
        'source_tissue_marker': constant_set.liver_marker, 'sink_tissue_marker': constant_set.muscle_marker}
    data_loader_func = data_loader_rabinowitz
    data_collection_func = mid_data_loader_model1234
    # model_mid_data_dict = data_loader_rabinowitz(mid_data_loader_model1234, data_collection_kwargs)

    hook_in_each_iteration = metabolite_contribution_model12
    hook_after_all_iterations = final_processing_parameter_sensitivity_model1
    model_construction_func = model1_construction
    parameter_construction_func = parameter_sensitivity_model1

    complete_flux_list = ['F{}'.format(i + 1) for i in range(10)] + ['G{}'.format(i + 1) for i in range(9)] + \
                         ['Fcirc_glc', 'Fcirc_lac']
    complete_flux_dict = {var: i for i, var in enumerate(complete_flux_list)}
    constant_flux_dict = {'Fcirc_glc': 150.9, 'Fcirc_lac': 374.4, 'F10': 100}

    min_flux_value = 1
    max_flux_value = 1000
    optimization_repeat_time = 10
    obj_tolerance = 0.2
    deviation_factor_dict = {'mid': [0.1, 0.9], 'flux': [0.1, 0.9]}
    sigma_dict = {'mid': 0.5, 'flux': 0.5}
    f1_range = [1, 150]
    g2_range = [1, 150]
    if test:
        f1_num = 21
        f1_display_interv = 20
        g2_num = 21
        g2_display_interv = 20
        parameter_sampling_num = 10
    else:
        f1_num = 100
        f1_display_interv = 20
        g2_num = 100
        g2_display_interv = 20
        parameter_sampling_num = 100

    return locals()


def model1_m5_parameters(test=False):
    model_name = "model1_m5"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

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
    model_name = "model1_m9"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

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
    model_name = "model1_lactate"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

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
    model_name = "model1_lactate_m4"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

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
    model_name = "model1_lactate_m10"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

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
    model_name = "model1_lactate_m11"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

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


def model1_all_tissue_m5(test=False):
    model_name = "model1_all_tissue_m5"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

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
    model_name = "model1_all_tissue_m9"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

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
    model_name = "model1_all_tissue_lactate"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

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
    model_name = "model1_all_tissue_lactate_m4"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

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
    model_name = "model1_all_tissue_lactate_m10"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

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
    model_name = "model1_all_tissue_lactate_m11"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

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


def model1_hypoxia_correction(test=False):
    model_name = "model1_all_tissue_hypoxia"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

    # Positive means increasing labeling ratio. Negative means decreasing labeling ratio
    # Correction of hypoxia should increase glucose in source, increase lactate in sink and decrease pyruvate in sink.
    # hypoxia_correction_parameter_dict = {
    #     'glc_source': 0.1,
    #     'lac_sink': 0.05,
    #     'pyr_sink': -0.05,
    # }
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
    model_name = "model1_all_tissue_lactate_m11"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

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


def model3_all_tissue(test=False):
    model_name = "model3_all_tissue"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

    model3_parameter_dict = model3_parameters(test)

    data_collection_kwargs = {
        'label_list': ["glucose"], 'mouse_id_list': ['M1'],
        'source_tissue_marker': constant_set.liver_marker,
        'sink_tissue_marker_list': [
            constant_set.heart_marker, constant_set.brain_marker, constant_set.muscle_marker,
            constant_set.kidney_marker, constant_set.lung_marker, constant_set.pancreas_marker,
            constant_set.intestine_marker, constant_set.spleen_marker]}
    data_collection_func = mid_data_loader_all_tissue
    parameter_construction_func = all_tissue_model3

    model3_parameter_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'data_collection_kwargs': data_collection_kwargs,
        'data_collection_func': data_collection_func,
        'parameter_construction_func': parameter_construction_func,
    })
    return model3_parameter_dict


def model3_all_tissue_old(test=False):
    model_name = "model3_all_tissue"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

    data_collection_kwargs = {
        'label_list': ["glucose"], 'mouse_id_list': ['M1'],
        'source_tissue_marker': constant_set.liver_marker,
        'sink_tissue_marker_list': [
            constant_set.heart_marker, constant_set.brain_marker, constant_set.muscle_marker,
            constant_set.kidney_marker, constant_set.lung_marker, constant_set.pancreas_marker,
            constant_set.intestine_marker, constant_set.spleen_marker]}
    data_loader_func = data_loader_rabinowitz
    data_collection_func = mid_data_loader_all_tissue

    hook_in_each_iteration = metabolite_contribution_model34
    hook_after_all_iterations = final_processing_dynamic_range_model345
    model_construction_func = model3_construction
    parameter_construction_func = all_tissue_model3

    complete_flux_list = ['F{}'.format(i + 1) for i in range(12)] + ['G{}'.format(i + 1) for i in range(11)] + \
                         ['J{}'.format(i + 1) for i in range(3)] + ['Fcirc_glc', 'Fcirc_lac', 'Fcirc_pyr']
    complete_flux_dict = {var: i for i, var in enumerate(complete_flux_list)}
    constant_flux_dict = {'Fcirc_glc': 150.9, 'Fcirc_lac': 374.4, 'Fcirc_pyr': 57.3, 'F12': 200}

    min_flux_value = 1
    max_flux_value = 1000
    optimization_repeat_time = 10
    obj_tolerance = 0.15
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
        total_point_num = int(1e2)
        ternary_resolution = int(2 ** 7)
    else:
        total_point_num = int(1e6)
        ternary_resolution = int(2 ** 8)

    return locals()


def model3_all_tissue_m5(test=False):
    model_name = "model3_all_tissue_m5"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

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
    model_name = "model3_all_tissue_m9"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

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
    model_name = "model3_all_tissue_lactate"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

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
    model_name = "model3_all_tissue_lactate_m4"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

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
    model_name = "model3_all_tissue_lactate_m10"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

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
    model_name = "model3_all_tissue_lactate_m11"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

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


def model6_m2_parameters(test=False):
    model_name = "model6_m2"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

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
    model_name = "model6_m3"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

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
    model_name = "model6_m4"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

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


def model7_m2_parameters(test=False):
    model_name = "model7_m2"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

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
    model_name = "model7_m3"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

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
    model_name = "model7_m4"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

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


def model1_unfitted_parameters(test=False):
    model_name = "model1_unfitted"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

    model1_parameter_dict = model1_parameters(test)
    model1_parameter_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'fitted': False,
        'obj_tolerance': 999999
    })

    return model1_parameter_dict


def model3_unfitted_parameters(test=False):
    model_name = "model3_unfitted"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

    model3_parameter_dict = model3_parameters(test)
    model3_parameter_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'fitted': False,
        'obj_tolerance': 999999
    })

    return model3_parameter_dict


def model5_unfitted_parameters(test=False):
    model_name = "model5_unfitted"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

    model5_parameter_dict = model5_parameters(test)
    model5_parameter_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'fitted': False,
        'obj_tolerance': 999999
    })

    return model5_parameter_dict


def model6_unfitted_parameters(test=False):
    model_name = "model6_unfitted"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

    model6_parameter_dict = model6_parameters(test)
    model6_parameter_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'fitted': False,
        'obj_tolerance': 999999
    })

    return model6_parameter_dict


def model7_unfitted_parameters(test=False):
    model_name = "model7_unfitted"
    output_direct = "{}/{}".format(constant_set.output_direct, model_name)

    model7_parameter_dict = model7_parameters(test)
    model7_parameter_dict.update({
        'model_name': model_name,
        'output_direct': output_direct,
        'fitted': False,
        'obj_tolerance': 999999
    })

    return model7_parameter_dict


