import xlrd
import numpy as np

kTissueList = ['Sr', 'AT', 'Br', 'Ht', 'Kd', 'Lg', 'Lv', 'Pc', 'SI', 'SkM', 'Sp']


# DataCollect:
# .exp_name: Sup_Fig_5_fasted
# .mid_data:
# {'glucose':
#       {
#           'mouse_id1':
#           {
#               'tissue1': {'glucose': array([0.9, 0, 0, 0, 0, 0, 0.1]), ...},
#               ...
#           }
#           ...
#       },
# ...
# }
class DataCollect:
    def __init__(self, experiment_name, mid_data_dict):
        self.experiment_name = experiment_name
        self.mid_data = mid_data_dict


class OldDataCollect:
    def __init__(self, label_name, serum_mid_dict, tissue_mid_dict):
        self.label_name = label_name
        self.serum_mids = serum_mid_dict
        self.tissue_mids = tissue_mid_dict


def data_loader(file_path, sheet_name):
    def load_one_part(metabolite_name_list, carbon_num_list, start_row_list):
        part_mids = {}
        for index, metabolite_name in enumerate(metabolite_name_list):
            carbon_num = carbon_num_list[index]
            start_row = start_row_list[index]
            mid_list = []
            for add_row in range(carbon_num + 1):
                mid_list.append(data_sheet.cell_value(start_row + add_row, experiment_col))
            part_mids[metabolite_name] = np.array(mid_list)
        return part_mids

    data_book = xlrd.open_workbook(str(file_path))
    data_sheet = data_book.sheet_by_name(sheet_name)
    experiment_label_list = ["glucose", "lactate"]
    experiment_col_list = [2, 3]
    serum_metabolite_name_list = ["glucose", "lactate", "pyruvate"]
    serum_carbon_num_list = [6, 3, 3]
    serum_start_row_list = [3, 11, 16]
    tissue_metabolite_name_list = [
        "glucose", "g6p", "3pg", "lactate", "pyruvate", "alanine",
        "serine", "citrate", "succinate", "malate", "akg", "s7p"]
    tissue_carbon_num_list = [6, 6, 3, 3, 3, 3, 3, 6, 4, 4, 5, 7]
    tissue_start_row_list = [23, 31, 39, 44, 48, 52, 56, 60, 67, 72, 77, 83]
    data_collect_dict = {}

    for experiment_label, experiment_col in zip(experiment_label_list, experiment_col_list):
        serum_mids = load_one_part(
            serum_metabolite_name_list, serum_carbon_num_list, serum_start_row_list)
        tissue_mids = load_one_part(
            tissue_metabolite_name_list, tissue_carbon_num_list, tissue_start_row_list)
        data_collect_dict[experiment_label] = OldDataCollect(experiment_label, serum_mids, tissue_mids)

    return data_collect_dict


def data_parser(file_path, experiment_name_prefix, label_list):
    data_book = xlrd.open_workbook(str(file_path))
    mid_data_dict = {}
    for label_name in label_list:
        current_label_data_dict = {}
        sheet_name = "{}_{}".format(experiment_name_prefix, label_name)
        data_sheet = data_book.sheet_by_name(sheet_name)
        current_col = 0
        metabolite_name_col = 0
        while current_col != data_sheet.ncols:
            sample_count_dict = {}
            top_row_value = data_sheet.cell_value(0, current_col)
            if top_row_value == 'Compound':
                metabolite_name_col = current_col
                current_col += 2
            else:
                if top_row_value[-1].isdigit():
                    sample_id = top_row_value[:-1]
                else:
                    sample_id = top_row_value
                try:
                    sample_count_dict[sample_id] += 1
                except KeyError:
                    sample_count_dict[sample_id] = 1
                sample_num = sample_count_dict[sample_id]
                mouse_id, tissue = sample_id.split('_')
                if tissue not in kTissueList:
                    raise ValueError("Tissue not recognized! Col: {} Tissue: {}".format(current_col, tissue))
                this_tissue_dict = {}
                mid_list_dict = {}
                for current_row in range(1, data_sheet.nrows):
                    current_value = data_sheet.cell_value(current_row, current_col)
                    metabolite_name = data_sheet.cell_value(current_row, metabolite_name_col)
                    if current_value is None or current_value == "":
                        break
                    else:
                        try:
                            mid_list_dict[metabolite_name].append(current_value)
                        except KeyError:
                            mid_list_dict[metabolite_name] = [current_value]
                if sample_num == 1:
                    for metabolite, mid_list in mid_list_dict.items():
                        this_tissue_dict[metabolite] = np.array(mid_list)
                else:
                    for metabolite, mid_list in mid_list_dict.items():
                        this_tissue_dict[metabolite] = (
                                                               this_tissue_dict[metabolite] * (
                                                               sample_num - 1) + np.array(mid_list)) / sample_num
                try:
                    current_label_data_dict[mouse_id][tissue] = this_tissue_dict
                except KeyError:
                    current_label_data_dict[mouse_id] = {tissue: this_tissue_dict}
                current_col += 1
        mid_data_dict[label_name] = current_label_data_dict
    return DataCollect(experiment_name_prefix, mid_data_dict)


def data_checker(data_collection, required_serum_metabolites, required_tissue_metabolites):
    data_dict = data_collection.mid_data
    new_mid_data_dict = {}
    for experiment, current_exp_data_dict in data_dict.items():
        new_exp_data_dict = {}
        for mouse_id, current_mouse_data_dict in current_exp_data_dict.items():
            valid = True
            for metabolite in required_serum_metabolites:
                if metabolite not in current_mouse_data_dict['Sr']:
                    valid = False
            for metabolite in required_tissue_metabolites:
                if metabolite not in current_mouse_data_dict['Lv']:
                    valid = False
            if valid:
                new_exp_data_dict[mouse_id] = current_mouse_data_dict
        new_mid_data_dict[experiment] = new_exp_data_dict
    return DataCollect(data_collection.experiment_name, new_mid_data_dict)


def main():
    file_path = "data_collection.xlsx"
    experiment_name_prefix = "Sup_Fig_5_fasted"
    label_list = ["glucose", "lactate"]
    data_collection = data_parser(file_path, experiment_name_prefix, label_list)
    print(data_collection)


if __name__ == '__main__':
    main()
