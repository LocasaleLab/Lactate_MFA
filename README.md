# Metabolic Flux Analysis for Energy Production in Physiological Condition

## Synopsis

This software is used for 13C metabolic flux analysis based on a multi-tissue model to analyze nutrition utilization in physiological condition. The 13C tracing data in `data_collection.xlsx` (low infusion data from [Hui *et al*, 2017](https://doi.org/10.1038/nature24057)) and `data_collection_Dan.xlsx` (high infusion data) are used as input. Users can choose model to run and corresponding results will be written to `new_models` folder. 

## Requirements

This software is developed and tested on Python 3.6. It also relies on following Python packages:

|   Packages |  Version has been tested |
|  ----  | ----  |
| `numpy`  | 1.16 |
| `scipy`  | 1.3 |
| `matplotlib`  | 2.2 |
| `tqdm`  | 4.30 |
| `python-ternary`  | 1.0 |
| `xlrd`  | 1.1 |

For convenience, an out-of-the-box Docker image is provided to run this code. This docker image is tested on `docker-ce` in Ubuntu with Docker version `19.03.1`. (See Usages part for details.)

## Usages

### Docker (recommended)
First install an appropriate Docker version ([see here for details](https://docs.docker.com/install/)). Then a model could be executed by the following script:

```shell script
MODEL=model1_m5
TARGET_PATH=/your/path/to/output
cd $TARGET_PATH
docker run -it --rm --name python_$MODEL -v `pwd`:/Lactate_MFA/new_models \
  locasalelab/lactate_mfa:latest $MODEL --test_mode
```

In this script, you could modify the value of `MODEL` to the name of your target model, and modify the value of `TARGET_PATH` to the path that you want to output results. Available model name is listed in following section. BE CAREFUL that the target path would be visited as root account in container! The flag `--test_mode` or `-t` makes the code run quickly in a test mode, and could be removed to run a formal mode. The formal running takes tens of hours.

### System Python interpreter

This script could also be executed as a raw Python project. Make sure Python 3.6 and all required packages are correctly installed. First switch to a target directory and download the source code:

```shell script
git clone https://github.com/LocasaleLab/Lactate_MFA
```

Switch to the source direct, add PYTHONPATH environment and run the `new_model_main.py`:

```shell script
MODEL=model1_m5
cd Lactate_MFA
export PYTHONPATH=$PYTHONPATH:`pwd`
python src/new_model_main.py $MODEL --test_mode
```

Similar with Docker, you could modify the value of `MODEL` to the name of your target model. Final results will be written to the `new_models` folder in current directory. Available model name is listed in following section. The flag `--test_mode` or `-t` could also be removed to run a formal mode.

### Parameters

-p, --parallel_num:
    
    Number of parallel processes. If not provided, it will be selected according to CPU cores.

-t, --test_mode:

    Whether the code is executed in test mode, which means less sample number and shorter time (several minites).


### List of models

|   Model name in this script |  Model name in methods | Source tissue | Sink tissue | Circulating metabolites| Data source |  Description |
|  ----  | ----  |  ----  |  ----  | ----  |  ----  |  ----  |
| `model1`  | Model A | Liver | Heart | Glucose; Lactate | Low-infusion glucose data: mouse M1 | Basic two-tissue model. |
| `model1_unfitted`  | Model A | Liver | Heart | Glucose; Lactate | Low-infusion glucose data: mouse M1 | Unfitted result of basic two-tissue model, as the negative result of fitting. |
| `model1_all`  | Model A | Liver | All 8 tissues | Glucose; Lactate | Low-infusion glucose data: mouse M1 | Basic two-tissue model with different sink tissues. |
| `model1_all_m5`  | Model A | Liver | All 8 tissues | Glucose; Lactate | Low-infusion glucose data: mouse M5 | Basic two-tissue model with different sink tissues and different mouse data. |
| `model1_all_m9`  | Model A | Liver | All 8 tissues | Glucose; Lactate | Low-infusion glucose data: mouse M9 | Basic two-tissue model with different sink tissues and different mouse data. |
| `model1_all_lactate`  | Model A | Liver | All 8 tissues | Glucose; Lactate | Low-infusion lactate data: mouse M3 | Basic two-tissue model with different sink tissues and different infusion data. |
| `model1_all_lactate_m4`  | Model A | Liver | All 8 tissues | Glucose; Lactate | Low-infusion lactate data: mouse M4 | Basic two-tissue model with different sink tissues and different infusion data. |
| `model1_all_lactate_m10`  | Model A | Liver | All 8 tissues | Glucose; Lactate | Low-infusion lactate data: mouse M10 | Basic two-tissue model with different sink tissues and different infusion data. |
| `model1_all_lactate_m11`  | Model A | Liver | All 8 tissues | Glucose; Lactate | Low-infusion lactate data: mouse M11 | Basic two-tissue model with different sink tissues and different infusion data. |
| `parameter`  | Model A | Liver | Heart | Glucose; Lactate | Low-infusion data: mouse M1 | Sensitivity analysis of data and other constraint fluxes. |
| `model6`  | Model B | Liver | Skeletal muscle | Glucose; Lactate | High-infusion data: mouse M1 | Two-tissue model with high-infusion data in different mouse strain. |
| `model6_unfitted`  | Model B | Liver | Skeletal muscle | Glucose; Lactate | High-infusion data: mouse M1 | Unfitted result of two-tissue model with high-infusion flux, as the negative result of fitting. |
| `model6_m2`  | Model B | Liver | Skeletal muscle | Glucose; Lactate | High-infusion data: mouse M2 | Two-tissue model with high-infusion data in different mouse strain. |
| `model6_m3`  | Model B | Liver | Skeletal muscle | Glucose; Lactate | High-infusion data: mouse M3 | Two-tissue model with high-infusion data in different mouse strain. |
| `model6_m4`  | Model B | Liver | Skeletal muscle | Glucose; Lactate | High-infusion data: mouse M4 | Two-tissue model with high-infusion data in different mouse strain. |
| `model3`  | Model D | Liver | Heart | Glucose; Pyruvate; Lactate | Low-infusion glucose data: mouse M1 | Two-tissue model with three circulatory metabolites. |
| `model3_unfitted`  | Model D | Liver | Heart | Glucose; Pyruvate; Lactate | Low-infusion glucose data: mouse M1 | Unfitted result of two-tissue model with three circulatory metabolites, as the negative result of fitting. |
| `model3_all`  | Model D | Liver | All 8 tissues | Glucose; Pyruvate; Lactate | Low-infusion glucose data: mouse M1 | Two-tissue model with three circulatory metabolites and different sink tissues. |
| `model5`  | Model C | Liver | Heart; Skeletal muscle | Glucose; Lactate | Low-infusion data: mouse M1 | Three-tissue model. |
| `model5_comb2`  | Model C | Liver | Brain; Skeletal muscle | Glucose; Lactate | Low-infusion data: mouse M1 | Three-tissue model. |
| `model5_comb3`  | Model C | Liver | Heart; Brain | Glucose; Lactate | Low-infusion data: mouse M1 | Three-tissue model. |
| `model5_unfitted`  | Model C | Liver | Heart; Skeletal muscle | Glucose; Lactate | Low-infusion data: mouse M1 | Unfitted result of three-tissue model, as the negative result of fitting. |
| `model7`  | Model E | Liver | Skeletal muscle | Glucose; Pyruvate; Lactate | High-infusion data: mouse M1 | Two-tissue model with three circulatory metabolites and high-infusion data. |
| `model7_unfitted`  | Model E | Liver | Skeletal muscle | Glucose; Pyruvate; Lactate | High-infusion data: mouse M1 | Unfitted result of two-tissue model with three circulatory metabolites and high-infusion flux, as the negative result of fitting. |

## Result display

Results generated in computation are plotted to figures by functions in Jupyter Notebook file `data_process.ipynb`. Figures in paper can also be found in this file.

## Contributors

**Shiyu Liu**

+ [http://github.com/liushiyu1994](http://github.com/liushiyu1994)

## License

This software is released under the [MIT License](LICENSE-MIT).
