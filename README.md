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

For convenience, an out-of-the-box Docker image is provided to run this code. This docker image is tested on `docker-ce` in Ubuntu with Docker version `19.03.1`.

## Usages

### Docker (recommended)
First install an appropriate Docker version ([see here for details](https://docs.docker.com/install/)). Then a model could be executed by the following script:

```shell script
MODEL=model1_m5
TARGET_PATH=/your/path/to/output
cd $TARGET_PATH
docker run -it --rm --name python_$MODEL -v `pwd`:/lactate_exchange/new_models \
  -e PARAM=$MODEL locasalelab/lactate_mfa:latest;
```

In this script, you could modify the value of `MODEL` to the name of your target model, and modify the value of `TARGET_PATH` to the path that you want to output results. Available model name is listed in following section. TAKE CARE that the target path would be visited as root account in container! 

### Raw Python

This script could also be executed as a raw Python project. Make sure Python 3.6 and all required packages are correctly installed. First switch to a target directory and download the source code:

```shell script
git clone https://github.com/LocasaleLab/Lactate_MFA
```

Switch to the source direct, add PYTHONPATH environment and run the `new_model_main.py`:

```shell script
cd Lactate_MFA
export PYTHONPATH=$PYTHONPATH:`pwd`
python src/new_model_main.py $MODEL
```

Similar with Docker, you could modify the value of `MODEL` to the name of your target model. Final results will be written to the `new_models` folder in current directory. Available model name is listed in following section.

### List of models

|   Model name in this script |  Model name in methods | Source tissue | Sink tissue | Circulatory metabolites| Data |  Description |
|  ----  | ----  |  ----  |  ----  | ----  |  ----  |  ----  |
| `model1`  | Model A | Liver | Heart | Glucose; Lactate | Low-infusion data: mouse M1 | Basic two-tissue model. |
| `model1_all`  | Model A | Liver | All 8 tissues | Glucose; Lactate | Low-infusion data: mouse M1 | Basic two-tissue model with different sink tissues. |
| `model1_m5`  | Model A | Liver | Heart | Glucose; Lactate | Low-infusion data: mouse M5 | Basic two-tissue model with different mouse data |
| `model1_m9`  | Model A | Liver | Heart | Glucose; Lactate | Low-infusion data: mouse M9 | Basic two-tissue model with different mouse data |
| `parameter`  | Model A | Liver | Heart | Glucose; Lactate | Low-infusion data: mouse M1 | Sensitivity analysis of data and other constraint fluxes. |
| `model3`  | Model D | Liver | Heart | Glucose; Pyruvate; Lactate | Low-infusion data: mouse M1 | Two-tissue model with three circulatory metabolites. |
| `model3_all`  | Model D | Liver | All 8 tissues | Glucose; Pyruvate; Lactate | Low-infusion data: mouse M1 | Two-tissue model with three circulatory metabolites and different sink tissues. |
| `model5`  | Model C | Liver | Heart; Skeletal muscle | Glucose; Lactate | Low-infusion data: mouse M1 | Three-tissue model. |
| `model6`  | Model B | Liver | Skeletal muscle | Glucose; Lactate | High-infusion data: mouse M1 | Two-tissue model with high-infusion data. |
| `model7`  | Model E | Liver | Skeletal muscle | Glucose; Pyruvate; Lactate | High-infusion data: mouse M1 | Two-tissue model with three circulatory metabolites and high-infusion data. |


## Contributors

**Shiyu Liu**

+ [http://github.com/liushiyu1994](http://github.com/liushiyu1994)

## License

This software is released under the [MIT License](LICENSE-MIT).
