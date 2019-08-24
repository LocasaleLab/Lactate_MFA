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

###Docker (recommended)


###Raw Python

The input data are in `13C-Glucose_tracing_Mike.xlsx` file. To solve the MFA model, run `main.py`. Results will be displayed in `.csv` files in the same path. Each `.csv` file corresponds to one experiment (one sheet in `13C-Glucose_tracing_Mike.xlsx`).

## Contributors

**Shiyu Liu**

+ [http://github.com/liushiyu1994](http://github.com/liushiyu1994)

## License

This software is released under the [MIT License](LICENSE-MIT).
