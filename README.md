# Metabolic Flux Analysis

## Synopsis

This software is used for 13C metabolic flux analysis based on a simple model. The 13C tracing data in `13C-Glucose_tracing_Mike.xlsx` is used as input. Each sheet contains data from one experiment. Relative fluxes and their standard deviations in different conditions are exported to `.csv` files after solving the model.

## Requirements

This software is developed and tested on Python 3.6. It also relies on Python package `pandas`, `numpy` and `scipy`. It has been tested on `pandas` 0.22, `numpy` 1.14 and `scipy` 1.0.

## Usages

The input data are in `13C-Glucose_tracing_Mike.xlsx` file. To solve the MFA model, run `main.py`. Results will be displayed in `.csv` files in the same path. Each `.csv` file corresponds to one experiment (one sheet in `13C-Glucose_tracing_Mike.xlsx`).

## Contributors

**Shiyu Liu**

+ [http://github.com/liushiyu1994](http://github.com/liushiyu1994)

## License

This software is released under the [MIT License](LICENSE-MIT).
