# Deflated Dynamics Value Iteration

This is the official implementation for the paper [Deflated Dynamics Value Iteration](https://openreview.net/pdf?id=IbQTE24aZw) at TMLR 2025.

## DDVI Experiments

To run the experiment for Figure 1, set the environment config in the files ``planning/exp_DDVI.py`` and ``planning/plotter_ddvi``. Then run:
```
python planning/exp_DDVI.py
python planning/plotter_ddvi.py
```

To run experiments for Figure 2, run
```
python planning/exp_error_comparison.py
python planning/exp_horizon_comparison.py
python planning/exp_size_comparison.py
python planning/plotter_comparison.py
```

## DDTD Experiments
For Figure 3, run the following in the ``sample_based`` directory:

```
python exp_pe_sample.py config_maze55_0.95.yaml ALL --num_trials 20
python exp_pe_sample.py config_chainwalk_0.95.yaml ALL --num_trials 20
```

For plotting, use plotter_pe_main.py and ``plotter_pe_multimodel_error.py``. The path to data should be set in the ``exp_dir`` variable.
