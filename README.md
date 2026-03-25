# Higher-Order Portfolio Optimization with QAOA -- Online Appendix

## Detailed portfolio optimization results with higher moments

<img src="visualizations/6_qubits_mean_variance_obj_vs_budget.png" alt="drawing" width="1000"/>

<img src="visualizations/7_qubits_mean_variance_obj_vs_budget.png" alt="drawing" width="1000"/>

<img src="visualizations/8_qubits_mean_variance_obj_vs_budget.png" alt="drawing" width="1000"/>

<img src="visualizations/9_qubits_mean_variance_obj_vs_budget.png" alt="drawing" width="1000"/>

<img src="visualizations/10_qubits_mean_variance_obj_vs_budget.png" alt="drawing" width="1000"/>

<img src="visualizations/11_qubits_mean_variance_obj_vs_budget.png" alt="drawing" width="1000"/>

<img src="visualizations/12_qubits_mean_variance_obj_vs_budget.png" alt="drawing" width="1000"/>

<img src="visualizations/13_qubits_mean_variance_obj_vs_budget.png" alt="drawing" width="1000"/>

<img src="visualizations/14_qubits_mean_variance_obj_vs_budget.png" alt="drawing" width="1000"/>

<img src="visualizations/15_qubits_mean_variance_obj_vs_budget.png" alt="drawing" width="1000"/>

## Variance of differences in mean-variance vs. higher order optimization cases

<img src="visualizations/variance_spectrum_differences.png" alt="drawing" width="1000"/>

## Spectrum example with higher number of qubits

<img src="visualizations/spectrum_plot_13.png" alt="drawing" width="1000"/>

## Main files in this repo

1. The notebook generate_portfolio_experiments.ipynb generates random portfolio optimization problems based on the real stock data from yfinance.
2. Python file experiments.py contains the main experiments
3. Python file coskweness_cokurtosis.py has the functions to compute the higher-order moments
4. Python file portfolio_higher_moments_classical.py contains the classical baselines