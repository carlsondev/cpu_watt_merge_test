import numpy as np
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd


def get_stats(real_data : List[float], pred_data : List[float]) -> Tuple[Tuple[float, float], float]:
    rmse_error = np.sqrt(mean_squared_error(real_data, pred_data))
    percent_error = (rmse_error / np.max(real_data)) * 100
    corr_matrix = np.corrcoef(real_data, pred_data)
    r_sq = corr_matrix[0,1]**2
    return (rmse_error, percent_error), r_sq

def get_regression_data(lin_reg_dict : Dict[str, Any]) -> Tuple[List[float], float, List[float]]:
    coefs : List[float] = lin_reg_dict["coefficients"]
    r_sq : float = lin_reg_dict["r_squared"]
    std_coef : List[float] = lin_reg_dict["std_coefficients"]
    return coefs, r_sq, std_coef

def generate_watts_from_cpu_util(cpu_utils : List[float], regression_dict : Dict[str, Any]) -> List[float]:
    coefs, r_sq, std_coefs = get_regression_data(regression_dict)

    scaled_cpu_utils : np.ndarray = np.array([x/100 for x in cpu_utils])
    generated_watts : List[float] = list(apply_poly_regression(coefs, scaled_cpu_utils, []))
    #generated_watts : List[float] = list(apply_exp_regression(coefs, scaled_cpu_utils))

    #print(f"Polynomial Regression Eq: {poly_reg_str(coefs)}, R^2: {r_sq:.2f}")
    #print(f"Exponential Regression Eq: {exp_reg_str(coefs)}, R^2: {r_sq:.2f}")
    return generated_watts


def read_cpu_energy_csv(cpu_energy_csv : str) -> Tuple[List[float], List[float]]:
    cpu_energy_df : pd.DataFrame = pd.read_csv(cpu_energy_csv)
    cpu_utils : List[float] = [row["cpu_util"]*100 for _, row in cpu_energy_df.iterrows()]
    watts : List[float] = [row["watts"] for _, row in cpu_energy_df.iterrows()]
    return cpu_utils, watts

def apply_poly_regression(coefs : List[float], x_vals : np.ndarray, y_vals : List[float]) -> np.ndarray:
    return np.polyval(coefs, x_vals)

def apply_exp_regression(coefs : List[float], x_vals : np.ndarray) -> np.ndarray:
    return coefs[0] * np.exp(coefs[1] * x_vals)

def poly_reg_str(coefs : List[float]) -> str:
    return f"{coefs[0]:.2f} + {coefs[1]:.2f}x + {coefs[2]:.2f}x^2"

def exp_reg_str(coefs : List[float]) -> str:
    return f"{coefs[0]:.2f}e^{coefs[1]:.2f}x"

def plot_regression(cpu_utils : List[float], watts : List[float], generated_watts : List[float]) -> None:
    plt.scatter(cpu_utils, watts, color="blue", label="Actual Watts")
    plt.scatter(cpu_utils, generated_watts, color="red", label="Generated Watts")
    plt.xlabel("CPU Utilization (%)")
    plt.ylabel("Watts")
    plt.legend()
    plt.show()

def perform_energy_linear_interp(cpu_df : pd.DataFrame, energy_df : pd.DataFrame)