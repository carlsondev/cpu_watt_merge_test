
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

class CpuEnergyPairTester:


    def __init__(self, pair_json_data : Optional[Dict[str, Any]], merged_data_csv_path : Optional[str]):

        if pair_json_data is None or merged_data_csv_path is None:
            self._cpu_utils = []
            self._watts = []
            self._regression_coefs = []
            self._gen_cpu_utils = []
            self._gen_watts = []

        cpu_energy_df : pd.DataFrame = pd.read_csv(merged_data_csv_path)

        self._cpu_utils : List[float] = [row["cpu_util"] for _, row in cpu_energy_df.iterrows()]
        self._watts : List[float] = [row["watts"] for _, row in cpu_energy_df.iterrows()]

        
        self._regression_coefs = pair_json_data["regression"]["coefs"]

        self._gen_cpu_utils = self._generate_rand_cpu_utils(pair_json_data["bin_ordering"], pair_json_data["bin_data"])
        self._gen_watts = self._generate_watts()


    def get_utilization_stats(self) -> Tuple[Tuple[float, float], float]:
        return self._get_stats(self._cpu_utils, self._gen_cpu_utils)
    
    def get_watts_stats(self) -> Tuple[Tuple[float, float], float]:
        return self._get_stats(self._watts, self._gen_watts)
    
    def plot_cpu_utilization(self):
        plt.plot(list(range(len(self._cpu_utils))), self._cpu_utils, label="Real CPU Utilization")
        plt.plot(list(range(len(self._gen_cpu_utils))), self._gen_cpu_utils, label="Predicted CPU Utilization")
        plt.legend()
        plt.show()

    def plot_watts(self):
        plt.plot(list(range(len(self._watts))), self._watts, label="Real Watts")
        plt.plot(list(range(len(self._gen_watts))), self._gen_watts, label="Predicted Watts")
        plt.legend()
        plt.show()

    def poly_reg_str(self) -> str:

        coefs : List[float] = list(reversed(self._regression_coefs))
        ret_str : str = ""
        for i in range(len(coefs)):
            if i == 0:
                ret_str += f"{coefs[i]:.2f}"
            elif i == 1:
                ret_str += f" + {coefs[i]:.2f}x"
            else:
                ret_str += f" + {coefs[i]:.2f}x^{i}"


        return ret_str

    def exp_reg_str(coefs : List[float]) -> str:
        return f"{coefs[0]:.2f}e^{coefs[1]:.2f}x"


    def _generate_rand_cpu_utils(bin_ordering : int, bin_data : Dict[str, Any]) -> List[float]:

        generated_cpu_utils : List[float] = []

        for bin_index in bin_ordering:
            bin_mean : float = bin_data[str(bin_index)]["mean"]
            bin_std : float = bin_data[str(bin_index)]["std"]

            gen_cpu : float = np.random.normal(bin_mean, bin_std)
            generated_cpu_utils.append(gen_cpu)

        return generated_cpu_utils
    
    def _generate_watts(self) -> List[float]:
        regression = np.poly1d(self._regression_coefs)
        generated_watts : List[float] = regression(self.gen_cpu_utils)

        r_sq = r2_score(self._watts, generated_watts)

        print(f"Polynomial Regression Eq: {self.poly_reg_str()}, R^2: {r_sq:.2f}")
        return generated_watts
    
    def _get_stats(real_data : List[float], pred_data : List[float]) -> Tuple[Tuple[float, float], float]:
        rmse_error = np.sqrt(mean_squared_error(real_data, pred_data))
        percent_error = (rmse_error / np.max(real_data)) * 100
        corr_matrix = np.corrcoef(real_data, pred_data)
        r_sq = corr_matrix[0,1]**2
        return (rmse_error, percent_error), r_sq
    

    def __add__(self, other : "CpuEnergyPairTester") -> "CpuEnergyPairTester":
        tester = CpuEnergyPairTester(None, None)
        tester._cpu_utils = self._cpu_utils + other._cpu_utils
        tester._watts = self._watts + other._watts
        tester._gen_cpu_utils = self._gen_cpu_utils + other._gen_cpu_utils
        tester._gen_watts = self._gen_watts + other._gen_watts
        return tester