
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
            return

        cpu_energy_df : pd.DataFrame = pd.read_csv(merged_data_csv_path)

        self._cpu_utils : List[float] = [row["cpu_util"] for _, row in cpu_energy_df.iterrows()]
        self._watts : List[float] = [row["watts"] for _, row in cpu_energy_df.iterrows()]

        self.generate_cpu_watts(pair_json_data, test_poly_stds=False)

    def _find_best_poly_std_period(self) -> Tuple[int, float]:
        min_rmse = float("inf")
        min_index = 0
        for i in range(1, len(self._gen_cpu_utils)):
            _, rmse = self._generate_watts(use_poly_stds=True, std_period=i, print_reg=False)
            if rmse < min_rmse:
                min_rmse = rmse
                min_index = i

        print(f"Best poly std period: {min_index} ({(min_index/len(self._gen_cpu_utils))*100}%), RMSE: {min_rmse}")
        return min_index, min_rmse

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


    def _generate_rand_cpu_utils(self, bin_ordering : int, bin_data : Dict[str, Any]) -> List[float]:

        generated_cpu_utils : List[float] = []

        for bin_index in bin_ordering:
            bin_mean : float = bin_data[str(bin_index)]["mean"]
            bin_std : float = bin_data[str(bin_index)]["std"]

            gen_cpu : float = np.random.normal(bin_mean, bin_std)

            gen_cpu = max(0, gen_cpu)
            gen_cpu = min(100, gen_cpu)
            
            generated_cpu_utils.append(gen_cpu)

        return generated_cpu_utils
    
    def _generate_watts(self, use_poly_stds : bool, std_period : int = 1, print_reg : bool = True) -> Tuple[List[float], float]:
        regression = np.poly1d(self._regression_coefs)
        generated_watts : List[float] = regression(self._gen_cpu_utils)

        if use_poly_stds:
                for idx, cpu_util in enumerate(self._gen_cpu_utils):

                    if idx % std_period != 0:
                        continue

                    cpu_std_idx = int(cpu_util) if cpu_util < 100 else 99
                    generated_watts[idx] += np.random.normal(0, self._regression_stds[cpu_std_idx])


        (rmse_error, percent_error), r_sq = self._get_stats(self._watts, generated_watts)

        if print_reg:
            print(f"Polynomial Regression Eq: {self.poly_reg_str()}, R^2: {r_sq:.2f}")
        return list(generated_watts), rmse_error
    
    def _get_stats(self, real_data : List[float], pred_data : List[float]) -> Tuple[Tuple[float, float], float]:
        rmse_error = np.sqrt(mean_squared_error(real_data, pred_data))
        percent_error = (rmse_error / np.max(real_data)) * 100
        corr_matrix = np.corrcoef(real_data, pred_data)
        r_sq = corr_matrix[0,1]**2
        return (rmse_error, percent_error), r_sq
    
    # Coefficents in the order of b_0, b_1, b_2, ..., b_n
    def test_custom_reg(self, reg_coefs : List[float]):

        reg_dict = {
            "coefs" : list(reversed(reg_coefs)),
            "poly_stds" : []
        }

        self.generate_cpu_watts({"regression" : reg_dict}, test_poly_stds=False, do_generate_cpu=False)

    def generate_cpu_watts(self, json_dict : Dict[str, Any], test_poly_stds : bool = False, do_generate_cpu : bool = True) -> None:
        regression_dict = json_dict["regression"]
        self._regression_coefs = regression_dict["coefs"]
        self._regression_stds = regression_dict["poly_stds"]

        if do_generate_cpu:
            self._gen_cpu_utils = self._generate_rand_cpu_utils(json_dict["bin_ordering"], json_dict["cpu_bins"])

        lin_gen_watts, lin_rmse = self._generate_watts(use_poly_stds=False)

        if not test_poly_stds:
            self._gen_watts = lin_gen_watts
            return

        best_poly_std_period, best_poly_rmse = self._find_best_poly_std_period()

        use_poly_stds = best_poly_rmse < lin_rmse
        if use_poly_stds:
            print(f"The model using polynomial stds is better (by {((lin_rmse - best_poly_rmse)/lin_rmse)*100:.3f}%), using that one.")
            self._gen_watts, _ = self._generate_watts(use_poly_stds=use_poly_stds, std_period=best_poly_std_period)
            return
        self._gen_watts = lin_gen_watts
    

    def __add__(self, other : "CpuEnergyPairTester") -> "CpuEnergyPairTester":
        tester = CpuEnergyPairTester(None, None)
        tester._cpu_utils = self._cpu_utils + other._cpu_utils
        tester._watts = self._watts + other._watts
        tester._gen_cpu_utils = self._gen_cpu_utils + other._gen_cpu_utils
        tester._gen_watts = self._gen_watts + other._gen_watts
        return tester