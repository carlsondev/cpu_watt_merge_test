import os
import pandas as pd
import json
from scipy.interpolate import interp1d
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import math
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

class CpuEnergyPair:


    def __init__(self, cpu_file_path : Optional[str] = None, energy_file_path : Optional[str] = None, cpu_df : Optional[pd.DataFrame] = None, energy_df : Optional[pd.DataFrame] = None):

        self._raw_cpu_path = cpu_file_path
        self._raw_energy_path = energy_file_path

        if cpu_df is not None:
            self._cpu_df = cpu_df
        else:
            self._cpu_df = self._convert_raw_cpu_to_csv()
            self._compute_cpu_utilizations()


        if energy_df is not None:
            self._energy_df = energy_df
        else:
            self._energy_df = self._convert_raw_energy_to_csv()

        self._merged_cpu_energy_df : Optional[pd.DataFrame] = None


    def merge_data(self) -> pd.DataFrame:

        merged_df = pd.DataFrame(columns=["collection_time", "cpu_util", "watts"])


        for cpu_row in self._cpu_df.itertuples():
            cpu_time = cpu_row.collection_time
            cpu_util = cpu_row.cpu_util
            wattage = self.find_energy_for_cpu_time(cpu_time)

            new_row = pd.DataFrame({"collection_time": [cpu_time], "cpu_util": [cpu_util], "watts": [wattage]})
            merged_df = pd.concat([merged_df, new_row], ignore_index=True, sort=False)

        self._merged_cpu_energy_df = merged_df

        print("Merged CPU and Energy data")
        return merged_df
            

    def apply_moving_average(self, window_size : int = 7):

        self._cpu_df["cpu_util"] = self._cpu_df["cpu_util"].rolling(window=window_size).mean()
        self._cpu_df.dropna(inplace=True)

        self._energy_df["watts"] = self._energy_df["watts"].rolling(window=window_size).mean()
        self._energy_df.dropna(inplace=True)

        print(f"Applied moving average of window size {window_size} to both CPU and Energy data")

    def find_energy_for_cpu_time(self, cpu_time : float) -> float:
        
        time_delta = 0.05

        energy_times = [float(x) for x in self._energy_df["collection_time"].to_list()]

        def bin_search():
            low = 0
            high = len(energy_times) - 1
            mid = 0
        
            while low <= high:
        
                mid = (high + low) // 2
        
                # If x is greater, ignore left half
                if energy_times[mid] < cpu_time - time_delta:
                    low = mid + 1
        
                # If x is smaller, ignore right half
                elif energy_times[mid] > cpu_time + time_delta:
                    high = mid - 1
        
                # means x is present at mid
                else:
                    return mid, True
        
            # If we reach here, then the element was not present
            return low, False

        energy_idx, was_found = bin_search()
        if was_found:
            return self._energy_df.iloc[energy_idx]["watts"]
        
        if energy_idx == 0:
            energy_rows = self._energy_df.iloc[0:energy_idx+2]
        elif energy_idx == len(energy_times) - 1:
            energy_rows = self._energy_df.iloc[energy_idx-2:energy_idx]
        else:
            energy_rows = self._energy_df.iloc[energy_idx-1:energy_idx+1]


        # Perform lerp of nearby values
        neighboring_times = energy_rows["collection_time"].to_list()
        neighboring_watts = energy_rows["watts"].to_list()

        interp = interp1d(neighboring_times, neighboring_watts)

        return interp(cpu_time)
        

    def compute_cpu_data(self, bin_count : int) -> Tuple[List[int], Dict[str, Dict[str, float]]]:

        cpu_data = self._cpu_df["cpu_util"].to_list()
        bin_width = (max(cpu_data) - min(cpu_data)) / bin_count

        cpu_bin_ordering : List[int] = []
        cpu_bins : Dict[str, List[float]] = {}
        for util in cpu_data:
            bin_idx = math.floor(((util - min(cpu_data)) / bin_width))
            if bin_idx == 10:
                bin_idx = 9 # Fix for rounding error
            cpu_bin_ordering.append(bin_idx)
            cpu_bins[str(bin_idx)] = cpu_bins.get(str(bin_idx), []) + [util]

        cpu_bin_data : Dict[str, Dict[str, float]] = {}
        for bin_idx, bin_data in cpu_bins.items():
            cpu_bin_data[bin_idx] = {
                "mean": np.mean(bin_data),
                "std": np.std(bin_data),
                "n": len(bin_data)
            }

        return cpu_bin_ordering, cpu_bin_data
        
    def compute_poly_regression(self, degree : int) -> Tuple[List[float], float]:

        if self._merged_cpu_energy_df is None:
            self._merged_cpu_energy_df = self.merge_data()

        cpu_data = self._merged_cpu_energy_df["cpu_util"].to_list()
        energy_data = self._merged_cpu_energy_df["watts"].to_list()

        poly_coefs = np.polyfit(cpu_data, energy_data, degree)
        poly_fn = np.poly1d(poly_coefs)

        r_2 = r2_score(energy_data, poly_fn(cpu_data))
        return poly_coefs, r_2


    def plot(self, regression : bool = False, degree : int = 1):

        if self._merged_cpu_energy_df is None:
            self._merged_cpu_energy_df = self.merge_data()
            
        cpu_data = self._merged_cpu_energy_df["cpu_util"].to_list()
        energy_data = self._merged_cpu_energy_df["watts"].to_list()

        plt.scatter(cpu_data, energy_data)

        if regression:
            poly_coefs, r_2 = self.compute_poly_regression(degree)
            poly_fn = np.poly1d(poly_coefs)

            plt.plot(cpu_data, poly_fn(cpu_data), color="red", label=f"Degree {degree} Polynomial Fit (R^2 = {r_2:.2f})")
            plt.legend()

        plt.xlabel("CPU Utilization (%)")
        plt.ylabel("Energy (Watts)")
        plt.show()

    def export_to_json_dict(self, bin_count : int = 10, regression_degree : int = 1):
    
        bin_ordering, bin_data = self.compute_cpu_data(bin_count)
        coefs, r_2 = self.compute_poly_regression(regression_degree)

        json_dict : Dict[str, Any] = {}
        json_dict = {
            "bin_ordering" : bin_ordering,
            "cpu_bins" : bin_data,
            "regression" : {
                "coefs" : coefs.tolist(),
                "r_2" : r_2
            }
        }

        return json_dict
    
    def export_merged_to_csv(self):

        cpu_name = "cpu"
        energy_name = "energy"

        if self._cpu_path is not None:
            cpu_name = Path(self._cpu_path).stem
        if self._energy_path is not None:
            energy_name = Path(self._energy_path).stem

        merged_path = f"merged_{cpu_name}_{energy_name}.csv"

        if self._merged_cpu_energy_df is None:
            self._merged_cpu_energy_df = self.merge_data()
        self._merged_cpu_energy_df.to_csv(merged_path, index=False)

    def _compute_cpu_utilizations(self):

        last_busy_cycles = 0
        last_total_cycles = 0

        new_cpu_df = pd.DataFrame(columns=["collection_time", "cpu_util"])

        for row in self._cpu_df.itertuples():
            busy_cycles = row.user + row.nice + row.system
            total_cycles = busy_cycles + row.idle

            if last_busy_cycles == 0 and last_total_cycles == 0:
                last_busy_cycles = busy_cycles
                last_total_cycles = total_cycles
                continue

            collection_time = row.seconds
            cpu_util = (busy_cycles - last_busy_cycles) / (total_cycles - last_total_cycles)
            cpu_util = cpu_util * 100

            new_row = pd.DataFrame({"collection_time": [collection_time], "cpu_util": [cpu_util]})
            new_cpu_df = pd.concat([new_cpu_df, new_row], ignore_index=True, sort=False)

            last_busy_cycles = busy_cycles
            last_total_cycles = total_cycles

        self._cpu_df = new_cpu_df

    def _convert_raw_cpu_to_csv(self) -> pd.DataFrame:

        if self._raw_cpu_path is None:
            raise ValueError("Must set raw CPU data path before converting to CSV")

        raw_path = Path(self._raw_cpu_path)

        if raw_path.suffix == ".csv":
            return pd.read_csv(self._raw_cpu_path)
        if raw_path.suffix != ".ssv":
            raise ValueError("CPU data file extension must be .ssv or .csv")

        out_path = Path(".", raw_path.stem + ".csv")

        if out_path.exists():
            print(f"Removing existing file: {out_path.as_posix()}")
            os.remove(out_path.as_posix())

        with open(self._raw_cpu_path, "r") as in_file, open(out_path.as_posix(), "w") as out_file:
            for line in in_file:
                csv_line = ",".join(line.split())
                out_file.write(csv_line + "\n")

            print("Successfully converted from SSV to CSV")

        return pd.read_csv(out_path)
    
    def _convert_raw_energy_to_csv(self) -> pd.DataFrame:

        if self._raw_energy_path is None:
            raise ValueError("Must set raw Energy data path before converting to CSV")

        raw_path = Path(self._raw_energy_path)

        if raw_path.suffix == ".csv":
            return pd.read_csv(raw_path.as_posix())
        if raw_path.suffix != ".txt" and raw_path.suffix != ".text":
            raise ValueError("Energy data file extension must be .txt, .text, or .csv")

        out_path = Path(".", raw_path.stem + ".csv")
        key_columns = ["collection_time", "amps", "volts", "watts"]

        if out_path.exists():
            print(f"Removing existing file: {out_path.as_posix()}")
            os.remove(out_path.as_posix())

        def json_to_csv_line(json_line : str) -> str:
            json_dict = json.loads(json_line)
            values = [str(json_dict[k]) for k in key_columns]
            return ",".join(values) + "\n"

        with open(raw_path.as_posix(), "r") as file, open(out_path.as_posix(), "w") as out:
            out.write(",".join(key_columns) + "\n")
            for json_line in file:
                out.write(json_to_csv_line(json_line))

        print("Successfully converted from JSON Text to CSV")


        return pd.read_csv(out_path.as_posix())
    
    def __add__(self, other : "CpuEnergyPair") -> "CpuEnergyPair":
        
        new_cpu_df = pd.concat([self._cpu_df, other._cpu_df], ignore_index=True, sort=False)
        new_energy_df = pd.concat([self._energy_df, other._energy_df], ignore_index=True, sort=False)

        return CpuEnergyPair(cpu_df=new_cpu_df, energy_df=new_energy_df)