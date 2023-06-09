import numpy as np
from typing import Any, Dict, List
import sys
import json
from cpu_energy_pair_tester import CpuEnergyPairTester

def main(cpu_energy_json : str, merged_cpu_energy_csvs : List[str]):

    # Open the specified json file
    cpu_energy_dict : Dict[str, Any] = json.load(open(cpu_energy_json, "r"))

    #regression_dict = cpu_energy_dict["all_data_linear_reg"]
    
    total_tester = CpuEnergyPairTester(None, None)

    cpu_percents : List[float] = []
    watts_percents : List[float] = []

    # Iterate through each pair of CPU data and energy regression data
    for video_pair_name, pair_dict_data in cpu_energy_dict.items():
        if not video_pair_name.startswith("pair_"):
            continue

        pair_idx = int(video_pair_name.split("_")[1])

        # Create a CpuEnergyPairTester object for the current pair dict and merged csv
        pair_tester = CpuEnergyPairTester(pair_dict_data, merged_cpu_energy_csvs[pair_idx-1])

        # Get CPU utilization and watts stats
        (pair_rmse_cpu_error, pair_percent_error_cpu), pair_r_sq_cpu = pair_tester.get_utilization_stats()
        (pair_rmse_watts_error, pair_percent_error_watts), pair_r_sq_watts = pair_tester.get_watts_stats()

        cpu_percents.append(pair_percent_error_cpu)
        watts_percents.append(pair_percent_error_watts)

        # Output stats
        print(f"Video {pair_idx} CPU   RMSE: {pair_rmse_cpu_error:.3f}, {pair_percent_error_cpu:.2f}%, R^2: {pair_r_sq_cpu:.3f}")
        print(f"Video {pair_idx} Watts RMSE: {pair_rmse_watts_error:.3f}, {pair_percent_error_watts:.2f}%, R^2: {pair_r_sq_watts:.3f}")

        print("-"*100)

        # pair_tester.plot_cpu_utilization()
        # pair_tester.plot_watts()

        total_tester += pair_tester

    # Power Pi
    #total_tester.test_custom_reg([1.5778, 0.181 / 100])

    # Software Power Model for Linux Single Board Computers
    #total_tester.test_custom_reg([1.722, 0.01048])

    # A Power Model for Monitoring Environmental Parameters on the Edge
    #total_tester.test_custom_reg([1.6249, 0.0081]) Ard1
    #total_tester.test_custom_reg([1.8029, 0.0108]) Ard2

    # A Preliminary Study of the Energy Impact of Software in Raspberry Pi devices
    #total_tester.test_custom_reg([4.142, 2.1514 / 100])

    # Optionally generate CPU watts using the general regression model
    #total_tester.generate_cpu_watts(cpu_energy_dict["all_data"], test_poly_stds=True)

    (rmse_cpu_error, percent_error_cpu), r_sq_cpu = total_tester.get_utilization_stats()
    (rmse_watts_error, percent_error_watts), r_sq_watts = total_tester.get_watts_stats()

    print(f"Overall CPU   RMSE: {rmse_cpu_error:.3f}, {percent_error_cpu:.2f}%, R^2: {r_sq_cpu:.2f}")
    print(f"Overall Watts RMSE: {rmse_watts_error:.3f}, {percent_error_watts:.2f}%, R^2: {r_sq_watts:.2f}")
    print(f"Average CPU Percent: {np.mean(cpu_percents):.2f}%, Average Watts Percent: {np.mean(watts_percents):.2f}%")
    
    total_tester.plot_cpu_utilization()

    total_tester.plot_watts()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 test_cpu_energy_accuracy.py <cpu_energy.json> <merged_cpu_energy.csv> * n")
        exit(1)
    args = sys.argv[1:]
    cpu_energy_json_path = args[0]
    merged_cpu_energy_csv_paths = args[1:]
    if cpu_energy_json_path[-5:] != ".json":
        print("Usage: python3 test_cpu_energy_accuracy.py <cpu_energy.json> <merged_cpu_energy.csv> * n")
        exit(1)

    for path in merged_cpu_energy_csv_paths:
        if path[-4:] != ".csv":
            print("Usage: python3 test_cpu_energy_accuracy.py <cpu_energy.json> <merged_cpu_energy.csv> * n")
            exit(1)

    main(cpu_energy_json_path, merged_cpu_energy_csv_paths)