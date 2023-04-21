from cpu_energy_pair import CpuEnergyPair

from typing import List, Tuple, Dict, Optional, Any
import sys
import json


def main(cpu_energy_path_pairs : List[Tuple[str, str]]):
    
    cpu_energy_pairs : List[CpuEnergyPair] = []
    for cpu_path, energy_path in cpu_energy_path_pairs:
        cpu_energy_pairs.append(CpuEnergyPair(cpu_path, energy_path))


    total_cpu_energy_data : Optional[CpuEnergyPair] = None

    regression_degree = 1
    bin_count = 10

    json_dict : Dict[str, Dict[str, Any]] = {}
    
    for pair_idx in range(len(cpu_energy_pairs)):
        cpu_energy_pair = cpu_energy_pairs[pair_idx]
        cpu_energy_pair.apply_moving_average()
        cpu_energy_pair.merge_data()

        bin_ordering, bin_data = cpu_energy_pair.compute_cpu_data(bin_count)

        json_dict[f"pair_{pair_idx+1}"] = {
            "bin_ordering" : bin_ordering,
            "cpu_bins" : bin_data
        }

        coefs, r_2 = cpu_energy_pair.compute_poly_regression(regression_degree)
        reg_dict = {
            "coefs" : coefs.tolist(),
            "r_2" : r_2
        }
        json_dict[f"pair_{pair_idx+1}"]["regression"] = reg_dict

        if total_cpu_energy_data is None:
            total_cpu_energy_data = cpu_energy_pair
            continue
        total_cpu_energy_data += cpu_energy_pair

    total_cpu_energy_data.merge_data()

    bin_ordering, bin_data = total_cpu_energy_data.compute_cpu_data(bin_count)
    coefs, r_2 = total_cpu_energy_data.compute_poly_regression(regression_degree)

    json_dict["all_data"] = {
        "bin_ordering" : bin_ordering,
        "cpu_bins" : bin_data,
        "regression" : {
            "coefs" : coefs.tolist(),
            "r_2" : r_2
        }
    }

    if len(cpu_energy_pairs) > 1:
        file_path = f"cpu_energy_data_1_{len(cpu_energy_pairs)}.json"
    else:
        file_path = "cpu_energy_data.json"

    with open(file_path, "w") as f:
        json.dump(json_dict, f, indent=4)

    total_cpu_energy_data.plot(regression=True, degree=regression_degree)



if __name__ == "__main__":

    args = sys.argv[1:]

    if len(args) < 2:
        print("Usage: python3 compute_cpu_energy.py <cpu.csv|cpu.ssv> <energy.txt|energy.csv> * n")
        exit(1)

    if len(args) % 2 != 0:
        print("Usage: python3 compute_cpu_energy.py <cpu.csv|cpu.ssv> <energy.txt|energy.csv> * n")
        exit(1)

    path_pairs : List[Tuple[str, str]] = []

    for path_idx in range(0, len(args), 2):
        cpu_path = args[path_idx]
        energy_path = args[path_idx + 1]
        path_pairs.append((cpu_path, energy_path))

    main(path_pairs)

