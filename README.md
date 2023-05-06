# Recorded CPU Util and Energy Data Merge and Test

Two different Python programs involved in the merging of CPU Utilization data and the testing of energy prediction algorithms

## CSV Fields
* `collection_time`: Fractional Unix Epoch
* `cpu_util`: CPU Utilization, 0-100%
* `watts`: Energy consumption in watts, float


## Compute CPU Energy

Usage: `python3 compute_cpu_energy.py <cpu.csv|cpu.ssv> <energy.txt|energy.csv> * n`

Arguments: 

* CPU Utilization File Options: 
    - CSV file with fields: `collection_time` and `cpu_util`
    - SSV File based on `/proc/stat` with `collection_time` preappended (`util_fetcher` outputs correct format)
* Energy Consumption File Options:
    - CSV file with fields: `collection_time` and `watts`
    - Text file, each line is a JSON dictionary generated from the use of [rdserial](https://github.com/rfinnie/rdserialtool/tree/main) (`rdserialtool ... --json > watts_file.txt`)

Description:

- Merges each pair of CPU and Energy files into a combined CSV with Wattage values for each CPU utilization value
- Appends all merged pair files together into CSV with all CPU and Wattage values
- Outputs a JSON file with CPU generation data, energy regression coefficents, and any additional data for each pair and all data

Example:
```
$ mkdir temp && cd temp
$ python3 compute_cpu_energy.py ../video1_cpu.ssv ../video1_energy.txt \
                                ../video2_cpu.csv ../video2_energy.csv \
                                ../video3_cpu.ssv ../video3_energy.txt
```

## Test CPU Energy Accuracy

Usage: `python3 test_cpu_energy_accuracy.py <cpu_energy.json> <merged_cpu_energy.csv> * n`

Arguments:
* CPU Energy JSON file
    - Output from `compute_cpu_energy.py`
    - Contains `n` "pair" dictionaries and an "all_data" dictionary
* Merged CPU Energy CSV files
    - Output from `compute_cpu_energy.py`
    - Properly merged CPU and Wattage CSV file
    - Must pass less than or equal to the amount of pairs in the CPU Energy JSON file (in order of pair index)

Description:
    - Computes the RMSE between the predicted CPU data (from JSON) and the actual CPU data (from CSV) for each pair
    - Computes the RMSE between the predicted Wattage data (from JSON) and the actual Wattage data (from CSV) for each pair
    - Computes the R^2 and RMSE Percent of Max Value for both CPU and Wattage for each pair
    - Computes above statistics for all merged data combined
    - Optional testing of optimal wattage noise period (slightly lower RMSE, O(n))
    - Allows for easy plotting, testing of different regression coefficents, 

Example:
```
$ cd temp
$ python3 test_cpu_energy_accuracy.py cpu_energy_data_1_3.json \
                                      merged_video1_cpu_video1_energy.csv \
                                      merged_video2_cpu_video2_energy.csv \
                                      merged_video3_cpu_video3_energy.csv
```
