import json
import os
import sys


file_path = sys.argv[1]
file_name, ext = os.path.splitext(file_path)
key_columns = ["collection_time", "amps", "volts", "watts"]

def json_to_csv_line(json_line : str) -> str:
    json_dict = json.loads(json_line)
    values = [str(json_dict[k]) for k in key_columns]
    return ",".join(values) + "\n"

out = open(file_name + ".csv", "a")

with open(file_path, "r") as file:
    out.write(",".join(key_columns) + "\n")
    for json_line in file:
        out.write(json_to_csv_line(json_line))

out.close()