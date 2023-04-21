import sys
import os

def main(file_path : str):
    in_path = file_path_str
    out_path = os.path.splitext(in_path)[0] + ".csv"

    in_file = open(in_path, "r")
    out_file = open(out_path, "w")
    try:
        for line in in_file:
            csv_line = ",".join(line.split())
            out_file.write(csv_line + "\n")

        print("Successfully converted from SSV to CSV")
    finally:
        in_file.close()
        out_file.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Did not supply a file input path!")
        exit(1)
    
    file_path_str = sys.argv[1]
    main(file_path_str)

