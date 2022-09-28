import sys

import argparse

# Initiate the parser
parser = argparse.ArgumentParser()

# Add long and short argument
parser.add_argument("--dataset", type=str, help="set the dataset name")
parser.add_argument("--num_runs", type=float, help="set the number of trials")

# Read arguments from the command line
args = parser.parse_args()


def main():
    print("the dataset name name is", args.dataset)
    print("the number of runs is", args.num_runs)

if __name__ == "__main__":
    main()