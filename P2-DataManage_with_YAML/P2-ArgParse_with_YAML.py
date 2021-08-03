from common.parser import parse_yaml
import argparse

# Define parser
parser = argparse.ArgumentParser()

# Set arguments
parser.add_argument('--dataset', type=str, default="./configs/datasets/people_dataset.yaml")
args = parser.parse_args()

# Parse yaml files
dataset_option = parse_yaml(args.dataset)

# Use arguments
dataset_name = dataset_option["NAME"]
dataset_path = dataset_option["PATH"]
