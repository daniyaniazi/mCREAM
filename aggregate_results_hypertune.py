import os
import csv
import pandas as pd
import argparse
from pathlib import Path
from collections import defaultdict


def find_and_aggregate_csvs(parent_folder: Path, output_file: Path) -> pd.DataFrame:
    """Works only for csv files with the same columns. Some runs have different hyperparams so this wont work!"""
    # Store all data from CSVs
    all_data = []
    headers = None

    # Walk through all subdirectories
    for root, dirs, files in os.walk(parent_folder):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)

                # Read each CSV file
                with open(file_path, "r") as csvfile:
                    reader = csv.reader(csvfile)
                    file_headers = next(reader)  # Get headers
                    try:
                        row = next(reader)  # Get values

                        # Store headers from first file
                        if headers is None:
                            headers = file_headers

                        # Check if headers match
                        if headers == file_headers:
                            all_data.append(row)
                    except StopIteration:
                        print(f"Warning: Empty file found at {file_path}")

    df = pd.DataFrame(all_data, columns=headers)
    # df.sort_values(by="val_accuracy")
    df.to_csv(output_file)
    # # Write aggregated data to new CSV
    # with open(output_file, "w", newline="") as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(headers)  # Write headers
    #     writer.writerows(all_data)  # Write all rows

    # Read and display using pandas
    df = pd.read_csv(output_file)
    return df


def find_and_aggregate_csvs_diff(parent_folder: Path, output_dir: Path) -> dict:
    """
    Recursively finds all CSV files in `parent_folder`, groups them by unique column structure, aggregates each group,
    and saves them to `output_dir` as separate CSV files; returns a dictionary of aggregated DataFrames by column structure.
    """

    # Dictionary to store dataframes grouped by unique columns
    dataframes_by_columns = defaultdict(list)

    # Walk through all subdirectories
    for root, dirs, files in os.walk(parent_folder):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)

                # Read each CSV file
                try:
                    # Use pandas to read the file and get headers
                    df = pd.read_csv(file_path)
                    column_tuple = tuple(df.columns)  # Use columns as a key
                    dataframes_by_columns[column_tuple].append(df)
                except pd.errors.EmptyDataError:
                    print(f"Warning: Empty file found at {file_path}")

    experiment_type = Path(parent_folder).parts[-2]
    output_dir = Path(output_dir) / experiment_type
    output_dir.mkdir(exist_ok=True)

    # Save each group of dataframes with matching columns into a separate CSV file
    aggregated_dataframes = {}
    counter = 1  # file counter

    for columns, dfs in dataframes_by_columns.items():
        # Concatenate all dataframes with the same columns
        combined_df = pd.concat(dfs, ignore_index=True)
        aggregated_dataframes[columns] = combined_df

        # Generate a filename based on version and counter
        output_file = output_dir / f"aggregated_{counter}.csv"

        combined_df.to_csv(output_file, index=False)
        counter += 1
        print(f"Aggregated file saved: {output_file}")

    return aggregated_dataframes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument(
        "--result_folder",
        type=str,
        required=True,
        help="Folder where all the csv (results) are stored",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        help="Path to the generated aggregated CSV file",
        default="aggregated_results",
    )
    args = parser.parse_args()

    # Usage
    df = find_and_aggregate_csvs_diff(args.result_folder, Path(args.output_dir))
    print("\nAggregated Data:")
    print(df)

    # Optional: Display some basic statistics
    # print("\nBasic Statistics:")
    # print(df.describe())
