import pandas as pd
import os
import argparse

def compute_label_mean(file_path):
    """
    Takes in input a path to a csv file and computes the mean and the variance of the label
    column in the file.

    Was used to infer some characteristics and to judge the quality of predictions done on the 
    test without submitting by comparing first order moments to the average of the dataset
    """


    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        return

    try:
        # Load the CSV
        df = pd.read_csv(file_path)

        if 'label' in df.columns:
            label_mean = df['label'].mean()
            label_var = df['label'].var()
            print(f"The mean of the 'label' column is: {label_mean:.4f}")
            print(f"Var of pred col: {label_var:.4f}")
        else:
            print("Error: The column 'label' does not exist in this CSV.")
            print(f"Available columns: {list(df.columns)}")

    except Exception as e:
        print(f"An error occurred: {e}")

#compute_label_mean(r"C:\Users\giaco\Documents\SCUOLA\Magistrale\25-26\BALANCE\predictions.csv")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute mean and variance of the 'label' column in a CSV.")
    parser.add_argument("path", help="The file path to the CSV file.")

    print(str())

    args = parser.parse_args()
    print(str(args))
    compute_label_mean(args.path)