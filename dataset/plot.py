import pandas as pd
import matplotlib.pyplot as plt
import sys

def plot_time_series(file_path, row_index, column_name):
    try:
        df = pd.read_csv(file_path, sep=';')
        data_raw = df.at[row_index, column_name]
        
        # Clean the string and split into a list
        raw_list = data_raw.strip('[]').split(',')
        
        processed_values = []
        special_indices = [] # To track indices of None or negative values

        for i, x in enumerate(raw_list):
            val_str = x.strip().replace("'", "").replace('"', "")
            
            # Check for None or non-numeric strings
            if val_str.lower() == 'none' or val_str == '':
                val = 0.0
                special_indices.append(i)
            else:
                try:
                    val = float(val_str)
                    if val < 0:
                        val = 0.0
                        special_indices.append(i)
                except ValueError:
                    val = 0.0
                    special_indices.append(i)
            
            processed_values.append(val)

        # Plotting
        plt.figure(figsize=(12, 6))
        
        # Plot the main series
        plt.plot(processed_values, color='blue', label='Valid Data', zorder=1)
        
        # Highlight "Special" (None/Negative) points in Red
        if special_indices:
            special_values = [processed_values[i] for i in special_indices]
            plt.scatter(special_indices, special_values, color='red', 
                        label='None/Negative (Set to 0)', zorder=2)

        plt.title(f"Time Series: Row {row_index}, Column '{column_name}'")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Usage: python script.py data.csv 0 "series_column"
    path = r"C:\Users\giaco\Documents\SCUOLA\Magistrale\25-26\BALANCE\dataset\group1_combined.csv"
    row = 11
    col = "stress_time_series"
    plot_time_series(path, row, col)