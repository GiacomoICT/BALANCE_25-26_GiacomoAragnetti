import pandas as pd
import numpy as np
import glob
import json
import ast

# --- CONFIGURATION ---
FILE_PATTERN = 'dataset/group*_combined.csv'

# Columns specifically mentioned in your snippet for Log Transformation
COLS_TO_LOG = [
    'act_distance', 
    'sleep_sleepTimeSeconds', 
    'sleep_deepSleepSeconds'
]

def clean_time_series_feature(series_str, target_length=720, min_quality_ratio=0.5):
    """
    Standardizes time series:
    1. Parses Python-style strings (fixing the 'None' issue).
    2. Replaces negatives/None with NaN.
    3. Rejects series that are too short (<50% data) to avoid distortion.
    4. Interpolates small gaps and resamples to the exact target grid.
    """
    try:
        # 1. PARSE: Use ast.literal_eval to handle strings like "[None, 60, 62]"
        if pd.isna(series_str) or str(series_str).strip() in ["", "[]"]:
            return [0.0] * target_length
        
        # This fixes the Group 7 "-1 length" error
        data = np.array(ast.literal_eval(str(series_str)))

        # 2. INITIAL CLEANING: Convert to float and mask bad data
        data = data.astype(float)
        data[data < 0] = np.nan # Fixes Problem #2 (negatives)
        
        # 3. QUALITY THRESHOLD: Avoid the "407 problem"
        # If we have less than 50% of the data, stretching it creates "fake" trends.
        valid_count = np.count_nonzero(~np.isnan(data))
        if valid_count < (target_length * min_quality_ratio):
            # Return daily average if possible, otherwise zeros
            fill_val = np.nanmean(data) if valid_count > 0 else 0.0
            return [float(fill_val)] * target_length

        # 4. INTERPOLATION: Fill internal gaps (Problem #3)
        s = pd.Series(data)
        # We limit internal interpolation to 10 points (~20-30 mins) 
        # to ensure we aren't "hallucinating" values across massive gaps.
        data_cleaned = s.interpolate(method='linear', limit=10).ffill().bfill().values

        # 5. RESAMPLING: Stretch or shrink to target (Fixes 719 vs 720 and 480 vs 720)
        # This maps whatever length we have (e.g. 480 or 719) onto the target grid.
        current_indices = np.linspace(0, 1, len(data_cleaned))
        target_indices = np.linspace(0, 1, target_length)
        
        resampled_data = np.interp(target_indices, current_indices, data_cleaned)
        
        return resampled_data.tolist()

    except Exception:
        # Fallback for completely malformed rows
        return [0.0] * target_length

def apply_ts_cleaning(df):
    """
    Applies cleaning to the specific TS columns in your dataset.
    """
    ts_cols = ['hr_time_series', 'resp_time_series', 'stress_time_series'] # Adjusted for your labels
    
    # We will target 720 samples (1 sample every 2 mins for a 24h period)
    # You can change this to 1440 for 1-min resolution.
    TARGET_SAMPLES = 720 
    i = 0
    for col in ts_cols:
        if col in df.columns:
            print(f"      -> Cleaning and resampling {col}...")
            if (i >= 2):
                TARGET_SAMPLES = 480
            df[col] = df[col].apply(lambda x: clean_time_series_feature(x, TARGET_SAMPLES))
            i += 1
            
    return df

def add_circadian_features_safe(df):
    """Calculates the difference between average Day HR and Night HR."""
    def get_circadian_delta(ts):
        ts = np.array(ts)
        if np.all(ts == 0) or np.all(ts == ts[0]):
            return None, 0
        
        day_avg = np.mean(ts[0:480])
        night_avg = np.mean(ts[480:720])
        return (day_avg - night_avg), 1

    if 'hr_time_series' in df.columns:
        res = df['hr_time_series'].apply(get_circadian_delta)
        df['ts_hr_circadian_delta'] = [r[0] for r in res]
        # We can reuse the existing hr_ts_is_reliable flag if you have it
        
        df['ts_hr_circadian_delta'] = df['ts_hr_circadian_delta'].fillna(df['ts_hr_circadian_delta'].mean())
    return df

def add_sleep_onset_features_safe(df):
    """Measures respiratory stability during the first 2 hours of sleep."""
    def get_onset_stability(ts):
        ts = np.array(ts)
        if np.all(ts == 0) or np.all(ts == ts[0]):
            return None, 0
        
        # Segment: first 120 minutes of sleep
        onset_window = ts[480:540]
        return np.std(np.diff(onset_window)), 1

    if 'resp_time_series' in df.columns:
        res = df['resp_time_series'].apply(get_onset_stability)
        df['ts_resp_onset_instability'] = [r[0] for r in res]
        
        df['ts_resp_onset_instability'] = df['ts_resp_onset_instability'].fillna(df['ts_resp_onset_instability'].mean())
    return df

def add_stress_peak_features_safe(df):
    """Counts the number of high-stress events (>75)."""
    def count_peaks(ts):
        ts = np.array(ts)
        if np.all(ts == 0):
            return None, 0
        
        # Count samples where stress > 75
        peaks = np.sum(ts > 75)
        return peaks, 1

    if 'stress_time_series' in df.columns:
        res = df['stress_time_series'].apply(count_peaks)
        df['ts_stress_peaks'] = [r[0] for r in res]
        
        # For a count, filling with the median is often safer than the mean
        df['ts_stress_peaks'] = df['ts_stress_peaks'].fillna(df['ts_stress_peaks'].median())
    return df

def add_resp_stability_features_safe(df):
    def get_stability(ts):
        ts = np.array(ts)
        # Check if the cleaning script rejected this (flat line)
        if np.all(ts == ts[0]) or np.all(ts == 0):
            return None, 0
        return np.std(np.diff(ts)), 1

    if 'resp_time_series' in df.columns:
        res = df['resp_time_series'].apply(get_stability)
        df['ts_resp_instability'] = [r[0] for r in res]
        df['resp_ts_is_reliable'] = [r[1] for r in res] # Shared flag for Resp
        
        # Fill with mean of valid rows
        df['ts_resp_instability'] = df['ts_resp_instability'].fillna(df['ts_resp_instability'].mean())
    return df


def add_stress_load_features_safe(df):
    def get_auc(ts):
        ts = np.array(ts)
        if np.all(ts == 0) or np.all(ts == ts[0]):
            return None, 0
        return np.trapezoid(ts), 1

    if 'stress_time_series' in df.columns:
        res = df['stress_time_series'].apply(get_auc)
        df['ts_stress_load'] = [r[0] for r in res]
        df['stress_ts_is_reliable'] = [r[1] for r in res] # Shared flag for Stress
        
        df['ts_stress_load'] = df['ts_stress_load'].fillna(df['ts_stress_load'].mean())
    return df


def add_hr_recovery_features_safe(df):
    """Calculates HR slope with a fallback to a neutral mean + a quality flag."""
    
    def get_slope_and_quality(ts):
        ts = np.array(ts)
        # If the series was rejected by the cleaning script (all same values/zeros)
        if np.all(ts == ts[0]) or np.all(ts == 0):
            return None, 0  # Mark as NaN and Unreliable
        
        # Calculate the actual slope
        recovery_segment = ts[-60:]
        x = np.arange(len(recovery_segment))
        slope, _ = np.polyfit(x, recovery_segment, 1)
        return slope, 1  # Mark as Success and Reliable

    if 'hr_time_series' in df.columns:
        # 1. Apply logic to get both values
        results = df['hr_time_series'].apply(get_slope_and_quality)
        df['ts_hr_recovery_slope'] = [r[0] for r in results]
        df['ts_hr_is_reliable'] = [r[1] for r in results]
        
        # 2. UNBIASED FILL: Fill the NaNs with the mean of the 'good' data
        # This prevents 0.0 bias.
        slope_mean = df['ts_hr_recovery_slope'].mean()
        df['ts_hr_recovery_slope'] = df['ts_hr_recovery_slope'].fillna(slope_mean)
        
    return df


def add_coupling_features_safe(df):
    def get_coupling(row):
        hr = np.array(row['hr_time_series'])
        str_ts = np.array(row['stress_time_series'])
        
        # Check if sensors are empty/zeroed out
        if np.all(hr == 0) or np.all(str_ts == 0):
            return None, 0
        
        # Check for zero variance (Flat Lines)
        # np.ptp (peak-to-peak) is 0 if all values are the same
        if np.ptp(hr) == 0 or np.ptp(str_ts) == 0:
            return None, 0 # Constant signal provides no coupling info

        # Resample Stress to match HR length
        str_resampled = np.interp(np.linspace(0, 1, 720), 
                                   np.linspace(0, 1, len(str_ts)), 
                                   str_ts)
        
        # Correlation calculation is now safe from division by zero
        corr = np.corrcoef(hr, str_resampled)[0, 1]
        
        # Check if corr itself is NaN (final safety)
        if np.isnan(corr):
            return None, 0
            
        return corr, 1

    if 'hr_time_series' in df.columns and 'stress_time_series' in df.columns:
        print("      -> Calculating HR-Stress Coupling...")
        res = df.apply(get_coupling, axis=1)
        df['ts_hr_stress_coupling'] = [r[0] for r in res]
        df['ts_system_coupling_reliable'] = [r[1] for r in res]
        
        # Use 0.5 (Neutral) as fallback
        df['ts_hr_stress_coupling'] = df['ts_hr_stress_coupling'].fillna(0.5)
    return df

def add_engineered_features(df):
    """
    Applies the specific logic from your snippet:
    1. Log transforms distance & sleep.
    2. Creates the Activity Intensity Ratio.
    """
    
    # 1. LOG TRANSFORM
    # We create NEW columns prefixed with 'log_' to keep both versions
    for col in COLS_TO_LOG:
        if col in df.columns:
            # np.log1p computes log(1 + x) to avoid log(0) errors
            df[f'log_{col}'] = np.log1p(df[col])
        else:
            print(f"   Warning: {col} not found, skipping log transform.")

    # 2. CREATE RATIOS (Universal Metrics)
    # Intensity: What % of total calories were active?
    if 'act_activeKilocalories' in df.columns and 'act_totalCalories' in df.columns:
        # Avoid division by zero by adding 1.0
        df['act_intensity_ratio'] = df['act_activeKilocalories'] / (df['act_totalCalories'] + 1.0)
    else:
        print("   Warning: Calorie columns missing, skipping intensity ratio.")

    # Note: Your snippet dropped the raw calorie columns. 
    # I have commented this out so you don't lose data, but you can uncomment to apply it.
    # df.drop(columns=['act_activeKilocalories', 'act_totalCalories'], inplace=True)
    
    return df

def process_csvs():
    files = glob.glob(FILE_PATTERN)
    
    if not files:
        print(f"No files found matching: {FILE_PATTERN}")
        return

    print(f"Found {len(files)} files. Starting feature engineering...")

    for filepath in files:
        try:
            print(f"Processing {filepath}...")
            
            # Load with your standard settings (semicolon separator)
            df = pd.read_csv(filepath, sep=';', on_bad_lines='skip')
            
            # Apply the engineering
            df = add_engineered_features(df)
            
            # Save to a new file to avoid overwriting the original
            new_filename = filepath.replace('.csv', '_with_features.csv')
            df.to_csv(new_filename, index=False, sep=';')
            
            print(f"  -> Saved to {new_filename}")

        except Exception as e:
            print(f"  Error processing {filepath}: {e}")


def add_advanced_features(df):
    # 1. RATIO: Deep Sleep Efficiency
    # (Deep Sleep / Total Sleep)
    # We sum the components to get a true "Total Sleep"

    df=add_engineered_features(df)

    if 'sleep_deepSleepSeconds' in df.columns and 'sleep_lightSleepSeconds' in df.columns:
        total_sleep = (df['sleep_deepSleepSeconds'] + 
                       df['sleep_lightSleepSeconds'] + 
                       df.get('sleep_remSleepSeconds', 0)) # Handle if REM is missing
        
        # Avoid division by zero
        df['ratio_deep_sleep'] = np.where(total_sleep > 0, 
                                          df['sleep_deepSleepSeconds'] / total_sleep, 
                                          0)

    # 2. RATIO: Heart Rate Recovery Index
    # (Resting HR / Average HR) -> Lower is usually better (efficient heart)
    if 'hr_restingHeartRate' in df.columns and 'sleep_avgHeartRate' in df.columns:
        df['ratio_hr_rest_to_avg'] = np.where(df['sleep_avgHeartRate'] > 0,
                                              df['hr_restingHeartRate'] / df['sleep_avgHeartRate'],
                                              1.0)

    # 3. DELTA: Relative Resting Heart Rate
    # (Today's HR - My Average HR)
    # This removes the "Group 0 vs Group 7" bias completely.
    if 'hr_restingHeartRate' in df.columns:
        my_average_hr = df['hr_restingHeartRate'].mean()
        df['delta_hr_resting'] = df['hr_restingHeartRate'] - my_average_hr


    # 4. DELTA: Relative Stress
    # (Today's Stress - My Average Stress)
    if 'str_avgStressLevel' in df.columns:
        my_average_stress = df['str_avgStressLevel'].mean()
        df['delta_stress'] = df['str_avgStressLevel'] - my_average_stress

    df = apply_ts_cleaning(df)

    df = add_resp_stability_features_safe(df)
    df = add_stress_load_features_safe(df)
    df = add_hr_recovery_features_safe(df)
    df = add_coupling_features_safe(df)

    df = add_circadian_features_safe(df)
    df = add_sleep_onset_features_safe(df)
    df = add_stress_peak_features_safe(df)

    return df

def process_csvs_new():
    files = glob.glob(FILE_PATTERN)
    if not files:
        print("No files found.")
        return

    print(f"Found {len(files)} files. Adding V2 features...")

    for filepath in files:
        try:
            # Load
            df = pd.read_csv(filepath, sep=';', on_bad_lines='skip')
            
            # Engineer
            df = add_advanced_features(df)
            
            # Save (Overwriting the original file)
            # Use the original filepath directly to replace the old file
            df.to_csv(filepath, index=False, sep=';')
            
            print(f"  -> Successfully updated and replaced: {filepath}")

        except Exception as e:
            print(f"  Error processing {filepath}: {e}")

def process_df(df):
    return add_advanced_features(df)

if __name__ == "__main__":
    process_csvs_new()




