import itertools
import pandas as pd
import sys 
sys.path.append('../main_experiment')
from config import NUM_LAYERS_DICT

def convert_settings_to_numeric(df, colunmns2convert=['encoding_setting', 'response_setting']):
    """
    Converts settings values like 'MLP_128' to 128, if possible, across all DataFrame columns.
    
    Args:
    df (pandas.DataFrame): The DataFrame to process.
    
    Returns:
    pandas.DataFrame: A DataFrame with converted values.
    """
    # Iterate over each column in the DataFrame
    for column in colunmns2convert:
        # Apply conversion to each cell in the column
        df[column] = df[column].apply(lambda x: convert_value(x))
    return df


def convert_value(value):
    """
    Convert a single value to an integer if it follows the pattern 'XXX_123'.
    
    Args:
    value (str): The value to convert.
    
    Returns:
    str or int: The converted value or the original value if conversion is not applicable.
    """
    if pd.isnull(value):
        return value
    if isinstance(value, str) and '_' in value:
        parts = value.split('_')
        # Applicable to PCA_128, MLP_64, LLAMA7B_6 etc. 
        if len(parts) == 2 and parts[1].isdigit():  # Only convert if there are exactly two parts and the second is a digit
            return int(parts[1])  # Convert the second part to an integer
        # For whisper models
        if parts[0].startswith('whisper'):
            if parts[1].isdigit():
                return int(parts[1])
            elif parts[1] == 'final' :
                return int(NUM_LAYERS_DICT[parts[0]]-1) #-1 because labels start from 0, but saved data includes 0 (i.e. 0~32 becomes 33!)
        
    return value  # Return the original value if not convertible

"""

# Example usage:
data = {
    'encoding_setting': ['whisper-large_final_default_default', 'whisper-large_2_default_default', 'MLP_512', 'MLP_128_128', 'PCA_8192', 'PCA_1024', None],
    'response_setting': ['default', 'LLAMA7B_6', 'MLP_32', 'MLP_256', 'MLP_16', 'MLP_128', 'MLP_128_128']
}

df = pd.DataFrame(data)
converted_df = convert_settings_to_numeric(df)
print(converted_df)

"""


def filter_groups_by_columns(df, column_pair):
    """Filter DataFrame groups based on a pair of columns, returning only those groups with more than one row."""
    grouped = df.groupby(list(column_pair))
    filtered = {name: group for name, group in grouped if len(group) > 1}
    return filtered

def analyze_custom_settings_combinations(df, settings):
    """Analyze combinations of settings from a custom list, grouping and identifying multi-row groups for each pair."""
    # Generate all possible combinations of pairs from the settings list
    column_pairs = list(itertools.combinations(settings, 2))
    
    # Dictionary to hold results
    results = {}
    
    # Iterate over each pair and process
    for pair in column_pairs:
        result = filter_groups_by_columns(df, pair)
        results[f"{pair[0]}__{pair[1]}"] = result
    
    return results

def filter_sort_and_convert_to_numeric(df, filter_condition : dict, exclude_condition = None,multimodal = False, sort_by = 'response_setting', colunmns2convert=['encoding_setting', 'response_setting']):
    df2 = df.copy()
    if not multimodal:
        df2 = df2[df2['stimulus_setting_2'].isnull()]
    if exclude_condition is not None:
        for key, value in exclude_condition.items():
            df2 = df2[df[key] != value]
    for key, value in filter_condition.items():
        df2 = df2[df2[key] == value]
    if colunmns2convert is not None:
        return convert_settings_to_numeric(df2, colunmns2convert=colunmns2convert).sort_values(by=sort_by)
    return df2.sort_values(by=sort_by)

"""
# Exam1e usage
# Assuming 'final_results' is your DataFrame and it has been loaded appropriately
settings_list = ['stimulus_setting', 'encoding_setting', 'response_setting', 'stimulus_setting_2']
results = analyze_custom_settings_combinations(final_results, settings_list)

# To display or process the results
for key, value in results.items():
    print(key)
    for group_key, group_df in value.items():
        print(f"  Group {group_key} with {len(group_df)} entries")
        # Here you can save, display or further analyze group_df
"""