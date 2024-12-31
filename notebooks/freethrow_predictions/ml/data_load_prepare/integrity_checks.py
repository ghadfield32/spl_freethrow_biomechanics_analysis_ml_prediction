

def check_data_integrity(df, debug=False):
    null_counts = df.isna().sum()
    
    if debug:
        print("\nDebug: Data integrity check - Missing counts for all columns:")
        print(null_counts)
    
    problematic_columns = null_counts[null_counts > 0]
    
    if not problematic_columns.empty:
        print("\nWarning: The following columns have missing data:")
        for col, count in problematic_columns.items():
            print(f"Column '{col}' has {count} missing values.")
    else:
        print("\nInfo: No columns with missing data detected.")
    
    return problematic_columns

    

# Test the function
if __name__ == "__main__":
    # from dataframe_creation import main_create_dataframe
    # from load_and_parse import load_single_ft_and_parse
    data, trial_id, result, landing_x, landing_y, entry_angle, _ = load_single_ft_and_parse("../../SPL-Open-Data/basketball/freethrow/data/P0001/BB_FT_P0001_T0001.json")
    df = main_create_dataframe(data, trial_id, result, landing_x, landing_y, entry_angle, debug=False)
    check_data_integrity(df, debug=False)
