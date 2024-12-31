

import os
import json

def load_and_parse_json(file_path, debug=False):
    if debug:
        print(f"Debug: Loading and parsing file: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            data = json.load(f)
        
        trial_id = data['trial_id']
        result = 1 if data['result'] == 'made' else 0
        landing_x = data['landing_x']
        landing_y = data['landing_y']
        entry_angle = data['entry_angle']
        release_frame = data.get('release_frame', None)

        if debug:
            print(f"Debug: Trial ID: {trial_id}, Result: {result}, Release Frame: {release_frame}")
        
        return data, trial_id, result, landing_x, landing_y, entry_angle, release_frame
    except Exception as e:
        print(f"Error: Failed to load or parse JSON file: {file_path}. Exception: {e}")
        return None, None, None, None, None, None, None

def load_single_ft_and_parse(file_path, debug=False):
    data = load_and_parse_json(file_path, debug=debug)
    return data

# Test the function
if __name__ == "__main__":
    test_file = "../../SPL-Open-Data/basketball/freethrow/data/P0001/BB_FT_P0001_T0001.json"
    load_single_ft_and_parse(test_file, debug=False)
