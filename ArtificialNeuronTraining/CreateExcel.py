import pandas as pd

def create_excel(feature_num, file_name, test_results):
    # Creates an Excel file with test results from model inference.

    # Generate column names: Feature_1, Feature_2, ..., Ground_Truth, Prediction
    feature_names = [f'Feature_{i+1}' for i in range(feature_num)]
    columns = feature_names + ['Ground_Truth', 'Prediction']

    # Create DataFrame with results and column names
    df = pd.DataFrame(test_results, columns=columns)
    
    # Export to Excel file without row indices
    df.to_excel(file_name, index=False)