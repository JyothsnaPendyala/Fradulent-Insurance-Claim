from data_extraction import extract_data

def analyze_data():
    data = extract_data()
    print("Analysis on Data:")
    print(data.describe())
    print(data.info())
    print("Features in the data:", data.columns)
    print(data.isnull().sum())
    categorical_features = data.select_dtypes("object").columns
    numerical_features = data.select_dtypes("number").columns
    print("Categorical: ",categorical_features)
    print("Numerical: ",numerical_features)
    return data, categorical_features, numerical_features

analyze_data()