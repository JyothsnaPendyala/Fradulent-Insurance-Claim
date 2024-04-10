from data_analysis import analyze_data
import pandas as pd

def preprocess_data():
    data, categorical_features, numerical_features = analyze_data()
    
    data = data.drop(['_c39','policy_number','auto_year','auto_model','auto_make',
                      'authorities_contacted','incident_location','policy_bind_date',
                      'incident_date','insured_occupation','incident_city','incident_state','insured_hobbies'],axis=1)
    cat_features = data.select_dtypes("object").columns
    data.replace('?', pd.NA, inplace=True)

    for feature in cat_features:
        mode_value = data[feature].mode()[0]
        data[feature].fillna(mode_value, inplace=True)

    for i in cat_features:
        print(data[i].value_counts())
    return data,cat_features

preprocess_data()