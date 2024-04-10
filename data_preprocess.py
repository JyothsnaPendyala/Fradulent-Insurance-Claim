from data_analysis import analyze_data

def preprocess_data():
    data, categorical_features, numerical_features = analyze_data()
    data = data.drop(['_c39','policy_number','auto_year','auto_model','auto_make',
                      'authorities_contacted','incident_location','policy_bind_date',
                      'incident_date','insured_occupation'],axis=1)
    return data

preprocess_data()