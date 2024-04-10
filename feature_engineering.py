from data_preprocess import preprocess_data
import pandas as pd
import numpy as np
from sklearn import preprocessing

def feature_engineer():
    data, categorical_features = preprocess_data()
    print(data['property_damage'].value_counts())
    print(data['police_report_available'].value_counts())
    # Label Encoding
    label_encoder = preprocessing.LabelEncoder() 
    for i in categorical_features:
        data[i]= label_encoder.fit_transform(data[i]) 
    # Outlier Detection
    outlier_columns = ['property_claim', 'total_claim_amount', 'umbrella_limit', 'policy_annual_premium']
    for col in outlier_columns:
        percentile25 = data[col].quantile(0.25)
        percentile75 = data[col].quantile(0.75)
        iqr = percentile75 - percentile25
        upper_limit = percentile75 + 1.5 * iqr
        lower_limit = percentile25 - 1.5 * iqr
        data[col] = np.where(
            data[col] > upper_limit,
            upper_limit,
            np.where(
                data[col] < lower_limit,
                lower_limit,
                data[col]
            ))
    # Balancing the data
    print(data['fraud_reported'].value_counts())
    notfraudulent_count, fradulent_count =data['fraud_reported'].value_counts()
    notfraudulent = data[data['fraud_reported'] == 0]
    fraudulent = data[data['fraud_reported'] == 1]
    fraudulent_over = fraudulent.sample(notfraudulent_count,replace=True)
    data_balanced = pd.concat([fraudulent_over,notfraudulent], axis=0)
    data_balanced['fraud_reported'].groupby(data_balanced['fraud_reported']).count()
    print(data_balanced['fraud_reported'].value_counts())
    data_balanced.to_csv('fraudulent_insurance_claim.csv',index=False)

feature_engineer()
