from data_preprocess import preprocess_data
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def feature_engineer():
    data = preprocess_data()
    # Label Encoding
    '''le = LabelEncoder()
    le_count = 0
    # Iterate through the columns
    for col in data:
        if data[col].dtype == 'object':
            # If 2 or fewer unique categories
            if len(list(data[col].unique())) <= 2:
                le.fit(data[col])
                data[col] = le.transform(data[col])
                le_count += 1     
    print('%d columns were label encoded.' % le_count)'''

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
