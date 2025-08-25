import pandas as pd
import ssl
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

def fetch_transform(data_path: str):
    # fetch dataset 
    covertype = fetch_ucirepo(id=31) 
    
    # data (as pandas dataframes) 
    Features = covertype.data.features 
    targets = covertype.data.targets 

    targets = targets - 1 # lablels now go from 0-6

    raw_data = pd.concat([Features, targets], axis = 1)
    raw_data.to_csv(f'{data_path}/raw/raw.csv', index = False)

    scaler = StandardScaler()

    # transform data
    X_quantitative_features = scaler.fit_transform((Features.iloc[:, 0:10]))
    X_quantitative_features = pd.DataFrame(X_quantitative_features, columns=[covertype.variables['name'][0:10]])
    X_binary_features = Features.iloc[:, 10:]
    X = pd.concat([X_quantitative_features, X_binary_features], axis = 1)

    transformed_data = pd.concat([X, targets], axis = 1)
    transformed_data.to_csv(f'{data_path}/transformed/transformed_data.csv', index = False)

    return X, targets


if __name__ == "__main__":
    fetch_transform('/Users/jasonluo/Documents/Neural_Net_Stuff/NN1/data')