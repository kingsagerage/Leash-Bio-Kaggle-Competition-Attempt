import pandas as pd
import numpy as np
import rdkit
import duckdb
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras.metrics import Precision
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

def prepare_dta(train_df,n_PCAs,ECFP_radius):
    def generate_ecfp(molecule, radius=ECFP_radius, bits=1024):
        if molecule is None:
            return None
        return list(AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=bits))


    train_df['molecule'] = train_df['molecule_smiles'].apply(Chem.MolFromSmiles)


    train_df['ecfp'] = train_df['molecule'].apply(generate_ecfp)


    one_hot_encoded = pd.get_dummies(train_df['protein_name'], prefix='Protein: ')
    one_hot_encoded = one_hot_encoded.astype(int)
    ndfn = train_df.drop('protein_name', axis=1)
    ndf = pd.concat([ndfn, one_hot_encoded], axis=1)

    # Convert the list column into separate columns
    expanded_df = pd.DataFrame(ndf['ecfp'].to_list(), columns=[f'ecfp_{i+1}' for i in range(1024)])

    # Combine the expanded DataFrame with the original DataFrame
    result_df = pd.concat([ndf, expanded_df], axis=1)

    # Drop the original 'ecfp' column
    result_df.drop(columns=['ecfp'], inplace=True)

    # Filter numeric columns
    numeric_columns = result_df.select_dtypes(include='number').columns

    # Keep only numeric columns
    df_numeric = result_df[numeric_columns]

    pc_list = []
    #Deciding on number of PCA features
    for i in range(1, n_PCAs+1):
        pc_list.append(f'PC{i}')

    #Prepping df for PCA
    dta_for_pca = df_numeric.drop(columns=['id','binds'])

    # Perform PCA
    pca = PCA(n_components = 400)  # Specify the number of components you want
    pca_result = pca.fit_transform(dta_for_pca)

    # Create a DataFrame to store the PCA results
    pca_df = pd.DataFrame(data=pca_result, columns=pc_list)
    explained_variance_ratio = pca.explained_variance_ratio_

    #Reasonable Variance explained (Hopefully :*) )
    variance_explained = explained_variance_ratio[:400].sum()

    #Readding ID and binds target class
    post_pca_df = pd.concat((pca_df,df_numeric[['id','binds']]),axis=1)

    # Split the data into train and test sets
    X = post_pca_df.drop(columns=['id','binds'])
    y = post_pca_df[['binds']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return variance_explained, X_train, X_test, y_train, y_test
