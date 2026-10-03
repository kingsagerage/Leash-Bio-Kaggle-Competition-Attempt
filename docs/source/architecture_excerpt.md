```
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

```

```
#Reading only 100,000 rows for the initial training and playground stuff
train_df = pd.read_csv(r"C:\dta_genes\train.csv", nrows=1000000)

```

```
#Converting to RDKit molecules
train_df['molecule'] = train_df['molecule_smiles'].apply(Chem.MolFromSmiles)

```

```
# Generate ECFPs
def generate_ecfp(molecule, radius=3, bits=1024):
    if molecule is None:
        return None
    return list(AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=bits))

train_df['ecfp'] = train_df['molecule'].apply(generate_ecfp)

```

```
train_df.head()

```

|       | **id** |                        **buildingblock1_smiles** | **buildingblock2_smiles** | **buildingblock3_smiles** |                                **molecule_smiles** | **protein_name** | **binds** |                                       **molecule** |                                          **ecfp** |
| ----: | -----: | -----------------------------------------------: | ------------------------: | ------------------------: | -------------------------------------------------: | ---------------: | --------: | -------------------------------------------------: | ------------------------------------------------: |
| **0** |      0 | C#CC[C@@H]\(CC(=O)O)NC(=O)OCC1c2ccccc2-c2ccccc21 |      C#CCOc1ccc(CN)cc1.Cl |   Br.Br.NCC1CCCN1c1cccnn1 |  C#CCOc1ccc(CNc2nc(NCC3CCCN3c3cccnn3)nc(N[C@@H]... |             BRD4 |         0 | \<rdkit.Chem.rdchem.Mol object at 0x000001C2661... | [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ... |
| **1** |      1 | C#CC[C@@H]\(CC(=O)O)NC(=O)OCC1c2ccccc2-c2ccccc21 |      C#CCOc1ccc(CN)cc1.Cl |   Br.Br.NCC1CCCN1c1cccnn1 |  C#CCOc1ccc(CNc2nc(NCC3CCCN3c3cccnn3)nc(N[C@@H]... |              HSA |         0 | \<rdkit.Chem.rdchem.Mol object at 0x000001C2661... | [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ... |
| **2** |      2 | C#CC[C@@H]\(CC(=O)O)NC(=O)OCC1c2ccccc2-c2ccccc21 |      C#CCOc1ccc(CN)cc1.Cl |   Br.Br.NCC1CCCN1c1cccnn1 |  C#CCOc1ccc(CNc2nc(NCC3CCCN3c3cccnn3)nc(N[C@@H]... |              sEH |         0 | \<rdkit.Chem.rdchem.Mol object at 0x000001C2661... | [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ... |
| **3** |      3 | C#CC[C@@H]\(CC(=O)O)NC(=O)OCC1c2ccccc2-c2ccccc21 |      C#CCOc1ccc(CN)cc1.Cl |         Br.NCc1cccc(Br)n1 | C#CCOc1ccc(CNc2nc(NCc3cccc(Br)n3)nc(N[C@@H]\(CC... |             BRD4 |         0 | \<rdkit.Chem.rdchem.Mol object at 0x000001C2661... | [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ... |
| **4** |      4 | C#CC[C@@H]\(CC(=O)O)NC(=O)OCC1c2ccccc2-c2ccccc21 |      C#CCOc1ccc(CN)cc1.Cl |         Br.NCc1cccc(Br)n1 | C#CCOc1ccc(CNc2nc(NCc3cccc(Br)n3)nc(N[C@@H]\(CC... |              HSA |         0 | \<rdkit.Chem.rdchem.Mol object at 0x000001C2661... | [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ... |

```
# One hot encoding protein name
one_hot_encoded = pd.get_dummies(train_df['protein_name'], prefix='Protein: ')
one_hot_encoded = one_hot_encoded.astype(int)
ndfn = train_df.drop('protein_name', axis=1)
ndf = pd.concat([ndfn, one_hot_encoded], axis=1)

```

```
print(ndf.columns)

```

```
Index(['id', 'buildingblock1_smiles', 'buildingblock2_smiles',
       'buildingblock3_smiles', 'molecule_smiles', 'binds', 'molecule', 'ecfp',
       'Protein: _BRD4', 'Protein: _HSA', 'Protein: _sEH'],
      dtype='object')

```

```
list_lengths = ndf['ecfp'].apply(len)
print(list_lengths.unique())

```

```
[1024]

```

```
# Convert the list column into separate columns
expanded_df = pd.DataFrame(ndf['ecfp'].to_list(), columns=[f'ecfp_{i+1}' for i in range(1024)])

# Combine the expanded DataFrame with the original DataFrame
result_df = pd.concat([ndf, expanded_df], axis=1)

# Drop the original 'ecfp' column
result_df.drop(columns=['ecfp'], inplace=True)

```

```
# Filter numeric columns
numeric_columns = result_df.select_dtypes(include='number').columns

# Keep only numeric columns
df_numeric = result_df[numeric_columns]

# Display the DataFrame with only numeric columns
print(df_numeric)

```

```
            id  binds  Protein: _BRD4  Protein: _HSA  Protein: _sEH  ecfp_1  \
0            0      0               1              0              0       0   
1            1      0               0              1              0       0   
2            2      0               0              0              1       0   
3            3      0               1              0              0       0   
4            4      0               0              1              0       0   
...        ...    ...             ...            ...            ...     ...   
999995  999995      0               0              0              1       0   
999996  999996      0               1              0              0       0   
999997  999997      0               0              1              0       0   
999998  999998      0               0              0              1       0   
999999  999999      0               1              0              0       0   

        ecfp_2  ecfp_3  ecfp_4  ecfp_5  ...  ecfp_1015  ecfp_1016  ecfp_1017  \
0            1       0       0       1  ...          0          0          0   
1            1       0       0       1  ...          0          0          0   
2            1       0       0       1  ...          0          0          0   
3            1       0       0       0  ...          0          0          0   
4            1       0       0       0  ...          0          0          0   
...        ...     ...     ...     ...  ...        ...        ...        ...   
999995       1       0       0       0  ...          0          0          0   
999996       1       0       0       0  ...          0          0          0   
999997       1       0       0       0  ...          0          0          0   
999998       1       0       0       0  ...          0          0          0   
999999       1       0       0       0  ...          0          0          0   

        ecfp_1018  ecfp_1019  ecfp_1020  ecfp_1021  ecfp_1022  ecfp_1023  \
0               0          0          1          0          0          0   
1               0          0          1          0          0          0   
2               0          0          1          0          0          0   
3               0          0          0          0          0          0   
4               0          0          0          0          0          0   
...           ...        ...        ...        ...        ...        ...   
999995          0          0          0          0          0          0   
999996          0          0          0          0          0          0   
999997          0          0          0          0          0          0   
999998          0          0          0          0          0          0   
999999          0          0          0          0          0          0   

        ecfp_1024  
0               0  
1               0  
2               0  
3               0  
4               0  
...           ...  
999995          0  
999996          1  
999997          1  
999998          1  
999999          0  

[1000000 rows x 1029 columns]

```

```
pc_list = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14']
#Deciding on number of PCA features
for i in range(15, 401):
    pc_list.append(f'PC{i}')

```

```
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

#Prepping df for PCA
dta_for_pca = df_numeric.drop(columns=['id','binds'])

# Perform PCA
pca = PCA(n_components = 400)  # Specify the number of components you want
pca_result = pca.fit_transform(dta_for_pca)

# Create a DataFrame to store the PCA results
pca_df = pd.DataFrame(data=pca_result, columns=pc_list)

# Optionally, you can access the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained variance ratio:", explained_variance_ratio)

# Optionally, you can access the principal components (eigenvectors)
principal_components = pca.components_
print("Principal components (eigenvectors):", principal_components)

```

```
#Reasonable Variance explained (Hopefully :*) )
variance_explained = explained_variance_ratio[:400].sum()
print(f"Variance explained by the components: {variance_explained * 100:.2f}%")

```

```
Variance explained by the components: 73.56%

```

```
#Readding ID and binds target class
post_pca_df = pd.concat((pca_df,df_numeric[['id','binds']]),axis=1)

```

```
#Seperating data
X = post_pca_df.drop(columns=['id','binds'])
y = post_pca_df[['binds']]

```

```
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

```

```
#Sanity checks
print(X_test.columns)
print(y.value_counts())

```

```
Index(['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10',
       ...
       'PC191', 'PC192', 'PC193', 'PC194', 'PC195', 'PC196', 'PC197', 'PC198',
       'PC199', 'PC200'],
      dtype='object', length=200)
binds
0        997424
1          2576
Name: count, dtype: int64

```

```
# 3. Define the Model
model = Sequential()

# Input layer
model.add(Dense(128, input_dim=400, activation='relu'))

# Hidden layers
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))

# Output layer
model.add(Dense(1, activation='sigmoid'))

# 4. Compile the Model
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='binary_crossentropy', 
              metrics=['accuracy', Precision()])

# 5. Train the Model
history = model.fit(X_train, y_train, 
                    validation_split=0.2, 
                    epochs=30, 
                    batch_size=32)

# 6. Evaluate the Model
loss, accuracy, precision = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')
print(f'Test Precision: {precision}')
```