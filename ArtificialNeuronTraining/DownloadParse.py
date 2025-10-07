from ucimlrepo import fetch_ucirepo 
import pandas as pd
import json

 
# fetch dataset 
breast_cancer_wisconsin_original = fetch_ucirepo(id=15) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_original.data.features 
y = breast_cancer_wisconsin_original.data.targets 

# retrieve only 2 and 4 classes
df = pd.concat([X, y], axis=1)
df_filtered = df[df['Class'].isin([2, 4])]

# drop any rows with NaN values
df_filtered = df_filtered.dropna()

# map class 2 -> 0, 4 -> 1
df_filtered['Class'] = df_filtered['Class'].map({2: 0, 4: 1})

# build JSON structure
output = {
        "records": df_filtered.to_dict(orient="records")
    }

# save to file
with open("breast_cancer_groups.json", "w") as f:
    json.dump(output, f, indent=2)
