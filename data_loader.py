import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv("heart.csv")

feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
target_col = "target"

X = data[feature_cols]
y = data[target_col]

ohe = OneHotEncoder(sparse_output=False, drop='first')
cp_encoded = ohe.fit_transform(X[['cp']])
cp_categories = ohe.categories_[0][1:]
cp_columns = [f'cp_{int(cat)}' for cat in cp_categories]
cp_encoded_df = pd.DataFrame(cp_encoded, columns=cp_columns)
X = pd.concat([X.drop('cp', axis=1), cp_encoded_df], axis=1)

feature_cols.remove("cp")
feature_cols += cp_columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

pd.concat([X_train, y_train], axis=1).to_csv('train.csv', index=False)
pd.concat([X_test, y_test], axis=1).to_csv('test.csv', index=False)
