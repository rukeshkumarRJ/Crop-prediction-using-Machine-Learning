import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier


df = pd.read_csv('Crop_recommendation_ds.csv')

X = df.drop('label', axis=1)
y = df['label']

from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(X, y,random_state=1)

# Define base models
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('lgb', lgb.LGBMClassifier(n_estimators=100, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=100, random_state=42))
]

# Initialize and train the Stacking Classifier model
stack_model = StackingClassifier(estimators=base_models, final_estimator=RandomForestClassifier())
stack_model.fit(x_train, y_train)

#custom_input = np.array([[89,40,40,20.87974371,80.00274423,6.302985292,202.9355362]])
#custom_input2 = np.array([[32,	57,	17,	28.17876451,	64.51663927,	8.202706015,	34.96933295]])

pickle.dump(stack_model,open('proj.pkl','wb'))