# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score

# Load model
model = joblib.load('C:/Users/jd/Desktop/titanic_survival_model.pkl')
scaler = joblib.load('C:/Users/jd/Desktop/titanic_scaler.pkl')
print('Model loaded:', type(model).__name__)

df = pd.read_csv('C:/Users/jd/Desktop/titanic_cleaned.csv')
print('Data loaded:', df.shape)

def preproc(d):
    p = d.copy()
    dc = [c for c in ['PassengerId','Name','Ticket','Cabin','Survived'] if c in p.columns]
    p.drop(columns=dc, inplace=True)
    p['Sex'] = p['Sex'].map({'male':1,'female':0})
    ed = pd.get_dummies(p['Embarked'], prefix='Embarked')
    for c in ['Embarked_C','Embarked_Q','Embarked_S']:
        if c not in ed.columns:
            ed[c] = 0
    p = pd.concat([p, ed[['Embarked_C','Embarked_Q','Embarked_S']]], axis=1)
    p.drop(columns=['Embarked'], inplace=True)
    p['FamilySize'] = p['SibSp'] + p['Parch'] + 1
    p['IsAlone'] = (p['FamilySize'] == 1).astype(int)
    p['AgeGroup'] = pd.cut(p['Age'], bins=[0,12,18,35,50,100], labels=[0,1,2,3,4]).astype(int)
    p['FareGroup'] = pd.qcut(p['Fare'].rank(method='first'), q=4, labels=[0,1,2,3]).astype(int)
    return p

# Predict full dataset
X_all = preproc(df)
X_all_scaled = scaler.transform(X_all)
y_pred = model.predict(X_all_scaled)
acc = accuracy_score(df['Survived'], y_pred)
print(f'Full dataset accuracy: {acc:.4f}')
print()

# Sample predictions
print('Sample predictions (first 10):')
for i in range(10):
    r = df.iloc[i]
    act = 'Survived' if r['Survived']==1 else 'Died'
    pred = 'Survived' if y_pred[i]==1 else 'Died'
    ok = 'CORRECT' if r['Survived']==y_pred[i] else 'WRONG'
    print(f'  {r["Name"][:22]:<22} Sex={r["Sex"]:<5} Age={r["Age"]:<5.0f} Class={r["Pclass"]} -> Actual={act:<8} Pred={pred:<8} [{ok}]')

print()

# Custom predictions
custom = pd.DataFrame([
    {'Pclass':1, 'Sex':'female', 'Age':25, 'SibSp':0, 'Parch':0, 'Fare':100, 'Embarked':'C'},
    {'Pclass':3, 'Sex':'male', 'Age':30, 'SibSp':1, 'Parch':0, 'Fare':7.25, 'Embarked':'S'},
    {'Pclass':1, 'Sex':'female', 'Age':5, 'SibSp':1, 'Parch':2, 'Fare':150, 'Embarked':'C'},
    {'Pclass':3, 'Sex':'male', 'Age':65, 'SibSp':0, 'Parch':0, 'Fare':8.05, 'Embarked':'S'},
])

Xc = preproc(custom)
Xcs = scaler.transform(Xc)
cp = model.predict(Xcs)

if hasattr(model, 'predict_proba'):
    cprob = model.predict_proba(Xcs)[:, 1]
else:
    cprob = model.decision_function(Xcs)
    cprob = 1/(1+np.exp(-cprob))

desc = ['Rich young woman', 'Poor young man', 'Rich girl with family', 'Poor old man']
print('Custom predictions:')
for i in range(4):
    status = 'Survived' if cp[i]==1 else 'Died'
    print(f'  {desc[i]:<25}: {status:<10} (probability: {cprob[i]:.1%})')

print()
print('Done!')
