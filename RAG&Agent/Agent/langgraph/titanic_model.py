# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. Load Data ====================
df = pd.read_csv('titanic_cleaned.csv')
print("=" * 60)
print("Original Data Info")
print("=" * 60)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nSurvived distribution:\n{df['Survived'].value_counts()}")
print(f"\nData types:\n{df.dtypes}")

# ==================== 2. Feature Engineering ====================
print("\n" + "=" * 60)
print("Feature Engineering")
print("=" * 60)

data = df.copy()

# 2.1 Drop irrelevant columns
drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
data.drop(columns=drop_cols, inplace=True)
print(f"Shape after dropping {drop_cols}: {data.shape}")

# 2.2 Encode categorical variables
le_sex = LabelEncoder()
data['Sex'] = le_sex.fit_transform(data['Sex'])
print(f"Sex encoding: {dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_)))}")

# Embarked: one-hot encoding
embarked_dummies = pd.get_dummies(data['Embarked'], prefix='Embarked')
data = pd.concat([data, embarked_dummies], axis=1)
data.drop(columns=['Embarked'], inplace=True)
print(f"Shape after Embarked one-hot: {data.shape}")

# 2.3 Create family size feature
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

# 2.4 Create IsAlone feature
data['IsAlone'] = (data['FamilySize'] == 1).astype(int)

# 2.5 Age grouping
data['AgeGroup'] = pd.cut(data['Age'], bins=[0, 12, 18, 35, 50, 100], labels=[0, 1, 2, 3, 4])
data['AgeGroup'] = data['AgeGroup'].astype(int)

# 2.6 Fare grouping
data['FareGroup'] = pd.qcut(data['Fare'].rank(method='first'), q=4, labels=[0, 1, 2, 3])
data['FareGroup'] = data['FareGroup'].astype(int)

print(f"\nShape after feature engineering: {data.shape}")
print(f"All features: {data.columns.tolist()}")

# ==================== 3. Split Features and Target ====================
X = data.drop(columns=['Survived'])
y = data['Survived']

print("\n" + "=" * 60)
print("Feature List")
print("=" * 60)
print(X.columns.tolist())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
print(f"Training set Survived distribution:\n{y_train.value_counts()}")
print(f"Test set Survived distribution:\n{y_test.value_counts()}")

# ==================== 4. Feature Scaling ====================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================== 5. Model Training and Evaluation ====================
print("\n" + "=" * 60)
print("Model Training and Evaluation")
print("=" * 60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42)
}

results = []
best_model = None
best_score = 0

for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'CV Mean Acc': f"{cv_scores.mean():.4f} (+-{cv_scores.std():.4f})",
        'Test Acc': f"{accuracy:.4f}",
        'Precision': f"{precision:.4f}",
        'Recall': f"{recall:.4f}",
        'F1 Score': f"{f1:.4f}"
    })
    
    print(f"\n--- {name} ---")
    print(f"Cross-val accuracy: {cv_scores.mean():.4f} (+-{cv_scores.std():.4f})")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    
    if accuracy > best_score:
        best_score = accuracy
        best_model = (name, model)

results_df = pd.DataFrame(results)
print("\n" + "=" * 60)
print("Model Performance Summary")
print("=" * 60)
print(results_df.to_string(index=False))

# ==================== 6. Hyperparameter Tuning for Best Model ====================
print("\n" + "=" * 60)
print(f"Best Model [{best_model[0]}] Hyperparameter Tuning")
print("=" * 60)

if best_model[0] == 'Random Forest':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
    )
elif best_model[0] == 'Gradient Boosting':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    grid_search = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
    )
elif best_model[0] == 'SVM':
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'kernel': ['rbf']
    }
    grid_search = GridSearchCV(
        SVC(random_state=42),
        param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
    )
else:
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs']
    }
    grid_search = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
    )

grid_search.fit(X_train_scaled, y_train)
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

final_model = grid_search.best_estimator_
y_pred_final = final_model.predict(X_test_scaled)

print(f"\nTuned test accuracy: {accuracy_score(y_test, y_pred_final):.4f}")
print(f"Tuned F1 score: {f1_score(y_test, y_pred_final):.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_final, target_names=['Not Survived', 'Survived']))

# ==================== 7. Feature Importance Analysis ====================
print("\n" + "=" * 60)
print("Feature Importance Analysis")
print("=" * 60)

if hasattr(final_model, 'feature_importances_'):
    importances = final_model.feature_importances_
    feature_names = X.columns
    indices = np.argsort(importances)[::-1]
    
    print("Feature importance ranking:")
    for i in range(len(feature_names)):
        print(f"  {i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
elif hasattr(final_model, 'coef_'):
    coefs = final_model.coef_[0]
    feature_names = X.columns
    indices = np.argsort(np.abs(coefs))[::-1]
    
    print("Feature coefficient ranking:")
    for i in range(len(feature_names)):
        print(f"  {i+1}. {feature_names[indices[i]]}: {coefs[indices[i]]:.4f}")

# ==================== 8. Save Model ====================
import joblib

joblib.dump(final_model, 'titanic_survival_model.pkl')
joblib.dump(scaler, 'titanic_scaler.pkl')
print("\n" + "=" * 60)
print("Model saved: titanic_survival_model.pkl")
print("Scaler saved: titanic_scaler.pkl")
print("=" * 60)
