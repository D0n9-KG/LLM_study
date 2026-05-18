# -*- coding: utf-8 -*-
"""
使用训练好的模型进行预测的演示脚本
"""
import pandas as pd
import numpy as np
import joblib

# ==================== 加载模型和标准化器 ====================
print("=" * 60)
print("加载已训练的模型")
print("=" * 60)

model = joblib.load('titanic_survival_model.pkl')
scaler = joblib.load('titanic_scaler.pkl')
print("模型加载成功！")

# ==================== 加载原始数据用于演示 ====================
df = pd.read_csv('titanic_cleaned.csv')

# ==================== 示例1：预测测试集 ====================
print("\n" + "=" * 60)
print("示例1：预测测试集（后20条数据）")
print("=" * 60)

def preprocess_data(data_df, is_training=False):
    """对数据进行与训练时相同的特征工程处理"""
    processed = data_df.copy()
    
    # 删除无关列
    drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived']
    cols_to_drop = [c for c in drop_cols if c in processed.columns]
    processed.drop(columns=cols_to_drop, inplace=True)
    
    # 编码Sex
    processed['Sex'] = processed['Sex'].map({'male': 1, 'female': 0})
    
    # Embarked独热编码
    embarked_dummies = pd.get_dummies(processed['Embarked'], prefix='Embarked')
    # 确保所有Embarked列都存在
    for col in ['Embarked_C', 'Embarked_Q', 'Embarked_S']:
        if col not in embarked_dummies.columns:
            embarked_dummies[col] = 0
    processed = pd.concat([processed, embarked_dummies[['Embarked_C', 'Embarked_Q', 'Embarked_S']]], axis=1)
    processed.drop(columns=['Embarked'], inplace=True)
    
    # 创建特征
    processed['FamilySize'] = processed['SibSp'] + processed['Parch'] + 1
    processed['IsAlone'] = (processed['FamilySize'] == 1).astype(int)
    processed['AgeGroup'] = pd.cut(processed['Age'], bins=[0, 12, 18, 35, 50, 100], labels=[0, 1, 2, 3, 4]).astype(int)
    processed['FareGroup'] = pd.qcut(processed['Fare'].rank(method='first'), q=4, labels=[0, 1, 2, 3]).astype(int)
    
    return processed

# 取测试集后20条
test_samples = df.tail(20).copy()
X_test_samples = preprocess_data(test_samples)

# 标准化
X_test_scaled = scaler.transform(X_test_samples)

# 预测
predictions = model.predict(X_test_scaled)
probabilities = model.decision_function(X_test_scaled) if hasattr(model, 'decision_function') else model.predict_proba(X_test_scaled)[:, 1]

print(f"{'乘客姓名':<35} {'实际':>6} {'预测':>6} {'置信度':>8}")
print("-" * 60)
for i, (_, row) in enumerate(test_samples.iterrows()):
    actual = row['Survived']
    pred = predictions[i]
    prob = probabilities[i] if isinstance(probabilities, np.ndarray) else probabilities
    # 将decision_function输出转为概率 [0,1]
    if hasattr(model, 'decision_function'):
        prob = 1 / (1 + np.exp(-prob))
    status = '幸存' if pred == 1 else '未幸存'
    actual_status = '幸存' if actual == 1 else '未幸存'
    print(f"{row['Name'][:34]:<35} {actual_status:>6} {status:>6} {prob:>7.1%}")

# ==================== 示例2：预测自定义乘客数据 ====================
print("\n" + "=" * 60)
print("示例2：预测自定义乘客数据")
print("=" * 60)

custom_passengers = pd.DataFrame([
    {
        'Pclass': 1,
        'Sex': 'female',
        'Age': 25.0,
        'SibSp': 0,
        'Parch': 0,
        'Fare': 100.0,
        'Embarked': 'C'
    },
    {
        'Pclass': 3,
        'Sex': 'male',
        'Age': 30.0,
        'SibSp': 1,
        'Parch': 0,
        'Fare': 7.25,
        'Embarked': 'S'
    },
    {
        'Pclass': 2,
        'Sex': 'male',
        'Age': 60.0,
        'SibSp': 0,
        'Parch': 2,
        'Fare': 30.0,
        'Embarked': 'S'
    },
    {
        'Pclass': 1,
        'Sex': 'female',
        'Age': 5.0,
        'SibSp': 1,
        'Parch': 2,
        'Fare': 150.0,
        'Embarked': 'C'
    }
])

X_custom = preprocess_data(custom_passengers)
X_custom_scaled = scaler.transform(X_custom)
custom_preds = model.predict(X_custom_scaled)

if hasattr(model, 'decision_function'):
    custom_probs = model.decision_function(X_custom_scaled)
    custom_probs = 1 / (1 + np.exp(-custom_probs))
else:
    custom_probs = model.predict_proba(X_custom_scaled)[:, 1]

print(f"\n{'序号':<4} {'船舱等级':<8} {'性别':<6} {'年龄':<6} {'票价':<8} {'登船口':<6} {'预测':<8} {'幸存概率':<10}")
print("-" * 70)
for i, (_, row) in enumerate(custom_passengers.iterrows()):
    pred = custom_preds[i]
    prob = custom_probs[i]
    status = '幸存' if pred == 1 else '未幸存'
    print(f"{i+1:<4} {row['Pclass']:<8} {row['Sex']:<6} {row['Age']:<6.0f} {row['Fare']:<8.1f} {row['Embarked']:<6} {status:<8} {prob:<10.1%}")

print("\n" + "=" * 60)
print("预测完成！")
print("=" * 60)
