# -*- coding: utf-8 -*-
import pandas as pd
df = pd.read_csv(r'C:\Users\jd\Desktop\titanic_cleaned.csv')

deceased = df[df['Survived'] == 0]
avg_age = deceased['Age'].mean()

survived = df[df['Survived'] == 1]
avg_age_s = survived['Age'].mean()

print("=" * 40)
print("Total:", len(df))
print("Deceased:", len(deceased))
print("Avg age(deceased): %.2f" % avg_age)
print("-" * 40)
print("Survived:", len(survived))
print("Avg age(survived): %.2f" % avg_age_s)
print("=" * 40)
