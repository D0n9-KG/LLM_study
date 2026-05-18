# -*- coding: utf-8 -*-
import os
os.environ['MPLBACKEND'] = 'agg'  # 覆盖环境变量

import matplotlib
matplotlib.use('agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv('titanic_cleaned.csv')

print("=" * 60)
print("Titanic Dataset Basic Info")
print("=" * 60)
print(f"Total samples: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print()

# 1. 计算遇难者的平均年龄
died_df = df[df['Survived'] == 0]
survived_df = df[df['Survived'] == 1]

died_mean_age = died_df['Age'].mean()
survived_mean_age = survived_df['Age'].mean()
overall_mean_age = df['Age'].mean()

print("=" * 60)
print("Age Statistics")
print("=" * 60)
print(f"Died passengers count: {len(died_df)}")
print(f"Died average age: {died_mean_age:.2f} years")
print(f"Survived passengers count: {len(survived_df)}")
print(f"Survived average age: {survived_mean_age:.2f} years")
print(f"Overall average age: {overall_mean_age:.2f} years")
print()

# 检查年龄缺失值
missing_age = df['Age'].isna().sum()
print(f"Missing age values: {missing_age}")
print()

# 2. 统计 Survived 列的分布
survived_counts = df['Survived'].value_counts()
survived_percentages = df['Survived'].value_counts(normalize=True) * 100

print("=" * 60)
print("Survived Column Distribution")
print("=" * 60)
print(f"Survived (1): {survived_counts.get(1, 0)} people ({survived_percentages.get(1, 0):.2f}%)")
print(f"Died (0): {survived_counts.get(0, 0)} people ({survived_percentages.get(0, 0):.2f}%)")
print()

# ========== Plotting ==========
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Titanic Data Analysis', fontsize=16, fontweight='bold')

# Plot 1: Survived pie chart
ax1 = axes[0, 0]
labels = ['Died (0)', 'Survived (1)']
colors = ['#FF6B6B', '#4ECDC4']
explode = (0.05, 0.05)
wedges, texts, autotexts = ax1.pie(
    survived_counts, 
    labels=labels, 
    autopct='%1.1f%%',
    colors=colors,
    explode=explode,
    startangle=90,
    textprops={'fontsize': 12}
)
ax1.set_title('Survived Distribution (Pie)', fontsize=13, fontweight='bold')

# Plot 2: Survived bar chart
ax2 = axes[0, 1]
bar_colors = ['#FF6B6B', '#4ECDC4']
bars = ax2.bar(['Died (0)', 'Survived (1)'], survived_counts, color=bar_colors, edgecolor='black', linewidth=1.2)
for bar, count in zip(bars, survived_counts):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
             str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('Survived Distribution (Bar)', fontsize=13, fontweight='bold')
ax2.set_ylim(0, max(survived_counts) * 1.15)

# Plot 3: Age distribution histogram
ax3 = axes[1, 0]
ax3.hist([died_df['Age'].dropna(), survived_df['Age'].dropna()], 
         bins=20, alpha=0.7, label=['Died', 'Survived'],
         color=['#FF6B6B', '#4ECDC4'], edgecolor='black', linewidth=0.8)
ax3.axvline(died_mean_age, color='#FF6B6B', linestyle='--', linewidth=2, 
            label=f'Died avg {died_mean_age:.1f}y')
ax3.axvline(survived_mean_age, color='#4ECDC4', linestyle='--', linewidth=2, 
            label=f'Survived avg {survived_mean_age:.1f}y')
ax3.set_xlabel('Age', fontsize=12)
ax3.set_ylabel('Count', fontsize=12)
ax3.set_title('Age Distribution (Died vs Survived)', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)

# Plot 4: Box plot
ax4 = axes[1, 1]
bp_data = [died_df['Age'].dropna(), survived_df['Age'].dropna()]
bp = ax4.boxplot(bp_data, labels=['Died (0)', 'Survived (1)'], 
                 patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor('#FF6B6B')
bp['boxes'][1].set_facecolor('#4ECDC4')
bp['medians'][0].set_color('black')
bp['medians'][1].set_color('black')
ax4.scatter(1, died_mean_age, color='darkred', s=100, marker='D', zorder=5, label=f'Mean {died_mean_age:.1f}')
ax4.scatter(2, survived_mean_age, color='darkgreen', s=100, marker='D', zorder=5, label=f'Mean {survived_mean_age:.1f}')
ax4.set_ylabel('Age', fontsize=12)
ax4.set_title('Age Box Plot (by Survival)', fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('titanic_analysis.png', dpi=150, bbox_inches='tight')
print("Image saved as: titanic_analysis.png")

# Extra: survival rate by Sex and Pclass
print("\n" + "=" * 60)
print("Survival Rate by Sex and Pclass")
print("=" * 60)
pivot = df.pivot_table(values='Survived', index='Sex', columns='Pclass', aggfunc='mean')
print(pivot.round(3))
