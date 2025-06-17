import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Data Import
print("Step 1/6: Importing dataset...")
df = pd.read_csv(r"C:\Users\alish\Desktop\VS Workspace\deepfake Scams.csv")
print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")

# 2. Data Preprocessing & Cleaning
print("\nStep 2/6: Cleaning data...")
# Handle missing values
initial_rows = len(df)
df = df.dropna()
print(f"- Removed {initial_rows - len(df)} rows with missing values")

# Convert data types
df['Detection_Time_Hours'] = pd.to_numeric(df['Detection_Time_Hours'], errors='coerce')
df['Loss_Amount_USD'] = pd.to_numeric(df['Loss_Amount_USD'], errors='coerce')

# Remove duplicates
df = df.drop_duplicates()
print(f"- Removed duplicates: {initial_rows - len(df)} rows total removed")

# Handle outliers
Q1 = df['Loss_Amount_USD'].quantile(0.25)
Q3 = df['Loss_Amount_USD'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df['Loss_Amount_USD'] >= lower_bound) & 
        (df['Loss_Amount_USD'] <= upper_bound)]

print(f"✓ Cleaned data: {len(df)} rows ({initial_rows - len(df)} outliers removed)")

# 3. Exploratory Data Analysis (EDA)
print("\nStep 3/6: Performing EDA...")
print("\nDescriptive Statistics:")
print(df.describe())

# 4. Insight Generation
print("\nStep 4/6: Generating insights...")
# Insight 1: Top 3 attack methods by loss
top_attacks = df.groupby('Attack_Method')['Loss_Amount_USD'].mean().nlargest(3)
print(f"\n[Insight 1] Highest Financial Impact Attacks:")
print(top_attacks)

# Insight 2: Prevention effectiveness
prevention_effect = df.groupby('Prevention_Measures')['Loss_Amount_USD'].mean().sort_values()
print(f"\n[Insight 2] Prevention Effectiveness:")
print(prevention_effect)

# 5. Data Visualization
print("\nStep 5/6: Creating visualizations...")
plt.figure(figsize=(12, 8))

# Visualization 1: Loss distribution by industry
plt.subplot(2, 2, 1)
sns.boxplot(x='Target_Industry', y='Loss_Amount_USD', data=df)
plt.title('Financial Loss by Industry')
plt.xticks(rotation=45)

# Visualization 2: Attack method frequency
plt.subplot(2, 2, 2)
df['Attack_Method'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title('Attack Method Distribution')
plt.ylabel('')

# Visualization 3: Detection time vs prevention
plt.subplot(2, 2, 3)
sns.barplot(x='Prevention_Measures', y='Detection_Time_Hours', data=df, ci=None)
plt.title('Detection Time by Prevention Measure')
plt.xticks(rotation=45)

# Visualization 4: Loss correlation
plt.subplot(2, 2, 4)
sns.scatterplot(x='Detection_Time_Hours', y='Loss_Amount_USD', hue='Target_Industry', data=df)
plt.title('Detection Time vs Financial Loss')

plt.tight_layout()
plt.savefig('ccp_visualizations.png')
print("✓ Saved visualizations to ccp_visualizations.png")

# 6. Documentation Prep (FIXED SECTION)
print("\nStep 6/6: Preparing documentation...")
report = f"""
# CCP: Deepfake Scam Analysis Report
## Key Findings
1. **Top 3 High-Impact Attacks**: 
   - {top_attacks.index[0]} (${top_attacks.iloc[0]:,.0f})
   - {top_attacks.index[1]} (${top_attacks.iloc[1]:,.0f})
   - {top_attacks.index[2]} (${top_attacks.iloc[2]:,.0f})

2. **Most Effective Prevention**: 
   - {prevention_effect.index[0]} (${prevention_effect.iloc[0]:,.0f} avg loss)
   - {prevention_effect.index[1]} (${prevention_effect.iloc[1]:,.0f} avg loss)

3. **Critical Correlation**: 
   - Longer detection time → Higher financial loss (r = {df['Detection_Time_Hours'].corr(df['Loss_Amount_USD']):.2f})
"""

# FIX: Add encoding='utf-8' here
with open('ccp_report.md', 'w', encoding='utf-8') as f:
    f.write(report)

print("\nCCP PROCESS COMPLETED SUCCESSFULLY!")