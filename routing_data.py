import pandas as pd
import os

# Load model result files
df1 = pd.read_csv('/home/ubuntu/STEAD/outputs/model1_results.csv')
df2 = pd.read_csv('/home/ubuntu/STEAD/outputs/model2_results.csv')
df3 = pd.read_csv('/home/ubuntu/STEAD/outputs/model3_results.csv')
df4 = pd.read_csv('/home/ubuntu/STEAD/outputs/model4_results.csv')

df1['name'] = df1['name'].apply(os.path.basename)

# Rename correctness columns for clarity
df1 = df1.rename(columns={'correct': 'model1_correct'})
df2 = df2.rename(columns={'correct': 'model2_correct'})
df3 = df3.rename(columns={'correct': 'model3_correct'})
df4 = df4.rename(columns={'correct': 'model4_correct'})

# Merge on common identifier column
combined = (
    df1[['name', 'model1_correct']]
    .merge(df2[['name', 'model2_correct']], on='name')
    .merge(df3[['name', 'model3_correct']], on='name')
    .merge(df4[['name', 'model4_correct']], on='name')
)

# Define the routing column based on priority
def first_correct(row):
    if row['model1_correct']:
        return 1
    elif row['model2_correct']:
        return 2
    elif row['model3_correct']:
        return 3
    elif row['model3_correct']:
        return 4
    else:
        return 3  # No model got it right

combined['routing'] = combined.apply(first_correct, axis=1)

# Add all_correct: 1 if any model got it right, else 0
combined['all_correct'] = combined[
    ['model1_correct', 'model2_correct', 'model3_correct']
].max(axis=1)

# Save combined results
combined.to_csv('/home/ubuntu/STEAD/outputs/routing_data.csv', index=False)

print(combined.head())
