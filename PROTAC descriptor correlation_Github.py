# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 15:02:34 2024

@author: PanagioM
"""

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, tstd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Define the path for the descriptor file
descriptor_file = 'descriptors_all_Kristen Drugs_with_NHISS.csv'

# Load and prepare descriptor data
df = pd.read_csv(descriptor_file).sort_values('Name').reset_index(drop=True)

# Load and prepare labeled drug data
label_file = "library_2.csv"
df2 = pd.read_csv(label_file).sort_values('Drug').reset_index(drop=True)

# Merge descriptor and label data
df = pd.merge(df, df2[['Drug', 'labels']], left_on='Name', right_on='Drug', how='inner')

# Drop unnecessary columns
df.drop(columns=['Drug'], inplace=True)

# Split the data into features (x_df) and labels (y_df)
x_df = df.drop(['Name', 'Smiles', 'labels'], axis=1)
y_df = df['labels']
drug_names = df['Name']

# Convert all data to numeric, coercing errors to NaN
x_df = x_df.apply(pd.to_numeric, errors='coerce')

x_train = x_df
y_train = y_df

# Separate data into 'yes' and 'no' based on the label
yes = x_train[y_train == 1]
no = x_train[y_train == 0]
yes_names = drug_names[y_train == 1]
no_names = drug_names[y_train == 0]

# Initialize a DataFrame to store statistical results
p = pd.DataFrame(columns=['Descriptors', 'p-value', 'Formers mean', 'Formers stdev', 'Non-Formers mean', 'Non-Formers stdev', 'Formers count', 'Non-Formers count'])

# Perform statistical analysis for each descriptor
results = []
for descriptor in x_train.columns:
    yes_col = pd.to_numeric(yes[descriptor], errors='coerce').dropna()
    no_col = pd.to_numeric(no[descriptor], errors='coerce').dropna()

    if len(yes_col) > 0 and len(no_col) > 0:
        p_value = mannwhitneyu(yes_col, no_col)[1]
        results.append({
            'Descriptors': descriptor,
            'p-value': p_value,
            'Formers mean': np.mean(yes_col),
            'Formers stdev': tstd(yes_col),
            'Non-Formers mean': np.mean(no_col),
            'Non-Formers stdev': tstd(no_col),
            'Formers count': len(yes_col),
            'Non-Formers count': len(no_col)
        })
    else:
        print(f"Skipping descriptor '{descriptor}' due to insufficient data.")
        results.append({'Descriptors': descriptor, 'p-value': np.nan})

p = pd.DataFrame(results)

# Remove descriptors with NaN p-values due to insufficient data
p = p.dropna(subset=['p-value'])

# Sort and filter features based on p-value
p.sort_values('p-value', ascending=True, inplace=True)
p.reset_index(drop=True, inplace=True)

# Calculate the correlation matrix for the top 20 descriptors
corr = x_train[p['Descriptors'].tolist()[:20]].corr().abs()

# Filter highly correlated features
droplist = []
correlated_pairs = []
for w in corr.columns:
    for s in corr.index[corr.index > w]:
        if corr[w][s] > 0.9 and s not in droplist:
            droplist.append(s)
            correlated_pairs.append((w, s, corr[w][s]))

print("len of droplist: ", len(droplist))
print("droplist: ", droplist)

# Print correlated pairs
print("Correlated pairs with correlation > 0.9:")
for pair in correlated_pairs:
    print(pair)

correlated_df = pd.DataFrame(correlated_pairs, columns=['Descriptor 1', 'Descriptor 2', 'Correlation'])
correlated_df.to_excel('highly_autocorrelated.xlsx')

# Exclude highly correlated descriptors from the number of tests
num_tests = len(p) - len(droplist)

# Calculate BH and BKY critical values for 10% and 20% FDR
fdr_values = [0.1, 0.2]
bh_results = {}
bky_results = {}

for fdr in fdr_values:
    c_m = np.sum(1.0 / np.arange(1, num_tests + 1))
    
    bh_critical_values = [(i + 1) / num_tests * fdr for i in range(num_tests)]
    extended_bh_critical_values = bh_critical_values + [None] * (len(p) - len(bh_critical_values))
    bh_results[fdr] = extended_bh_critical_values
    
    bky_critical_values = [(i + 1) / num_tests * fdr / c_m for i in range(num_tests)]
    extended_bky_critical_values = bky_critical_values + [None] * (len(p) - len(bky_critical_values))
    bky_results[fdr] = extended_bky_critical_values

# Add BH and BKY critical values to the DataFrame
p['BH critical value (10%)'] = bh_results[0.1]
p['BH critical value (20%)'] = bh_results[0.2]
p['BKY critical value (10%)'] = bky_results[0.1]
p['BKY critical value (20%)'] = bky_results[0.2]

# Determine significant features based on BH and BKY critical values
p['Significant BH (10%)'] = np.where(p['p-value'] <= p['BH critical value (10%)'], 'yes', 'no')
p['Significant BH (20%)'] = np.where(p['p-value'] <= p['BH critical value (20%)'], 'yes', 'no')
p['Significant BKY (10%)'] = np.where(p['p-value'] <= p['BKY critical value (10%)'], 'yes', 'no')
p['Significant BKY (20%)'] = np.where(p['p-value'] <= p['BKY critical value (20%)'], 'yes', 'no')

# Select the top 20 descriptors
top_20_descriptors = p.head(20)

# Filter features based on p-value threshold
feats = top_20_descriptors['Descriptors'].tolist()

print("Length of feats: ", len(feats))
print("Feats: ", feats)

# Determine the suffix based on the label file
suffix = "_1" if "_1" in label_file else "_2" if "_2" in label_file else "_3"

# Create Excel writer object and save the top 20 descriptors
with pd.ExcelWriter(f"top_20_descriptors_NHISS{suffix}.xlsx", engine='xlsxwriter') as writer:
    # Save top 20 descriptors
    top_20_descriptors.to_excel(writer, sheet_name='Top_20_Descriptors', index=False)
    
    # Save all descriptors with significant markings
    p.to_excel(writer, sheet_name='All_Descriptors', index=False)
    
    # Create DataFrames for formers and non-formers
    formers_df = yes[feats].copy()
    formers_df['Drug Name'] = yes_names.values
    non_formers_df = no[feats].copy()
    non_formers_df['Drug Name'] = no_names.values

    # Save formers and non-formers data to separate sheets
    formers_df.to_excel(writer, sheet_name='Formers', index=False)
    non_formers_df.to_excel(writer, sheet_name='Non_Formers', index=False)

    # Save the number of tests
    num_tests_df = pd.DataFrame({'Number of Tests': [num_tests]})
    num_tests_df.to_excel(writer, sheet_name='Number_of_Tests', index=False)

# Visualize and save the correlation matrix
if not corr.empty:
    plt.figure(dpi=600)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='RdYlBu_r')
    plt.savefig(f"correlation_matrix{suffix}.png")
    plt.show()

# Number of features to plot
if len(feats) < 9:
    num_feats = len(feats)
else:
    num_feats = 9

# Calculate the number of rows and columns
ncols = 3
nrows = math.ceil(num_feats / ncols)

# Plotting the top descriptors
if num_feats > 0:
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, dpi=600, figsize=(15, 5*nrows))
    fig.tight_layout(pad=4.0)

    var_count = 0
    for j in range(nrows):
        for k in range(ncols):
            if var_count < num_feats:
                desc = feats[var_count]
                ax = axes[j, k] if nrows > 1 else axes[k]
                sns.stripplot(
                    x=y_train, y=x_train[desc], ax=ax, palette='tab10',
                    size=5, marker='o', edgecolor='black', linewidth=0.5
                )

                # Add drug names to the scatter plot
                for line in range(0, x_train.shape[0], 10):  # Adding text to a subset of points to avoid clutter
                    ax.text(
                        y_train.iloc[line],
                        x_train[desc].iloc[line],
                        drug_names.iloc[line],
                        horizontalalignment='left',
                        size='x-small',  # Smaller font size
                        color='black'
                    )

                ax.spines[['right', 'top']].set_visible(False)
                ax.set_xlabel('Label')
                ax.set_ylabel(desc)
                ax.set_xticks([1, 0])  # Swap the order to put formers (1) on the left and non-formers (0) on the right
                ax.set_xticklabels(['Formers', 'Non-Formers'])
                ax.set_title(f'{desc} (p={p["p-value"].iloc[var_count]:.3e})')
                var_count += 1
            else:
                if nrows > 1:
                    axes[j, k].axis('off')
                else:
                    axes[k].axis('off')

    plt.savefig(f"top_descriptors{suffix}.png")
    plt.show()
else:
    print("No features available for plotting.")
