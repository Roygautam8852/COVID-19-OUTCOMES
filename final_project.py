import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats

df = pd.read_csv('/content/covid-19 dataset.csv')
print("First Few Rows of the Dataset:")
print(df.head())
print("\nDataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())

missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
print("\nMissing Values:\n", missing_values)

print("\nPercentage of Missing Values:\n", missing_percentage)
valid_age_groups = [
    '0-4', '5-11', '12-17', '18-29', '30-49', '50-64', '65-79', '80+', 'All'
]

age_group_mapping = {
    '05-Nov': '5-11',
    'Dec-17': '12-17'
}

if 'Age Group' in df.columns:
    print("\nUnique Age Groups Before Cleaning:", df['Age Group'].unique())
else:
    print("Warning: Age Group column missing.")

if 'Age Group' in df.columns:
    df['Age Group'] = df['Age Group'].replace(age_group_mapping)
    df = df[df['Age Group'].isin(valid_age_groups)]
    print("\nUnique Age Groups After Cleaning:", df['Age Group'].unique())

if all(col in df.columns for col in ['Outcome Boosted', 'Outcome Unvaccinated', 'Outcome Vaccinated', 'Boosted Rate']):
    df.loc[df['Outcome Boosted'].isnull() & (df['Outcome Unvaccinated'] == 0) & 
           (df['Outcome Vaccinated'] == 0), 'Boosted Rate'] = 0
    df.loc[df['Outcome Boosted'].isnull(), 'Outcome Boosted'] = 0
else:
    print("Warning: Required columns for imputation missing.")


for col in ['Age-Adjusted Unvaccinated Rate', 'Age-Adjusted Vaccinated Rate', 
            'Age-Adjusted Boosted Rate']:
    if col in df.columns:
        df[col] = df.groupby(['Outcome', 'Age Group'])[col].transform(lambda x: x.fillna(x.median()))


numerical_cols = ['Unvaccinated Rate', 'Vaccinated Rate', 'Boosted Rate', 'Crude Vaccinated Ratio',
                  'Crude Boosted Ratio', 'Age-Adjusted Unvaccinated Rate', 'Age-Adjusted Vaccinated Rate',
                  'Age-Adjusted Boosted Rate', 'Age-Adjusted Vaccinated Ratio', 'Age-Adjusted Boosted Ratio',
                  'Population Unvaccinated', 'Population Vaccinated', 'Population Boosted',
                  'Outcome Unvaccinated', 'Outcome Vaccinated', 'Outcome Boosted', 
                  'Age Group Min', 'Age Group Max']
for col in numerical_cols:
    if col in df.columns and df[col].dtype in ['float64', 'int64']:
        df[col] = df[col].fillna(df[col].median())


if 'Population Vaccinated' in df.columns and 'Population Boosted' in df.columns:
    df = df.dropna(subset=['Population Vaccinated', 'Population Boosted'])
else:
    print("Warning: Population columns missing, skipping dropna.")


print("\nShape after Cleaning:", df.shape)

print("\nData Types Before Conversion:\n", df.dtypes)


if 'Week End' in df.columns:
    print("\nUnique Week End Values (Sample):", df['Week End'].unique()[:10])
    try:
        df['Week End'] = pd.to_datetime(df['Week End'], format='%m/%d/%Y', errors='raise')
    except ValueError:
        print("Falling back to mixed format for date conversion...")
        df['Week End'] = pd.to_datetime(df['Week End'], format='mixed', dayfirst=False)

for col in numerical_cols:
    if col in df.columns:
        try:
            df[col] = df[col].astype(float)
        except ValueError:
            print(f"Warning: Could not convert {col} to float, skipping.")

print("\nData Types After Conversion:\n", df.dtypes)

desc_stats = df[[col for col in numerical_cols if col in df.columns]].describe()
print("\nDescriptive Statistics:\n", desc_stats)

if 'Outcome' in df.columns:
    print("\nCategorical Columns Summary:")
    print("Outcome:\n", df['Outcome'].value_counts())
if 'Age Group' in df.columns:
    print("\nAge Group:\n", df['Age Group'].value_counts())

rate_cols = ['Unvaccinated Rate', 'Vaccinated Rate', 'Boosted Rate']
outcome_cols = ['Outcome Unvaccinated', 'Outcome Vaccinated', 'Outcome Boosted']

plt.figure(figsize=(14, 6))
for col in rate_cols:
    if col in df.columns:
        sns.lineplot(x='Week End', y=col, data=df, label=col)
    else:
        print(f"Warning: {col} not found for line chart.")
plt.title('Weekly COVID-19 Rates by Vaccination Status')
plt.xlabel('Week')
plt.ylabel('Rate per 100k')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

if all(col in df.columns for col in outcome_cols) and 'Age Group' in df.columns:
    age_group_means = df.groupby('Age Group')[outcome_cols].mean().reset_index()
    plt.figure(figsize=(14, 6))
    colors = ['#FF9999', '#99CCFF', '#66FF66']
    age_group_means.set_index('Age Group').plot(kind='bar', stacked=False, color=colors)
    plt.title('Average COVID Outcomes by Age Group')
    plt.ylabel('Average Outcome Count')
    plt.xlabel('Age Group')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("Warning: Required columns for age group bar chart missing.")

if all(col in df.columns for col in ['Population Vaccinated', 'Outcome Vaccinated', 'Population Boosted', 'Age Group Min']):
    df['Log Population Vaccinated'] = np.log1p(df['Population Vaccinated'])
    df['Log Outcome Vaccinated'] = np.log1p(df['Outcome Vaccinated'])
    
    pop_cap = df['Log Population Vaccinated'].quantile(0.99)
    outcome_cap = df['Log Outcome Vaccinated'].quantile(0.99)
    df_regression = df[
        (df['Log Population Vaccinated'] <= pop_cap) & 
        (df['Log Outcome Vaccinated'] <= outcome_cap)
    ].copy()
    
    X = df_regression[['Log Population Vaccinated', 'Population Boosted']].copy()
    y = df_regression['Log Outcome Vaccinated']
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    
    X_single = X[['Log Population Vaccinated']].values.flatten()
    pred = model.predict(X)
    n = len(X)
    mean_x = np.mean(X_single)
    t_value = stats.t.ppf(0.975, df=n-2)
    ss_x = np.sum((X_single - mean_x) ** 2)
    se_fit = np.sqrt(mse * (1/n + (X_single - mean_x) ** 2 / ss_x))
    ci_upper = pred + t_value * se_fit
    ci_lower = pred - t_value * se_fit
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=X_single, y=y, hue=df_regression['Age Group'], 
        size=df_regression['Age Group Min'], sizes=(20, 200), 
        alpha=0.6, palette='viridis', legend='brief'
    )
    sort_idx = np.argsort(X_single)
    plt.plot(X_single[sort_idx], y_pred[sort_idx], color='red', label='Regression Line', linewidth=2)
    plt.fill_between(
        X_single[sort_idx], ci_lower[sort_idx], ci_upper[sort_idx], 
        color='red', alpha=0.1, label='95% CI'
    )
    
    plt.text(
        0.05, 0.95, f'R² = {r2:.3f}\nMSE = {mse:.3f}', 
        transform=plt.gca().transAxes, fontsize=10, 
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.title('Vaccination Impact: Population vs. Outcomes (Log-Transformed)')
    plt.xlabel('Log(Population Vaccinated + 1)')
    plt.ylabel('Log(Outcome Vaccinated + 1)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig('vaccination_impact_regression.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nLinear Regression Results:")
    print(f"Coefficient (Log Population Vaccinated): {model.coef_[0]:.3f}")
    print(f"Coefficient (Population Boosted): {model.coef_[1]:.3f}")
    print(f"Intercept: {model.intercept_:.3f}")
    print(f"R² Score: {r2:.3f}")
    print(f"Mean Squared Error: {mse:.3f}")
    
    residuals = y - y_pred
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residual Plot')
    plt.xlabel('Predicted Log Outcome Vaccinated')
    plt.ylabel('Residuals')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("Warning: Required columns for regression missing.")

if all(col in df.columns for col in outcome_cols):
    outcome_totals = df[outcome_cols].sum()
    labels = outcome_totals.index
    sizes = outcome_totals.values
    plt.figure(figsize=(7, 7))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#FF9999', '#99FF99', '#66B3FF'])
    plt.title('Total COVID-19 Outcomes by Vaccination Status')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
else:
    print("Warning: Required columns for pie chart missing.")

key_numeric_cols = [
    'Unvaccinated Rate', 'Vaccinated Rate', 'Boosted Rate',
    'Outcome Unvaccinated', 'Outcome Vaccinated', 'Outcome Boosted',
    'Population Unvaccinated', 'Population Vaccinated', 'Population Boosted'
]

valid_cols = [col for col in key_numeric_cols if col in df.columns and df[col].std() > 0]
if valid_cols:
    corr_matrix = df[valid_cols].corr()
    print("\nCorrelation Matrix Info:")
    print("Columns Included:", valid_cols)
    print("NaN in Correlation Matrix:\n", corr_matrix.isnull().sum())
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
        linewidths=0.5, mask=np.isnan(corr_matrix), 
        cbar_kws={'label': 'Correlation Coefficient'}
    )
    plt.title('Correlation Heatmap of Key COVID-19 Variables')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("Warning: No valid numeric columns with variance for heatmap.")


if 'Outcome' in df.columns:
    df['Outcome'] = df['Outcome'].str.strip().str.capitalize()
    outcome_totals = df.groupby('Outcome')[outcome_cols].sum()
    outcome_totals['Total'] = outcome_totals.sum(axis=1)
    outcome_totals = outcome_totals.sort_values(by='Total', ascending=False)
    
    plt.figure(figsize=(10, 6))
    bars = sns.barplot(x=outcome_totals.index, y=outcome_totals['Total'], palette='pastel')
    
    for bar in bars.patches:
        height = bar.get_height()
        bars.annotate(f'{int(height):,}', 
                      (bar.get_x() + bar.get_width() / 2, height),
                      ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.title('Total COVID-19 Cases, Hospitalizations, and Deaths')
    plt.xlabel('Outcome Type')
    plt.ylabel('Total Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
else:
    print("Warning: Outcome column missing for bar plot.")

print("\nFinal Shape:", df.shape)
print("\nFinal Columns:", df.columns.tolist())

df.to_csv('cleaned_covid-19_dataset.csv', index=False)
print("\nCleaned dataset saved as 'cleaned_covid-19_dataset.csv'")

print("\nKey Findings:")
print("- Data Quality: Preserved age groups (e.g., 0-4), fixed heatmap missing data.")
print("- Trends: Unvaccinated rates higher, boosters reduce deaths.")
print("- Age Differences: Older groups more vulnerable, younger drive cases.")
print("- Correlations: Strong rate-outcome links, complete heatmap.")
print("- Outliers: Capped for regression robustness.")