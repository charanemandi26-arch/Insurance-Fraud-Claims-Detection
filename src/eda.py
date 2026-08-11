import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('deep')

def create_eda():
    # Ensure figures directory exists
    os.makedirs('docs/figures', exist_ok=True)
    
    # Load dataset
    df = pd.read_csv('data/raw/insurance_claims.csv')
    df.replace('?', np.nan, inplace=True)
    
    # 1. Fraud vs Non-Fraud Distribution
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x='fraud_reported', data=df, palette='Set2')
    plt.title('Distribution of Fraudulent vs Non-Fraudulent Claims', fontsize=14, pad=15)
    plt.xlabel('Fraud Reported', fontsize=12)
    plt.ylabel('Number of Claims', fontsize=12)
    
    # Add percentages
    total = len(df)
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        x = p.get_x() + p.get_width() / 2 - 0.05
        y = p.get_y() + p.get_height() + 10
        ax.annotate(percentage, (x, y), size=12)
        
    plt.tight_layout()
    plt.savefig('docs/figures/fraud_distribution.png', dpi=300)
    plt.close()
    
    # 2. Incident Severity vs Fraud
    plt.figure(figsize=(10, 6))
    sns.countplot(x='incident_severity', hue='fraud_reported', data=df, palette='Set2', 
                  order=['Trivial Damage', 'Minor Damage', 'Major Damage', 'Total Loss'])
    plt.title('Fraud Reported by Incident Severity', fontsize=14, pad=15)
    plt.xlabel('Incident Severity', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title='Fraud Reported')
    plt.tight_layout()
    plt.savefig('docs/figures/incident_severity_vs_fraud.png', dpi=300)
    plt.close()
    
    # 3. Total Claim Amount Distribution
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='fraud_reported', y='total_claim_amount', data=df, palette='Set2')
    plt.title('Total Claim Amount vs Fraud', fontsize=14, pad=15)
    plt.xlabel('Fraud Reported', fontsize=12)
    plt.ylabel('Total Claim Amount ($)', fontsize=12)
    plt.tight_layout()
    plt.savefig('docs/figures/total_claim_amount_dist.png', dpi=300)
    plt.close()
    
    # 4. Correlation Matrix of Numerical Features
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    # Drop identifiers and completely missing columns
    numerical_cols = [col for col in numerical_cols if col not in ['policy_number', 'incident_hour_of_the_day', '_c39']]
    
    corr = df[numerical_cols].corr()
    
    # We just want a simple heatmap of claim amounts and some features
    selected_cols = ['months_as_customer', 'age', 'total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim']
    corr_subset = df[selected_cols].corr()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_subset, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Correlation Matrix of Key Numerical Features', fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig('docs/figures/correlation_matrix.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    print("Generating EDA charts...")
    create_eda()
    print("Charts saved to docs/figures/")
