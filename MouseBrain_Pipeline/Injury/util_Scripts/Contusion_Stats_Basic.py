# Complete spinal cord contusion analysis with automatic logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import sys
import os
from datetime import datetime
from contextlib import contextmanager

# Logging functionality
class TeeOutput:
    """Writes to both terminal and log file simultaneously"""
    
    def __init__(self, log_filepath):
        self.terminal = sys.stdout
        self.log_file = open(log_filepath, 'w', encoding='utf-8')  # Fix Unicode issue
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        script_name = os.path.basename(sys.argv[0]) if sys.argv else "contusion_analysis"
        header = f"=== {script_name} - Complete Contusion Analysis Log ===\n"
        header += f"Started: {timestamp}\n"
        header += "=" * 60 + "\n\n"
        
        self.log_file.write(header)
        self.log_file.flush()
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        if hasattr(self, 'log_file'):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            footer = f"\n\n" + "=" * 60 + "\n"
            footer += f"Analysis completed: {timestamp}\n"
            footer += "=" * 60 + "\n"
            self.log_file.write(footer)
            self.log_file.close()

@contextmanager
def log_analysis(log_filename=None):
    if log_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"contusion_analysis_{timestamp}.txt"
    
    script_directory = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_directory, log_filename)
    
    original_stdout = sys.stdout
    tee = None
    
    try:
        tee = TeeOutput(log_path)
        sys.stdout = tee
        print(f"[LOGGING] Starting complete contusion analysis - saving to: {log_filename}")
        print()
        yield log_path
    finally:
        sys.stdout = original_stdout
        if tee:
            tee.close()
        print(f"[COMPLETE] Analysis logged to: {log_path}")

# Start the logged analysis
with log_analysis():
    
    # Load and examine data
    df = pd.read_excel('Contusion_Stats.xlsx')

    # Print data overview
    print("Dataset shape:", df.shape)
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nBasic statistics:")
    print(df.describe())

    # Check for missing values
    print("\nMissing values per column:")
    print(df.isnull().sum())

    # Convert survival to binary (1 for survived, 0 for died)
    df['Survived_Binary'] = (df['Survived'] == 'Y').astype(int)

    # Check the distribution of survival
    print("\nSurvival distribution:")
    print(df['Survived'].value_counts())
    print(f"Survival rate: {df['Survived_Binary'].mean():.2%}")

    # Compare survivors vs non-survivors for all parameters
    print("\n" + "="*60)
    print("=== COMPARISON: SURVIVORS vs NON-SURVIVORS ===")
    print("="*60)
    for column in ['Stage_Height', 'Actual_kd', 'Actual_displacement', 'Actual_Velocity']:
        survivors = df[df['Survived'] == 'Y'][column]
        non_survivors = df[df['Survived'] == 'N'][column]
        
        print(f"\n{column}:")
        print(f"  Survivors: Mean={survivors.mean():.2f}, Std={survivors.std():.2f}")
        print(f"  Non-survivors: Mean={non_survivors.mean():.2f}, Std={non_survivors.std():.2f}")
        
        # Statistical test
        t_stat, p_value = stats.ttest_ind(survivors, non_survivors)
        print(f"  T-test p-value: {p_value:.4f}")

    # Create visualizations (they will display but not be saved to log)
    print(f"\n{'='*60}")
    print("=== CREATING VISUALIZATIONS ===")
    print("="*60)
    print("Creating plots... (Check your screen for visual output)")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Spinal Cord Contusion Data Analysis', fontsize=16)

    # 1. Survival distribution
    axes[0, 0].pie(df['Survived'].value_counts(), labels=['Survived', 'Died'], 
                   autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Survival Distribution')

    # 2. Box plots for each parameter
    numeric_columns = ['Stage_Height', 'Actual_kd', 'Actual_displacement', 'Actual_Velocity']

    for i, column in enumerate(numeric_columns):
        row = (i + 1) // 3
        col = (i + 1) % 3
        
        sns.boxplot(data=df, x='Survived', y=column, ax=axes[row, col])
        axes[row, col].set_title(f'{column} by Survival')
        axes[row, col].set_xlabel('Survived')

    plt.tight_layout()
    plt.show()

    # Correlation matrix
    plt.figure(figsize=(10, 8))
    # Include only numeric columns for correlation
    numeric_df = df[numeric_columns + ['Survived_Binary']]
    correlation_matrix = numeric_df.corr()

    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.3f')
    plt.title('Correlation Matrix of Contusion Parameters')
    plt.show()

    # Detailed statistical analysis
    print(f"\n{'='*60}")
    print("=== DETAILED STATISTICAL TESTS ===")
    print("="*60)

    # Test each parameter for significant differences between groups
    parameters = ['Stage_Height', 'Actual_kd', 'Actual_displacement', 'Actual_Velocity']

    for param in parameters:
        survivors = df[df['Survived'] == 'Y'][param]
        non_survivors = df[df['Survived'] == 'N'][param]
        
        # Shapiro-Wilk test for normality
        _, p_norm_surv = stats.shapiro(survivors)
        _, p_norm_non = stats.shapiro(non_survivors)
        
        print(f"\n{param}:")
        print(f"  Normality tests - Survivors: p={p_norm_surv:.4f}, Non-survivors: p={p_norm_non:.4f}")
        
        # Choose appropriate test based on normality
        if p_norm_surv > 0.05 and p_norm_non > 0.05:
            # Both normal - use t-test
            t_stat, p_val = stats.ttest_ind(survivors, non_survivors)
            test_name = "T-test"
        else:
            # Non-normal - use Mann-Whitney U
            u_stat, p_val = stats.mannwhitneyu(survivors, non_survivors, alternative='two-sided')
            test_name = "Mann-Whitney U"
        
        print(f"  {test_name} p-value: {p_val:.4f}")
        print(f"  Significant: {'Yes' if p_val < 0.05 else 'No'}")

    # Focused analysis: Just survival vs kd
    print(f"\n{'='*60}")
    print("===== FOCUSED ANALYSIS: SURVIVAL vs ACTUAL_KD =====")
    print("="*60)

    # Look at the data first
    print("=== DATA OVERVIEW ===")
    print(f"Total subjects: {len(df)}")
    print(f"Survivors (Y): {sum(df['Survived'] == 'Y')}")
    print(f"Non-survivors (N): {sum(df['Survived'] == 'N')}")
    print()

    # Separate the Actual_kd values by survival group
    survivors_kd = df[df['Survived'] == 'Y']['Actual_kd']
    non_survivors_kd = df[df['Survived'] == 'N']['Actual_kd']

    # Basic statistics for each group
    print("=== ACTUAL_KD BY SURVIVAL GROUP ===")
    print(f"Survivors:")
    print(f"  Count: {len(survivors_kd)}")
    print(f"  Mean: {survivors_kd.mean():.2f}")
    print(f"  Std: {survivors_kd.std():.2f}")
    print(f"  Range: {survivors_kd.min():.1f} - {survivors_kd.max():.1f}")

    print(f"\nNon-survivors:")
    print(f"  Count: {len(non_survivors_kd)}")
    print(f"  Mean: {non_survivors_kd.mean():.2f}")
    print(f"  Std: {non_survivors_kd.std():.2f}")
    print(f"  Range: {non_survivors_kd.min():.1f} - {non_survivors_kd.max():.1f}")

    print(f"\nDifference in means: {survivors_kd.mean() - non_survivors_kd.mean():.2f}")
    print()

    # Statistical significance test
    # Use t-test to compare the two groups
    t_statistic, p_value = stats.ttest_ind(survivors_kd, non_survivors_kd)

    print("=== STATISTICAL TEST RESULTS ===")
    print(f"T-statistic: {t_statistic:.4f}")
    print(f"P-value: {p_value:.4f}")

    # Interpret the p-value
    if p_value < 0.001:
        significance = "HIGHLY SIGNIFICANT (p < 0.001)"
    elif p_value < 0.01:
        significance = "VERY SIGNIFICANT (p < 0.01)"
    elif p_value < 0.05:
        significance = "SIGNIFICANT (p < 0.05)"
    else:
        significance = "NOT SIGNIFICANT (p ≥ 0.05)"

    print(f"Result: {significance}")

    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt(((len(survivors_kd)-1) * survivors_kd.var() + 
                         (len(non_survivors_kd)-1) * non_survivors_kd.var()) / 
                         (len(survivors_kd) + len(non_survivors_kd) - 2))
    cohens_d = (survivors_kd.mean() - non_survivors_kd.mean()) / pooled_std

    print(f"Effect size (Cohen's d): {cohens_d:.3f}")

    if abs(cohens_d) < 0.2:
        effect_size = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_size = "small"
    elif abs(cohens_d) < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large"

    print(f"Effect size interpretation: {effect_size}")
    print()

    # Create focused visualization for kd analysis
    print("Creating Actual_kd visualization... (Check your screen)")
    plt.figure(figsize=(10, 6))

    # Box plot
    plt.subplot(1, 2, 1)
    data_to_plot = [survivors_kd, non_survivors_kd]
    labels = ['Survived', 'Died']
    plt.boxplot(data_to_plot, tick_labels=labels)  # Fix matplotlib deprecation warning
    plt.ylabel('Actual_kd')
    plt.title('Actual_kd by Survival Status')
    plt.grid(True, alpha=0.3)

    # Histogram
    plt.subplot(1, 2, 2)
    plt.hist(survivors_kd, alpha=0.7, label='Survived', bins=8, color='blue')
    plt.hist(non_survivors_kd, alpha=0.7, label='Died', bins=8, color='red')
    plt.xlabel('Actual_kd')
    plt.ylabel('Frequency')
    plt.title('Distribution of Actual_kd')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Final summary
    print(f"\n{'='*60}")
    print("=== FINAL SUMMARY ===")
    print("="*60)
    if p_value < 0.05:
        direction = "higher" if survivors_kd.mean() > non_survivors_kd.mean() else "lower"
        print(f"RESULT: Actual_kd is significantly {direction} in survivors")
        print(f"  Mean difference: {abs(survivors_kd.mean() - non_survivors_kd.mean()):.2f}")
        print(f"  This is a {effect_size} effect size")
    else:
        print("RESULT: No significant difference in Actual_kd between survivors and non-survivors")
        
    print(f"\nFor your paper: p = {p_value:.4f}, d = {cohens_d:.3f}")
    
    print(f"\n{'='*60}")
    print("=== ANALYSIS COMPLETE ===")
    print("="*60)
    print("All statistical results have been logged to the text file.")
    print("Plots are displayed on screen but not saved to the log.")
    print("Check the generated log file for a complete record of all statistical output.")