# MANOVA Analysis: Do survivors and non-survivors differ across multiple injury parameters simultaneously?
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import sys
import os
from datetime import datetime
from contextlib import contextmanager

# Logging functionality
class TeeOutput:
    """Writes to both terminal and log file simultaneously"""
    
    def __init__(self, log_filepath):
        self.terminal = sys.stdout
        self.log_file = open(log_filepath, 'w', encoding='utf-8')
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        script_name = os.path.basename(sys.argv[0]) if sys.argv else "manova_analysis"
        header = f"=== {script_name} - MANOVA Analysis Log ===\n"
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
            footer += f"MANOVA Analysis completed: {timestamp}\n"
            footer += "=" * 60 + "\n"
            self.log_file.write(footer)
            self.log_file.close()

@contextmanager
def log_analysis(log_filename=None):
    if log_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"manova_analysis_{timestamp}.txt"
    
    script_directory = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(script_directory, log_filename)
    
    original_stdout = sys.stdout
    tee = None
    
    try:
        tee = TeeOutput(log_path)
        sys.stdout = tee
        print(f"[LOGGING] Starting MANOVA analysis - saving to: {log_filename}")
        print()
        yield log_path
    finally:
        sys.stdout = original_stdout
        if tee:
            tee.close()
        print(f"[COMPLETE] MANOVA analysis logged to: {log_path}")

# Custom MANOVA function (since Python doesn't have a built-in one)
def perform_manova(data, group_col, dependent_vars):
    """
    Perform MANOVA using Wilks' Lambda
    """
    from scipy.stats import f
    
    groups = data[group_col].unique()
    n_groups = len(groups)
    n_vars = len(dependent_vars)
    n_total = len(data)
    
    # Get group data
    group_data = {}
    group_means = {}
    group_sizes = {}
    
    for group in groups:
        group_subset = data[data[group_col] == group][dependent_vars]
        group_data[group] = group_subset.values
        group_means[group] = group_subset.mean().values
        group_sizes[group] = len(group_subset)
    
    # Calculate overall mean
    overall_mean = data[dependent_vars].mean().values
    
    # Calculate Sum of Squares matrices
    # Total Sum of Squares (T)
    T = np.zeros((n_vars, n_vars))
    for i, row in data[dependent_vars].iterrows():
        diff = row.values - overall_mean
        T += np.outer(diff, diff)
    
    # Between-groups Sum of Squares (B)
    B = np.zeros((n_vars, n_vars))
    for group in groups:
        diff = group_means[group] - overall_mean
        B += group_sizes[group] * np.outer(diff, diff)
    
    # Within-groups Sum of Squares (W)
    W = T - B
    
    # Calculate Wilks' Lambda
    try:
        W_inv = np.linalg.inv(W)
        wilks_lambda = np.linalg.det(W) / np.linalg.det(B + W)
    except:
        return None, None, None, None
    
    # Calculate F-statistic and p-value
    df1 = n_vars
    df2 = n_total - n_groups - n_vars + 1
    
    if df2 > 0:
        F_stat = ((1 - wilks_lambda) / wilks_lambda) * (df2 / df1)
        p_value = 1 - f.cdf(F_stat, df1, df2)
    else:
        F_stat = None
        p_value = None
    
    return wilks_lambda, F_stat, p_value, (df1, df2)

# Start the logged analysis
with log_analysis():
    
    # Load data
    df = pd.read_excel('Contusion_Stats.xlsx')
    
    # Define our variables
    injury_params = ['Stage_Height', 'Actual_kd', 'Actual_displacement', 'Actual_Velocity']
    
    print("="*60)
    print("=== MANOVA: MULTIVARIATE ANALYSIS OF INJURY PARAMETERS ===")
    print("="*60)
    print()
    print("Research Question: Do survivors and non-survivors differ across")
    print("multiple injury parameters simultaneously?")
    print()
    print(f"Dependent Variables: {', '.join(injury_params)}")
    print(f"Independent Variable: Survival (Y vs N)")
    print(f"Total sample size: {len(df)}")
    print(f"Survivors: {sum(df['Survived'] == 'Y')}")
    print(f"Non-survivors: {sum(df['Survived'] == 'N')}")
    print()
    
    # Check data completeness
    missing_data = df[injury_params].isnull().sum()
    print("=== DATA COMPLETENESS CHECK ===")
    for param in injury_params:
        print(f"{param}: {missing_data[param]} missing values")
    print()
    
    # Descriptive statistics by group
    print("=== DESCRIPTIVE STATISTICS BY GROUP ===")
    for param in injury_params:
        survivors = df[df['Survived'] == 'Y'][param]
        non_survivors = df[df['Survived'] == 'N'][param]
        
        print(f"\n{param}:")
        print(f"  Survivors:     Mean={survivors.mean():.3f}, SD={survivors.std():.3f}")
        print(f"  Non-survivors: Mean={non_survivors.mean():.3f}, SD={non_survivors.std():.3f}")
        print(f"  Difference:    {survivors.mean() - non_survivors.mean():.3f}")
    print()
    
    # Check correlations between dependent variables
    print("=== CORRELATIONS BETWEEN INJURY PARAMETERS ===")
    correlation_matrix = df[injury_params].corr()
    print(correlation_matrix.round(3))
    print()
    
    # Check for strong correlations (>0.7)
    strong_corrs = []
    for i in range(len(injury_params)):
        for j in range(i+1, len(injury_params)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                strong_corrs.append((injury_params[i], injury_params[j], corr_val))
    
    if strong_corrs:
        print("Strong correlations found (|r| > 0.7):")
        for var1, var2, corr in strong_corrs:
            print(f"  {var1} <-> {var2}: r = {corr:.3f}")
    else:
        print("No strong correlations (|r| > 0.7) found between injury parameters.")
    print()
    
    # MANOVA assumption checking
    print("=== MANOVA ASSUMPTION CHECKING ===")
    
    # 1. Normality check for each variable by group
    print("\n1. Normality Tests (Shapiro-Wilk):")
    for param in injury_params:
        survivors = df[df['Survived'] == 'Y'][param]
        non_survivors = df[df['Survived'] == 'N'][param]
        
        _, p_surv = stats.shapiro(survivors)
        _, p_non = stats.shapiro(non_survivors)
        
        print(f"  {param}:")
        print(f"    Survivors: p = {p_surv:.4f} {'(Normal)' if p_surv > 0.05 else '(Non-normal)'}")
        print(f"    Non-survivors: p = {p_non:.4f} {'(Normal)' if p_non > 0.05 else '(Non-normal)'}")
    
    # 2. Homogeneity of variances (Levene's test)
    print("\n2. Homogeneity of Variances (Levene's Test):")
    for param in injury_params:
        survivors = df[df['Survived'] == 'Y'][param]
        non_survivors = df[df['Survived'] == 'N'][param]
        
        _, p_levene = stats.levene(survivors, non_survivors)
        print(f"  {param}: p = {p_levene:.4f} {'(Equal variances)' if p_levene > 0.05 else '(Unequal variances)'}")
    print()
    
    # Perform MANOVA
    print("="*60)
    print("=== MANOVA RESULTS ===")
    print("="*60)
    
    # Create binary survival variable for analysis
    df['Survival_Binary'] = (df['Survived'] == 'Y').astype(int)
    
    wilks_lambda, F_stat, p_value, dfs = perform_manova(df, 'Survival_Binary', injury_params)
    
    if wilks_lambda is not None:
        print(f"Wilks' Lambda: {wilks_lambda:.6f}")
        print(f"F-statistic: {F_stat:.4f}")
        print(f"Degrees of freedom: {dfs[0]}, {dfs[1]}")
        print(f"P-value: {p_value:.6f}")
        print()
        
        if p_value < 0.001:
            significance = "HIGHLY SIGNIFICANT (p < 0.001)"
        elif p_value < 0.01:
            significance = "VERY SIGNIFICANT (p < 0.01)"
        elif p_value < 0.05:
            significance = "SIGNIFICANT (p < 0.05)"
        else:
            significance = "NOT SIGNIFICANT (p >= 0.05)"
        
        print(f"Result: {significance}")
        print()
        
        # Effect size (eta-squared)
        eta_squared = 1 - wilks_lambda
        print(f"Effect size (eta-squared): {eta_squared:.4f}")
        
        if eta_squared < 0.01:
            effect_size = "small"
        elif eta_squared < 0.06:
            effect_size = "medium"
        else:
            effect_size = "large"
        
        print(f"Effect size interpretation: {effect_size}")
        print()
        
    else:
        print("ERROR: Could not calculate MANOVA due to computational issues")
        print("This may be due to small sample size or collinearity")
        print()
    
    # Univariate follow-up tests (ALWAYS run these - regardless of MANOVA significance)
    print("="*60)
    print("=== UNIVARIATE FOLLOW-UP TESTS ===")
    print("="*60)
    print("(Running post hocs regardless of main effect - large effect sizes warrant investigation)")
    print("(Individual ANOVAs for each injury parameter)")
    print()
    
    significant_vars = []
    effect_sizes = []
    
    for param in injury_params:
        survivors = df[df['Survived'] == 'Y'][param]
        non_survivors = df[df['Survived'] == 'N'][param]
        
        # Perform t-test (equivalent to one-way ANOVA for 2 groups)
        t_stat, p_val = stats.ttest_ind(survivors, non_survivors)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(survivors)-1) * survivors.var() + 
                             (len(non_survivors)-1) * non_survivors.var()) / 
                             (len(survivors) + len(non_survivors) - 2))
        cohens_d = (survivors.mean() - non_survivors.mean()) / pooled_std
        
        # Interpret effect size
        if abs(cohens_d) < 0.2:
            effect_interp = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_interp = "small"
        elif abs(cohens_d) < 0.8:
            effect_interp = "medium"
        else:
            effect_interp = "large"
        
        print(f"{param}:")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_val:.4f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''}")
        print(f"  Cohen's d: {cohens_d:.4f} ({effect_interp} effect)")
        
        # Flag variables with medium+ effect sizes or significance
        if abs(cohens_d) >= 0.5 or p_val < 0.05:
            significant_vars.append(param)
            effect_sizes.append((param, abs(cohens_d), p_val))
            direction = "higher" if cohens_d > 0 else "lower"
            print(f"  --> {direction} in survivors (worth investigating)")
        
        print()
    
    # Sort by effect size
    effect_sizes.sort(key=lambda x: x[1], reverse=True)
    
    print("VARIABLES WITH MEDIUM+ EFFECT SIZES OR SIGNIFICANCE:")
    if effect_sizes:
        for param, effect_size, p_val in effect_sizes:
            print(f"  - {param}: d = {effect_size:.3f}, p = {p_val:.4f}")
    else:
        print("  - No variables with medium+ effect sizes found")
    print()
    
    # Discriminant Analysis to find which variables best separate groups
    print("="*60)
    print("=== LINEAR DISCRIMINANT ANALYSIS ===")
    print("="*60)
    print("(Which injury parameters best discriminate between survivors and non-survivors?)")
    print()
    
    # Prepare data for LDA
    X = df[injury_params].values
    y = df['Survival_Binary'].values
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_scaled, y)
    
    # Get discriminant coefficients
    coefficients = lda.coef_[0]
    
    print("Discriminant Function Coefficients:")
    print("(Higher absolute values indicate more important variables)")
    print()
    
    coef_data = []
    for i, param in enumerate(injury_params):
        coef_data.append((param, coefficients[i], abs(coefficients[i])))
    
    # Sort by absolute coefficient value
    coef_data.sort(key=lambda x: x[2], reverse=True)
    
    for param, coef, abs_coef in coef_data:
        print(f"  {param}: {coef:.4f}")
    
    print()
    print("Variable Importance Ranking:")
    for i, (param, coef, abs_coef) in enumerate(coef_data, 1):
        print(f"  {i}. {param} (|coef| = {abs_coef:.4f})")
    
    # Calculate classification accuracy
    y_pred = lda.predict(X_scaled)
    accuracy = np.mean(y_pred == y)
    print(f"\nClassification Accuracy: {accuracy:.1%}")
    print()
    
    # Create visualizations
    print("="*60)
    print("=== CREATING VISUALIZATIONS ===")
    print("="*60)
    print("Creating multivariate plots... (Check your screen)")
    print()
    
    # 1. Correlation heatmap
    plt.figure(figsize=(12, 10))
    
    # Subplot 1: Correlation matrix
    plt.subplot(2, 2, 1)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.3f', cbar_kws={'label': 'Correlation'})
    plt.title('Correlations Between Injury Parameters')
    
    # Subplot 2: Box plots for most important variable
    most_important_var = coef_data[0][0]
    plt.subplot(2, 2, 2)
    survivors_data = df[df['Survived'] == 'Y'][most_important_var]
    non_survivors_data = df[df['Survived'] == 'N'][most_important_var]
    
    plt.boxplot([survivors_data, non_survivors_data], tick_labels=['Survived', 'Died'])
    plt.ylabel(most_important_var)
    plt.title(f'Most Discriminating Variable: {most_important_var}')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: LDA projection
    plt.subplot(2, 2, 3)
    lda_scores = lda.transform(X_scaled)
    
    survivors_scores = lda_scores[y == 1]
    non_survivors_scores = lda_scores[y == 0]
    
    plt.hist(survivors_scores, alpha=0.7, label='Survived', bins=8, color='blue')
    plt.hist(non_survivors_scores, alpha=0.7, label='Died', bins=8, color='red')
    plt.xlabel('Discriminant Function Score')
    plt.ylabel('Frequency')
    plt.title('LDA Projection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Scatter plot of top 2 variables
    if len(coef_data) >= 2:
        var1, var2 = coef_data[0][0], coef_data[1][0]
        plt.subplot(2, 2, 4)
        
        survivors_df = df[df['Survived'] == 'Y']
        non_survivors_df = df[df['Survived'] == 'N']
        
        plt.scatter(survivors_df[var1], survivors_df[var2], 
                   alpha=0.7, label='Survived', color='blue', s=50)
        plt.scatter(non_survivors_df[var1], non_survivors_df[var2], 
                   alpha=0.7, label='Died', color='red', s=50)
        plt.xlabel(var1)
        plt.ylabel(var2)
        plt.title(f'Top 2 Discriminating Variables')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Final summary
    print("="*60)
    print("=== FINAL SUMMARY ===")
    print("="*60)
    
    if wilks_lambda is not None and p_value is not None:
        if p_value < 0.05:
            print("MANOVA RESULT: SIGNIFICANT multivariate difference found!")
            print(f"  - Survivors and non-survivors differ significantly across the")
            print(f"    combination of all four injury parameters")
            print(f"  - Wilks' Lambda = {wilks_lambda:.4f}, p = {p_value:.4f}")
            print(f"  - Effect size: {effect_size} (eta² = {eta_squared:.4f})")
        else:
            print("MANOVA RESULT: No significant multivariate difference")
            print(f"  - p = {p_value:.4f} (not significant)")
            print(f"  - However, effect size is {effect_size} (eta² = {eta_squared:.4f})")
    
    print(f"\nMost Important Discriminating Variables:")
    for i, (param, coef, abs_coef) in enumerate(coef_data[:2], 1):
        print(f"  {i}. {param}")
    
    print(f"\nClassification accuracy using all variables: {accuracy:.1%}")
    
    print(f"\nFor your paper:")
    if wilks_lambda is not None:
        print(f"  MANOVA: Wilks' Lambda = {wilks_lambda:.4f}, F({dfs[0]},{dfs[1]}) = {F_stat:.2f}, p = {p_value:.4f}")
    
    print(f"\nInterpretation:")
    if p_value is not None and p_value < 0.05:
        print(f"  The overall pattern of injury parameters differs significantly between")
        print(f"  survivors and non-survivors, suggesting that survival depends on the")
        print(f"  combination of injury characteristics rather than any single parameter.")
    else:
        print(f"  While the multivariate test wasn't significant (likely due to small sample size),")
        if effect_sizes:
            print(f"  several variables show medium-large effect sizes worth investigating:")
            for param, effect_size, p_val in effect_sizes[:3]:  # Top 3
                print(f"    - {param} (d = {effect_size:.3f})")
        print(f"  These patterns suggest biological relevance that warrants larger studies.")