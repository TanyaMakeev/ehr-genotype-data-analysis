from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_distributions(df, feature_name, title, x_label, y_label, stat = "count", count_distribution=False):
    """
    Plots distributions of key variables.
    
    Parameters:
    - df: DataFrame containing the data
    - feature_name: Column name to plot
    - title: Plot title
    - x_label: X-axis label
    - y_label: Y-axis label
    - count_distribution: If True, plots distribution of counts per unique value.
                        If False, plots the feature values directly.
    """
    plt.figure(figsize=(10, 6))
    
    if count_distribution:
        # Count records per unique value (original behavior)
        rows_per_value = (
            df.groupby(feature_name)
               .size()
               .reset_index(name="n_records")
        )
        sns.histplot(
            data=rows_per_value,
            x="n_records",
            discrete=True,
            stat = stat
        )
        plt.xticks(
            range(
                rows_per_value["n_records"].min(),
                rows_per_value["n_records"].max() + 1
            )
        )
    else:
        # Plot feature values directly (new behavior)
        sns.histplot(
            data=df,
            x=feature_name,
            discrete=True,
            stat = stat
        )
        plt.xticks(
            range(
                int(df[feature_name].min()),
                int(df[feature_name].max()) + 1
            )
        )
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()

def double_barplot(data, feature1, feature2, hue, plt_title, hue_title, order = None):
    """
    Plots barplots of key variables.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=data,
        x=feature1,
        y=feature2,
        hue=hue,
        palette = 'viridis',
        order = order
    )
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title(plt_title)
    plt.xticks(rotation=45)
    plt.legend(title=hue_title, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def labeled_barplot(data, feature, perc=True, n=None):
    """
    Create a labeled bar plot showing frequency distribution of categorical data.

    Parameters:
    -----------
    data : pandas.DataFrame
        The input dataframe containing the feature to plot
    feature : str
        The column name to create frequency distribution for
    perc : bool, default=True
        If True, display percentages; if False, display counts
    n : int or None, default=None
        Number of top categories to display. If None, display all categories

    Returns:
    --------
    matplotlib.axes.Axes or None
        Returns axes object if show_plot=False, otherwise displays the plot
    """
    # Calculate total observations and unique categories
    total = len(data[feature])
    count = data[feature].nunique()

    # Set figure size based on number of categories
    if n is None:
        plt.figure(figsize=(count + 1, 5))
    else:
        plt.figure(figsize=(n + 1, 5))

    # Configure plot aesthetics
    plt.xticks(rotation=90, fontsize=15)
    plt.title(f"Frequency Distribution of {feature}", fontsize=15)

    # Create count plot with ordered categories
    ax = sns.countplot(
        data=data,
        x=feature,
        hue=feature,      # Match x to hue to fix deprecation warning
        palette="Paired",
        legend=False,     # Remove redundant legend
        order=data[feature].value_counts().index[:n].sort_values(),
    )

    # Add percentage/count labels to each bar
    for p in ax.patches:
        if perc == True:
            # Calculate percentage of total
            label = "{:.1f}%".format(100 * p.get_height() / total)
        else:
            # Use raw count
            label = p.get_height()

        # Calculate label position (center of bar)
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()

        # Add annotation with slight offset for better visibility
        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )

    plt.show()

# Horizontal sorted barplot
def horizontal_barplot(data, x,y,x_label,y_label,title):
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=data,
        x=x,
        y=y
    )
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def genotype_distribution(snp_info,genotype_df):
    snp_list = list(snp_info.keys())
    # Transform data for visualization: one row per patient-SNP combination
    genotype_melted = genotype_df.melt(
        id_vars=['patient_id'],
        value_vars=snp_list,
        var_name='SNP',
        value_name='Genotype'
    )

    # Create stacked bar plot showing genotype distribution for each SNP
    plt.figure(figsize=(12, 8))
    (pd.crosstab(genotype_melted['SNP'], genotype_melted['Genotype'], normalize='index') * 100).plot(
        kind='bar', stacked=True, colormap='crest', ax=plt.gca()
    )
    plt.title('Genotype Distribution Across SNPs (0=No Risk, 1=Heterozygous, 2=Homozygous Risk)')
    plt.xlabel('SNP Identifier')
    plt.ylabel('Percentage of Patients')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Genotype', bbox_to_anchor=(1.2, 1), loc='upper right')
    plt.tight_layout()
    plt.show()

def correlation_analysis(df):
    numerical_columns = df.select_dtypes(include=np.number).columns.tolist()

    plt.figure(figsize=(15, 10))
    sns.heatmap(
        df[numerical_columns].corr(numeric_only=True),
        annot=True,
        vmin=-1,
        vmax=1,
        fmt=".2f",
        cmap="Spectral"
    )
    plt.title('Correlation Matrix of Numerical Variables')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0, ha='right')
    plt.tight_layout()
    plt.show()

def plot_reactions(df):
    # Prevalence of different allergic reaction phenotypes
    # This shows which reaction types are most common in the population
    
    # Dynamically find all reaction columns (those starting with 'reaction_')
    reaction_phenotype_columns = [col for col in df.columns if col.startswith('reaction_')]
    
    if not reaction_phenotype_columns:
        print("No reaction columns found in DataFrame")
        return
    
        # Calculate prevalence as proportion of patients with each reaction type
    # Since these columns contain 0s and 1s, the mean directly tells you the proportion of patients who experienced that reaction.
    reaction_prevalence = df[reaction_phenotype_columns].mean().sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    (reaction_prevalence * 100).plot(kind="bar", color='skyblue', edgecolor='navy')
    plt.ylabel("Proportion of Patients (%)")
    plt.title("Prevalence of Allergic Reaction Phenotypes")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def gender_severity(df):
    # Filter out missing severity values (-1 indicates no severity data recorded)
    severity_gender_data = df[df['max_severity'] != -1].copy()
    
    if severity_gender_data.empty:
        print("No severity data available to plot")
        return

    # Calculate severity percentages within each gender
    gender_severity_counts = severity_gender_data.groupby(['gender', 'max_severity']).size().reset_index(name='count')
    gender_totals = severity_gender_data.groupby('gender').size().reset_index(name='total')

    # Merge to calculate percentages
    gender_severity_percent = gender_severity_counts.merge(gender_totals, on='gender')
    gender_severity_percent['percentage'] = (gender_severity_percent['count'] / gender_severity_percent['total']) * 100

    # Order severity levels for consistent plotting
    severity_levels = severity_gender_data['max_severity'].value_counts().index.sort_values(ascending=False)

    # Create grouped bar plot
    double_barplot(gender_severity_percent, 'max_severity', 'percentage', 'gender', 'Allergy Severity Distribution by Gender',
     'gender', order = severity_levels)

def plot_urticaria_genotype(df, snp_column_list):

    # Calculate urticaria prevalence for each genotype across all SNPs
    urticaria_genotype_data = []

    for snp_identifier in snp_column_list:
        # Check if urticaria column exists
        urticaria_col = 'reaction_urticaria'
        if urticaria_col not in df.columns:
            print(f"ERROR: Column {urticaria_col} not found. Available columns: {reaction_cols}")
            return
            
        # Group by genotype and calculate urticaria prevalence
        # Mean of binary variable (0/1) gives proportion of patients with urticaria
        genotype_prevalence = df.groupby(snp_identifier)[urticaria_col].mean().reset_index()
        genotype_prevalence.columns = ['Genotype', 'Urticaria_Prevalence']
        genotype_prevalence['SNP'] = snp_identifier
        urticaria_genotype_data.append(genotype_prevalence)

    # Combine all SNP data for comprehensive visualization
    urticaria_genotype_df = pd.concat(urticaria_genotype_data)
        
    # Create grouped bar plot
    double_barplot(urticaria_genotype_df, 'SNP', 'Urticaria_Prevalence', 'Genotype', 'Urticaria Distribution by Genotype',
     'Genotype (0/1/2)')

def plot_severity_genotype(df, snp_column_list):
        
    # Calculate severity prevalence for each genotype across all SNPs
    severity_genotype_data = []
    
    if df.empty:
        print("DataFrame is empty - no data to plot")
        return
    
    for snp_identifier in snp_column_list:
        # Check if SNP column exists
        if snp_identifier not in df.columns:
            print(f"WARNING: SNP column '{snp_identifier}' not found. Available columns: {df.columns.tolist()}")
            continue
        
        # Filter out missing severity values (-1 indicates no data)
        severity_data = df[df['max_severity'] != -1].copy()
        
        print(f"\nSNP {snp_identifier}:")
        print(f"  Records before filtering: {len(df)}")
        print(f"  Records after filtering out -1: {len(severity_data)}")
        
        # Convert max_severity to numeric, coercing errors to NaN
        severity_data['max_severity'] = pd.to_numeric(severity_data['max_severity'], errors='coerce')
        
        # Drop any remaining NaN values
        severity_data = severity_data.dropna(subset=['max_severity'])
        
        print(f"  Records after numeric conversion: {len(severity_data)}")
        
        if severity_data.empty:
            print(f"  Skipping {snp_identifier} - no valid severity data")
            continue
            
        # Group by genotype and calculate mean severity
        genotype_prevalence = severity_data.groupby(snp_identifier)['max_severity'].mean().reset_index()
        genotype_prevalence.columns = ['Genotype', 'Severity_Prevalence']
        genotype_prevalence['SNP'] = snp_identifier
        severity_genotype_data.append(genotype_prevalence)

    if not severity_genotype_data:
        print("\nNo valid severity data available for plotting")
        print("This likely means all max_severity values are -1 (missing data)")
        return
        
    # Combine all SNP data for comprehensive visualization
    severity_genotype_df = pd.concat(severity_genotype_data)
        
    # Create grouped bar plot
    double_barplot(severity_genotype_df, 'SNP', 'Severity_Prevalence', 'Genotype', 'Severity Distribution by Genotype',
     'Genotype (0/1/2)')

def violin_plot(df, x, y, title, xlabel, ylabel):
    # Filter out missing severity values (-1 indicates no data)
    if y == 'max_severity' or x == 'max_severity':
        plot_data = df[df[y if y == 'max_severity' else x] != -1].copy()
        if plot_data.empty:
            print("No valid severity data available for violin plot")
            return
        
    else:
        plot_data = df
    
    plt.figure(figsize=(10, 7))

# Create violin plot comparing genetic risk scores by urticaria status
    sns.violinplot(
        data=plot_data,
        x=x,
        y=y,
        hue=x,
        inner='quartile',
        palette='viridis',
        legend=False
    )

    # Add individual data points for transparency
    sns.stripplot(
        data=plot_data,
        x=x,
        y=y,
        color='black',
        alpha=0.5,
        jitter=True,
        size=5
    )

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def regplot(df, x, y, title, xlabel, ylabel):
    plt.figure(figsize=(12, 8))
    
    sns.regplot(
        data=df,
        x=x,
        y=y,
        lowess=True,                    # Add locally weighted scatterplot smoothing
        scatter_kws={"alpha": 0.6, "s": 50},  # Semi-transparent points
        line_kws={"color": "red", "linewidth": 2}  # Prominent trend line
    )
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()