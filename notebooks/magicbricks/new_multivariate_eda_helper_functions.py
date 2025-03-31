import os
import numpy as np
import pandas as pd
import ast
from scipy.stats import iqr,yeojohnson, skew, kurtosis
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

import missingno as msno

import regex as re
import matplotlib.gridspec as gridspec
import scipy.stats as stats

import warnings
warnings.filterwarnings("ignore")

#use to plot Boxen Plot, Box Plot, Bar Plot (Mean), Bar Plot (Median), Count Plot
def num_charts_plot(df, feature):
    """
    Alternative plots for a numerical feature with high cardinality:
      - Boxen Plot
      - Box Plot
      - Violin Plot
      - Line Plot of aggregated price by feature (assumed ordinal)
      - Heatmap of median price by feature
      - Jittered Strip Plot of Price by feature
      - Count Plot      
    """
    
    # Increase figure height to accommodate an additional plot (16 rows)
    fig = plt.figure(constrained_layout=True, figsize=(20, 35))
    grid = gridspec.GridSpec(ncols=2, nrows=16, figure=fig)
    
    # 1. Boxen Plot
    ax1 = fig.add_subplot(grid[0:2, :])
    ax1.set_title('Boxen Plot')
    sns.boxenplot(x=df[feature], y=df['price'], ax=ax1)
    ax1.tick_params(axis='x', rotation=90)
    
    # 2. Box Plot
    ax2 = fig.add_subplot(grid[2:4, :])
    ax2.set_title('Box Plot')
    sns.boxplot(x=df[feature], y=df['price'], ax=ax2)
    ax2.tick_params(axis='x', rotation=90)
    
    # 3. Violin Plot
    ax3 = fig.add_subplot(grid[4:6, :])
    ax3.set_title('Violin Plot')
    sns.violinplot(x=df[feature], y=df['price'], data=df, ax=ax3)
    ax3.tick_params(axis='x', rotation=90)
    
    # 4. Line Plot (Mean and Median)
    ax4 = fig.add_subplot(grid[6:8, :])
    ax4.set_title('Line Plot of Aggregated Price by ' + feature)
    
    # Compute aggregated values
    agg_data = df.groupby(feature)['price'].agg(['mean', 'median']).reset_index()
    agg_data = agg_data.sort_values(feature)
    
    # Plot mean and median
    ax4.plot(agg_data[feature], agg_data['mean'], marker='o', label='Mean Price', color='#1f77b4')
    ax4.plot(agg_data[feature], agg_data['median'], marker='o', label='Median Price', color='#ff7f0e')
    
    # Offsets for text placement
    y_offset_mean = 0.5
    y_offset_median = -0.5
    
    # Annotate mean (vertical text)
    for x_val, y_val in zip(agg_data[feature], agg_data['mean']):
        ax4.text(x_val, y_val + y_offset_mean, f"{y_val:.2f}",
                 ha='center', va='bottom', fontsize=10, color='#1f77b4', rotation=90)
    
    # Annotate median (vertical text)
    for x_val, y_val in zip(agg_data[feature], agg_data['median']):
        ax4.text(x_val, y_val + y_offset_median, f"{y_val:.2f}",
                 ha='center', va='top', fontsize=10, color='#ff7f0e', rotation=90)
    
    ax4.legend()
    ax4.tick_params(axis='x', rotation=90)
    ax4.set_xlabel(feature)
    ax4.set_ylabel('Mean and Median Price')
    
    # 5. Heatmap of Median Price by feature
    ax5 = fig.add_subplot(grid[8:10, :])
    ax5.set_title('Heatmap of Median Price by ' + feature)
    
    # Reuse the agg_data focusing on the median
    heat_data = agg_data.set_index(feature)[['median']].T
    
    sns.heatmap(
        heat_data,
        annot=True,           # show numeric values
        fmt=".2f",            # decimal format for numbers
        cmap="viridis", 
        ax=ax5,
        cbar=True,            # show colorbar
        linewidths=0.5,       # lines between cells
        linecolor='white',
        annot_kws={"size": 12, "rotation": 90}  # vertical annotation text
    )
    
    # Optionally rotate tick labels for x or y axes
    ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45, ha='right')
    ax5.set_yticklabels(ax5.get_yticklabels(), rotation=0)
    
    # 6. Jittered Strip Plot
    ax6 = fig.add_subplot(grid[10:12, :])
    ax6.set_title('Jittered Strip Plot of Price by ' + feature)
    
    # Compute the median price by feature and sort it
    table = df.groupby(feature, as_index=False)['price'].median().sort_values(feature)
    
    # Create the strip plot with sorted order
    a1 = sns.stripplot(data=df, x=feature, y='price', order=table[feature].tolist(), 
                       jitter=True, alpha=0.6, size=5, ax=ax6)
    a1.set_xticklabels(a1.get_xticklabels(), rotation=90)
    
    # 7. Count Plot
    ax7 = fig.add_subplot(grid[12:14, :])
    ax7.set_title('Count Plot')
    sns.countplot(x=df[feature], ax=ax7)
    ax7.tick_params(axis='x', rotation=90)
    
    plt.show()


#Confidence Interval comparison
def ci_mean(data, confidence=0.95):
    """Compute the confidence interval for the mean."""
    n = len(data)
    mean = np.mean(data)
    sem = st.sem(data)
    h = sem * st.t.ppf((1 + confidence) / 2., n-1)
    return mean, mean - h, mean + h

def ci_median(data, confidence=0.95, n_bootstrap=1000):
    """Compute the confidence interval for the median using bootstrapping."""
    boot_medians = [np.median(np.random.choice(data, size=len(data), replace=True)) 
                    for _ in range(n_bootstrap)]
    med = np.median(data)
    lower = np.percentile(boot_medians, (1-confidence)/2 * 100)
    upper = np.percentile(boot_medians, (1+confidence)/2 * 100)
    return med, lower, upper

def bar_plot_mean_median_ci(df, feature, price_col='price', confidence=0.95):
    """
    Create a bar plot with error bars to compare the mean and median price 
    (with their confidence intervals) for each level of the feature and annotate the CI.
    """
    # Group data by the feature
    groups = df.groupby(feature)[price_col]
    results = []
    
    for name, group in groups:
        m, m_low, m_high = ci_mean(group, confidence)
        med, med_low, med_high = ci_median(group, confidence)
        results.append((name, m, m_low, m_high, med, med_low, med_high))
        
    # Build DataFrame and sort if needed
    res_df = pd.DataFrame(results, columns=[feature, 'mean', 'mean_low', 'mean_high', 
                                              'median', 'median_low', 'median_high'])
    res_df.sort_values(feature, inplace=True)
    
    # Calculate error bars
    res_df['mean_err_low'] = res_df['mean'] - res_df['mean_low']
    res_df['mean_err_high'] = res_df['mean_high'] - res_df['mean']
    # For median, we compute asymmetric error bars
    res_df['median_err_low'] = res_df['median'] - res_df['median_low']
    res_df['median_err_high'] = res_df['median_high'] - res_df['median']
    
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.35  # width of the bars
    indices = np.arange(len(res_df))
    
    # Bar for mean
    bars_mean = ax.bar(indices - width/2, res_df['mean'], width, 
                       yerr=[res_df['mean_err_low'], res_df['mean_err_high']], 
                       capsize=5, label='Mean Price', color='#1f77b4')
    
    # Bar for median
    bars_median = ax.bar(indices + width/2, res_df['median'], width, 
                         yerr=[res_df['median_err_low'], res_df['median_err_high']], 
                         capsize=5, label='Median Price', color='#ff7f0e')
    
    # Annotate the CI on each bar
    for i, row in res_df.iterrows():
        # Mean annotation
        ax.text(i - width/2, row['mean_high'] + 0.5, 
                f"CI: [{row['mean_low']:.1f}, {row['mean_high']:.1f}]", 
                ha='center', va='bottom', fontsize=9, color='#1f77b4')
        # Median annotation
        ax.text(i + width/2, row['median_high'] + 0.5, 
                f"CI: [{row['median_low']:.1f}, {row['median_high']:.1f}]", 
                ha='center', va='bottom', fontsize=9, color='#ff7f0e')
    
    ax.set_xlabel(feature)
    ax.set_ylabel('Price')
    ax.set_title('Mean and Median Price with 95% Confidence Intervals by ' + feature)
    ax.set_xticks(indices)
    ax.set_xticklabels(res_df[feature], rotation=90)
    ax.legend()
    
    plt.show()

#use to plot Distribution Plot and Scatter Plot
def num_two_chart_plot(df, feature):
    """
    The plots include:
    - Distribution Plot
    - Scatter Plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Distribution Plot
    sns.histplot(df[feature], kde=True, ax=axes[0])
    axes[0].set_title('Distribution Plot', fontsize=12)
    axes[0].set_xlabel(feature, fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].tick_params(axis='both', labelsize=10)
    axes[0].tick_params(axis='x', rotation=90)  # Apply rotation to x-axis

    # Scatter Plot
    sns.scatterplot(x=df[feature], y=df['price'], ax=axes[1])
    axes[1].set_title('Scatter Plot', fontsize=12)
    axes[1].set_xlabel(feature, fontsize=12)
    axes[1].set_ylabel('Price', fontsize=12)
    axes[1].tick_params(axis='both', labelsize=10)
    axes[1].tick_params(axis='x', rotation=90)  # Apply rotation to x-axis

    # Tight layout to reduce overlapping
    plt.tight_layout()
    plt.show()

#use to plot count plot, bar plot, KDE plot
def region_plots(df, feature):
    """
    Plots various relationships between the given feature and other attributes, ensuring consistent colors for 'addressregion'.
    """

    # Define a fixed color palette for 'addressregion'
    unique_regions = df['addressregion'].dropna().unique()
    palette = sns.color_palette("tab10", len(unique_regions))  # Using 'tab10' for distinct colors
    region_palette = dict(zip(unique_regions, palette))  # Mapping colors to each region

    # Create a figure with a grid of 4 rows x 2 columns
    fig = plt.figure(constrained_layout=True, figsize=(30, 30))
    grid = gridspec.GridSpec(ncols=3, nrows=5, figure=fig)

    # 1. Count plot
    ax1 = fig.add_subplot(grid[2, 0])
    ax1.set_title(f"1.Count Plot of '{feature}' by 'addressregion'")
    sns.countplot(x=feature, hue='addressregion', data=df, ax=ax1, palette=region_palette)

    # 2. Bar plot: price_category vs feature
    ax2 = fig.add_subplot(grid[2, 1])
    ax2.set_title(f"2. Barplot of '{feature}' mean and price categories")
    sns.barplot(x='price_category', y=feature, data=df, ax=ax2, estimator=np.mean, errorbar=None)
    ax2.tick_params(axis='x', rotation=90)

    # 3. Bar plot: price vs feature by addressregion
    ax3 = fig.add_subplot(grid[0, 0])
    ax3.set_title(f"3. Bar Plot of 'price' vs '{feature}' by 'addressregion'")
    sns.barplot(x=feature, y='price', hue='addressregion', data=df, ax=ax3, palette=region_palette)

    # 4. Bar plot: area vs feature by addressregion
    ax4 = fig.add_subplot(grid[0, 1])
    ax4.set_title(f"4. Bar Plot of 'area' vs '{feature}' by 'addressregion'")
    sns.barplot(x=feature, y='area', hue='addressregion', data=df, ax=ax4, palette=region_palette)

    # 5. Bar plot: costpersqft vs feature by addressregion
    ax5 = fig.add_subplot(grid[0, 2])
    ax5.set_title(f"5. Bar Plot of 'costpersqft' vs '{feature}' by 'addressregion'")
    sns.barplot(x=feature, y='costpersqft', hue='addressregion', data=df, ax=ax5, palette=region_palette)

    # 6. Bar plot: feature vs addressregion
    ax6 = fig.add_subplot(grid[2, 2])
    ax6.set_title(f"6. Bar Plot of '{feature}' mean vs 'addressregion'")
    sns.barplot(x='addressregion', y=feature, data=df, ax=ax6, estimator=np.mean, errorbar=None, palette=region_palette)

    # 7. KDE plot: Price distribution by feature
    ax7 = fig.add_subplot(grid[3, 0])
    ax7.set_title(f"7. Price Distribution by '{feature}' (KDE Plot)")
    sns.kdeplot(data=df, x="price", hue=feature, fill=True, common_norm=False, palette="tab10", linewidth=1.5, ax=ax7, warn_singular=False)
    ax7.set_xlabel("Price")
    ax7.set_ylabel("Density")

    # 8. Price vs Bath (line plot)
    ax8 = fig.add_subplot(grid[1, 0])
    ax8.set_title(f"8. Price vs {feature}")
    sns.lineplot(data=df, x=feature, y="price", hue="addressregion", ax=ax8, palette=region_palette)

    # 9. Area vs Bath (line plot)
    ax9 = fig.add_subplot(grid[1, 1])
    ax9.set_title(f"9. Area vs {feature}")
    sns.lineplot(data=df, x=feature, y="area", hue="addressregion", ax=ax9, palette=region_palette)

    # 10. Cost per Sqft vs Bath (line plot)
    ax10 = fig.add_subplot(grid[1, 2])
    ax10.set_title(f"10. Cost per Sqft vs {feature}")
    sns.lineplot(data=df, x=feature, y="costpersqft", hue="addressregion", ax=ax10, palette=region_palette)

    plt.show()

#median plot
def median_plot(df, feature):
    # Drop rows with missing values for relevant columns
    df = df.dropna(subset=[feature, 'costpersqft', 'area', 'price'])
    
    # Compute median values for each category in the feature
    median_cost = df.groupby(feature)['costpersqft'].median()
    median_area = df.groupby(feature)['area'].median()
    median_price = df.groupby(feature)['price'].median()
    
    # Apply log transformation
    log_median_cost = np.log1p(median_cost)
    log_median_area = np.log1p(median_area)
    log_median_price = np.log1p(median_price)
    
    # Combine data for visualization
    median_df = (
        pd.DataFrame({
            feature: log_median_cost.index, 
            'Log Median Cost per Sqft': log_median_cost.values, 
            'Log Median Area': log_median_area.values, 
            'Log Median Price': log_median_price.values
        })
        .melt(id_vars=[feature], var_name='Metric', value_name='Log Value')
    )
    
    # Plot the graph
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=median_df, x=feature, y='Log Value', hue='Metric', marker='o')
    
    # Annotate each point
    for line in ax.lines:
        for x, y in zip(line.get_xdata(), line.get_ydata()):
            ax.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Set labels and title
    plt.xlabel(feature, fontsize=12)
    plt.ylabel("Log Median Values", fontsize=12)
    plt.title("Log Median Cost per Sqft, Area, and Price by " + feature + " (Line Plot)", fontsize=14)
    plt.xticks(rotation=45)
    
    # Move legend outside the plot
    plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, frameon=True)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    
    # Create a sorted DataFrame (from expensive to cheapest)
    sorted_df = pd.DataFrame({
        feature: log_median_cost.index, 
        'log_median_costpersqft_values': log_median_cost.values  # Renamed column
    }).sort_values(by='log_median_costpersqft_values', ascending=False)
    
    # Print the sorted table
    print(sorted_df.to_string(index=False))

#median plot for high cardinality
def median_plot_high_card(df, feature):
    # Drop rows with missing values for the relevant columns to avoid issues during groupby or log transform
    df = df.dropna(subset=[feature, 'costpersqft', 'area', 'price'])
    
    # Compute median values for each category in the feature
    median_cost = df.groupby(feature)['costpersqft'].median()
    median_area = df.groupby(feature)['area'].median()
    median_price = df.groupby(feature)['price'].median()
    
    # Apply log transformation using np.log1p to handle zeros gracefully
    median_cost = np.log1p(median_cost)
    median_area = np.log1p(median_area)
    median_price = np.log1p(median_price)
    
    # Combine data into a single DataFrame suitable for Seaborn plotting
    median_df = (
        pd.DataFrame({
            feature: median_cost.index, 
            'Log Median Cost per Sqft': median_cost.values, 
            'Log Median Area': median_area.values, 
            'Log Median Price': median_price.values
        })
        .melt(id_vars=[feature], var_name='Metric', value_name='Log Value')
    )
    
    # Create the line plot
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=median_df, x=feature, y='Log Value', hue='Metric', marker='o')
    
    # Set labels and title
    plt.xlabel(feature, fontsize=12)
    plt.ylabel("Log Median Values", fontsize=12)
    plt.title("Log Median Cost per Sqft, Area, and Price by " + feature + " (Line Plot)", fontsize=14)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability if needed
    
    # Move legend outside the plot area
    plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=12, frameon=True)
    
    # Adjust layout to accommodate the legend outside the plot
    plt.tight_layout()
    plt.show()

#plot_log_median_subplots
def plot_log_median_subplots(df, feature):
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharex=True, sharey=True)
    
    regions = ['mumbai', 'navi mumbai', 'palghar', 'thane']
    labels = ['Log Median costpersqft', 'Log Median area', 'Log Median price']
    columns = ['costpersqft', 'area', 'price']
    
    for i, region in enumerate(regions):
        ax = axes[i]
        subset = df[df['addressregion'] == region]
        
        median_values = subset.groupby(feature)[columns].median()
        log_median_values = np.log1p(median_values)
        
        for j, col in enumerate(columns):
            sns.lineplot(
                x=log_median_values.index,
                y=log_median_values[col],
                ax=ax,
                label=f"{labels[j]} for {region}",
                marker='o',
                linewidth=2
            )
            
            # Annotate each point
            for x_val, y_val in zip(log_median_values.index, log_median_values[col]):
                ax.text(x_val, y_val, f'{y_val:.2f}', ha='right', va='bottom', fontsize=12)
        
        ax.set_title(region.title(), fontsize=14, fontweight='bold')
        ax.set_xlabel(feature, fontsize=12)
        ax.set_ylabel("Log Median Values", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=12, frameon=True)

        # Rotate x-axis labels properly
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

#log median subplots for high cardinality
def plot_log_median_subplots_high_card(df, feature):
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharex=True, sharey=True)
    
    regions = ['mumbai', 'navi mumbai', 'palghar', 'thane']
    labels = ['Log Median costpersqft', 'Log Median area', 'Log Median price']
    columns = ['costpersqft', 'area', 'price']
    
    for i, region in enumerate(regions):
        ax = axes[i]
        subset = df[df['addressregion'] == region]
        
        median_values = subset.groupby(feature)[columns].median()
        log_median_values = np.log1p(median_values)
        
        for j, col in enumerate(columns):
            sns.lineplot(
                x=log_median_values.index,
                y=log_median_values[col],
                ax=ax,
                label=f"{labels[j]} for {region}",
                marker='o',
                linewidth=2
            )
        
        ax.set_title(region.title(), fontsize=14, fontweight='bold')
        ax.set_xlabel(feature, fontsize=12)
        ax.set_ylabel("Log Median Values", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=12, frameon=True)
    
    plt.tight_layout()
    plt.show()

#region plot for high cardinality features
def region_plots_high_card(df, feature):
    """
    Plots:
      1. Bar plot: price_category vs feature
      2. Bar plot: feature vs addressregion
      3. Line plot: Price vs feature (colored by addressregion)
      4. Line plot: Area vs feature (colored by addressregion)
      5. Line plot: Cost per Sqft vs feature (colored by addressregion)
    """
    # Create a color palette based on the number of unique regions
    region_palette = sns.color_palette("husl", n_colors=df["addressregion"].nunique())
    
    # Create a figure with a grid of 3 rows x 2 columns
    fig = plt.figure(constrained_layout=True, figsize=(20, 20))
    grid = gridspec.GridSpec(ncols=2, nrows=4, figure=fig)
    
    # 1. Bar plot: price_category vs feature at grid position (0, 0)
    ax1 = fig.add_subplot(grid[0, 0])
    ax1.set_title(f"1. Barplot of '{feature}' mean and price categories", fontsize=12)
    sns.barplot(
        x='price_category', y=feature, data=df, 
        estimator=np.mean, errorbar=None, ax=ax1
    )
    # Rotate tick labels for better readability
    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    for label in ax1.get_xticklabels():
        label.set_ha('right')  
    ax1.tick_params(axis='y', labelsize=14)
    ax1.set_xlabel('price_category', fontsize=14)
    ax1.set_ylabel(feature, fontsize=14)
    
    # 2. Bar plot: feature vs addressregion at grid position (0, 1)
    ax2 = fig.add_subplot(grid[0, 1])
    ax2.set_title(f"2. Bar Plot of '{feature}' mean vs 'addressregion'", fontsize=12)
    sns.barplot(
        x='addressregion', y=feature, data=df,
        estimator=np.mean, errorbar=None, ax=ax2
    )
    ax2.tick_params(axis='x', rotation=45, labelsize=10)
    ax2.tick_params(axis='y', labelsize=14)
    ax2.set_xlabel('addressregion', fontsize=14)
    ax2.set_ylabel(feature, fontsize=14)
    
    # 3. Line plot: Price vs feature at grid position (1, 0)
    ax3 = fig.add_subplot(grid[1, :])
    ax3.set_title(f"3. Price vs  + '{feature}'", fontsize=12)
    sns.lineplot(data=df, x=feature, y="price", hue="addressregion", ax=ax3, palette=region_palette)
    
    # 4. Line plot: Area vs feature at grid position (1, 1)
    ax4 = fig.add_subplot(grid[2, :])
    ax4.set_title(f"4. Area vs  + '{feature}'", fontsize=12)
    sns.lineplot(data=df, x=feature, y="area", hue="addressregion", ax=ax4, palette=region_palette)
    
    # 5. Line plot: Cost per Sqft vs feature at grid position (2, 0)
    ax5 = fig.add_subplot(grid[3, :])
    ax5.set_title(f"5. Cost per Sqft vs  + '{feature}'", fontsize=12)
    sns.lineplot(data=df, x=feature, y="costpersqft", hue="addressregion", ax=ax5, palette=region_palette)
    
    # If you want to hide the empty subplot at grid position (2, 1)
    ax6 = fig.add_subplot(grid[2, 1])
    ax6.axis('off')
    
    # Tight layout to ensure minimal overlapping
    plt.tight_layout()
    plt.show()

#summary table
def summarize_properties(df, feature):
    """
    Returns a summary DataFrame containing overall metrics and region-specific counts 
    and median prices for properties, grouped by the specified feature.
    """
    # Define regions of interest.
    regions = ['mumbai', 'navi mumbai', 'thane', 'palghar']
    
    # Group by the feature column to compute overall metrics.
    overall = df.groupby(feature).agg(
        price_mean=('price', 'mean'),
        price_median=('price', 'median'),
        count=('price', 'count')
    ).reset_index().rename(columns={feature: 'feature_value'})
    
    # Prepare lists to collect region-specific metrics for each unique feature value.
    region_counts = {region: [] for region in regions}
    region_median_prices = {region: [] for region in regions}
    
    unique_values = overall['feature_value'].unique()
    for val in unique_values:
        df_val = df[df[feature] == val]
        for region in regions:
            # Use a case-insensitive match on 'addressregion'
            region_df = df_val[df_val['addressregion'].str.lower() == region]
            region_counts[region].append(len(region_df))
            median_price = region_df['price'].median() if len(region_df) > 0 else None
            region_median_prices[region].append(median_price)
    
    # Add region-specific columns to the overall DataFrame.
    for region in regions:
        overall[region] = region_counts[region]
        overall[f"{region}_median_price"] = region_median_prices[region]
    
    return overall

#region plots for categorical features
def region_plots_for_categorical_features(df, feature):
    """
    Plots various relationships between the given feature and other attributes, ensuring consistent colors for 'addressregion' and feature.
    """

    # Define a fixed color palette for 'addressregion'
    unique_regions = df['addressregion'].dropna().unique()
    palette_regions = sns.color_palette("tab10", len(unique_regions))
    region_palette = dict(zip(unique_regions, palette_regions))

    # Define a separate palette for the 'feature'
    unique_feature_values = df[feature].dropna().unique()
    palette_feature = sns.color_palette("tab10", len(unique_feature_values))
    feature_palette = dict(zip(unique_feature_values, palette_feature))

    # Create a figure with a grid of 4 rows x 2 columns
    fig = plt.figure(constrained_layout=True, figsize=(30, 30))
    grid = gridspec.GridSpec(ncols=3, nrows=5, figure=fig)

    # 7. Count plot
    ax7 = fig.add_subplot(grid[2, 0])
    ax7.set_title(f"7. Count Plot of '{feature}' by 'addressregion'")
    sns.countplot(x=feature, hue='addressregion', data=df, ax=ax7, palette=region_palette)
    ax7.tick_params(axis='x', rotation=45)

    # 1. Bar plot: price vs feature by addressregion
    ax1 = fig.add_subplot(grid[0, 0])
    ax1.set_title(f"1. Bar Plot of 'price' vs '{feature}' by 'addressregion'")
    sns.barplot(x=feature, y='price', hue='addressregion', data=df, ax=ax1, palette=region_palette)
    ax1.tick_params(axis='x', rotation=45)

    # 2. Bar plot: area vs feature by addressregion
    ax2 = fig.add_subplot(grid[0, 1])
    ax2.set_title(f"2. Bar Plot of 'area' vs '{feature}' by 'addressregion'")
    sns.barplot(x=feature, y='area', hue='addressregion', data=df, ax=ax2, palette=region_palette)
    ax2.tick_params(axis='x', rotation=45)

    # 3. Bar plot: costpersqft vs feature by addressregion
    ax3 = fig.add_subplot(grid[0, 2])
    ax3.set_title(f"3. Bar Plot of 'costpersqft' vs '{feature}' by 'addressregion'")
    sns.barplot(x=feature, y='costpersqft', hue='addressregion', data=df, ax=ax3, palette=region_palette)
    ax3.tick_params(axis='x', rotation=45)

    # 10. KDE plot: Price distribution by feature
    ax10 = fig.add_subplot(grid[3, 0])
    ax10.set_title(f"10. Price Distribution by '{feature}' (KDE Plot)")
    sns.kdeplot(data=df, x="price", hue=feature, fill=True, common_norm=False, palette=feature_palette, linewidth=1.5, ax=ax10, warn_singular=False)
    ax10.set_xlabel("Price")
    ax10.set_ylabel("Density")

    # Skip plots 4, 5, and 6 if the feature is 'addressregion'
    if feature != 'addressregion':
        # 4. Price vs Feature (line plot)
        ax4 = fig.add_subplot(grid[1, 0])
        ax4.set_title(f"4. Price vs {feature}")
        sns.lineplot(data=df, x=feature, y="price", hue="addressregion", ax=ax4, palette=region_palette)
        ax4.tick_params(axis='x', rotation=45)

        # 5. Area vs Feature (line plot)
        ax5 = fig.add_subplot(grid[1, 1])
        ax5.set_title(f"5. Area vs {feature}")
        sns.lineplot(data=df, x=feature, y="area", hue="addressregion", ax=ax5, palette=region_palette)
        ax5.tick_params(axis='x', rotation=45)

        # 6. Cost per Sqft vs Feature (line plot)
        ax6 = fig.add_subplot(grid[1, 2])
        ax6.set_title(f"6. Cost per Sqft vs {feature}")
        sns.lineplot(data=df, x=feature, y="costpersqft", hue="addressregion", ax=ax6, palette=region_palette)
        ax6.tick_params(axis='x', rotation=45)

    # 8. Count plot with corrected palette
    ax8 = fig.add_subplot(grid[2, 1])
    ax8.set_title(f"8. Count Plot of '{feature}' by 'price_category'")
    sns.countplot(x='price_category', hue=feature, data=df, ax=ax8, palette=feature_palette)
    ax8.tick_params(axis='x', rotation=90)

    # 9. Median price vs feature by price_category
    ax9 = fig.add_subplot(grid[2, 2])
    ax9.set_title(f"9. Median Price by '{feature}' and 'price_category'")
    sns.barplot(x='price_category', y='price', hue=feature, data=df, ax=ax9, estimator=np.median, errorbar=None, palette=feature_palette)
    ax9.tick_params(axis='x', rotation=90)

    plt.show()


#plots for high cardinality categorical features
def num_charts_plot_high_card_categorical(df, feature, top_n=10):
    """
    Alternative plots for a categorical feature with high cardinality,
    displaying only the top N categories based on frequency.
    """
    # Get the top N categories based on count
    top_categories = df[feature].value_counts().nlargest(top_n).index.tolist()
    df_filtered = df[df[feature].isin(top_categories)]
    
    # Sort categories based on count
    df_filtered[feature] = pd.Categorical(df_filtered[feature], categories=top_categories, ordered=True)
    df_filtered = df_filtered.sort_values(feature)
    
    # Increase figure height to accommodate plots
    fig = plt.figure(constrained_layout=True, figsize=(20, 35))
    grid = gridspec.GridSpec(ncols=2, nrows=16, figure=fig)
    
    # 1. Boxen Plot
    ax1 = fig.add_subplot(grid[0:2, :])
    ax1.set_title('Boxen Plot')
    sns.boxenplot(x=df_filtered[feature], y=df_filtered['price'], ax=ax1)
    ax1.tick_params(axis='x', rotation=90)
    
    # 2. Box Plot
    ax2 = fig.add_subplot(grid[2:4, :])
    ax2.set_title('Box Plot')
    sns.boxplot(x=df_filtered[feature], y=df_filtered['price'], ax=ax2)
    ax2.tick_params(axis='x', rotation=90)
    
    # 3. Violin Plot
    ax3 = fig.add_subplot(grid[4:6, :])
    ax3.set_title('Violin Plot')
    sns.violinplot(x=df_filtered[feature], y=df_filtered['price'], data=df_filtered, ax=ax3)
    ax3.tick_params(axis='x', rotation=90)
    
    # 4. Line Plot (Mean and Median)
    ax4 = fig.add_subplot(grid[6:8, :])
    ax4.set_title('Line Plot of Aggregated Price by ' + feature)
    
    # Compute aggregated values
    agg_data = df_filtered.groupby(feature)['price'].agg(['mean', 'median']).reset_index()
    
    # Plot mean and median
    ax4.plot(agg_data[feature], agg_data['mean'], marker='o', label='Mean Price', color='#1f77b4')
    ax4.plot(agg_data[feature], agg_data['median'], marker='o', label='Median Price', color='#ff7f0e')
    
    ax4.legend()
    ax4.tick_params(axis='x', rotation=90)
    ax4.set_xlabel(feature)
    ax4.set_ylabel('Mean and Median Price')
    
    # 5. Heatmap of Median Price by feature
    ax5 = fig.add_subplot(grid[8:10, :])
    ax5.set_title('Heatmap of Median Price by ' + feature)
    
    heat_data = agg_data.set_index(feature)[['median']].T
    sns.heatmap(heat_data, annot=True, fmt=".2f", cmap="viridis", ax=ax5, cbar=True, linewidths=0.5, linecolor='white')
    ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45, ha='right')
    
    # 6. Jittered Strip Plot
    ax6 = fig.add_subplot(grid[10:12, :])
    ax6.set_title('Jittered Strip Plot of Price by ' + feature)
    sns.stripplot(data=df_filtered, x=feature, y='price', jitter=True, alpha=0.6, size=5, ax=ax6)
    ax6.set_xticklabels(ax6.get_xticklabels(), rotation=90)
    
    # 7. Count Plot
    ax7 = fig.add_subplot(grid[12:14, :])
    ax7.set_title('Count Plot')
    sns.countplot(x=df_filtered[feature], ax=ax7, order=top_categories)
    ax7.tick_params(axis='x', rotation=90)
    
    plt.show()

def region_plots_high_card_categorical(df, feature, top_n=10):
    """
    Plots various relationships between the given feature and other attributes, ensuring consistent colors for 'addressregion'.
    """

    # Get the top N categories based on count
    top_categories = df[feature].value_counts().nlargest(top_n).index.tolist()
    df_filtered = df[df[feature].isin(top_categories) & df['addressregion'].notna()]

    # Sort categories based on count
    df_filtered[feature] = pd.Categorical(df_filtered[feature], categories=top_categories, ordered=True)
    df_filtered = df_filtered.sort_values(feature)

    # Define a fixed color palette for 'addressregion'
    unique_regions = df_filtered['addressregion'].unique()
    palette = sns.color_palette("tab10", len(unique_regions))  
    region_palette = dict(zip(unique_regions, palette)) 

    # Create a figure with a grid
    fig = plt.figure(constrained_layout=True, figsize=(30, 30))
    grid = gridspec.GridSpec(ncols=3, nrows=5, figure=fig)

    # 7. Count plot
    ax7 = fig.add_subplot(grid[2, 0])
    ax7.set_title(f"7. Count Plot of '{feature}' by 'addressregion'")
    sns.countplot(x=feature, hue='addressregion', data=df_filtered, ax=ax7, palette=region_palette)

    ax1 = fig.add_subplot(grid[0, 0])
    ax1.set_title(f"1. Price vs '{feature}' by 'addressregion'")
    sns.barplot(x=feature, y='price', hue='addressregion', data=df_filtered, ax=ax1, palette=region_palette)

    ax2 = fig.add_subplot(grid[0, 1])
    ax2.set_title(f"2. Area vs '{feature}' by 'addressregion'")
    sns.barplot(x=feature, y='area', hue='addressregion', data=df_filtered, ax=ax2, palette=region_palette)

    ax3 = fig.add_subplot(grid[0, 2])
    ax3.set_title(f"3. Cost per Sqft vs '{feature}' by 'addressregion'")
    sns.barplot(x=feature, y='costpersqft', hue='addressregion', data=df_filtered, ax=ax3, palette=region_palette)

    # KDE plot
    ax8 = fig.add_subplot(grid[2, 1])
    ax8.set_title(f"8. Price Distribution by '{feature}' (KDE Plot)")
    sns.kdeplot(data=df_filtered, x="price", hue=feature, fill=True, common_norm=False, palette="tab10", linewidth=1.5, ax=ax8, warn_singular=False)

    # Line plots
    ax4 = fig.add_subplot(grid[1, 0])
    ax4.set_title(f"4. Price vs {feature}")
    sns.lineplot(data=df_filtered, x=feature, y="price", hue="addressregion", ax=ax4, palette=region_palette)

    ax5 = fig.add_subplot(grid[1, 1])
    ax5.set_title(f"5. Area vs {feature}")
    sns.lineplot(data=df_filtered, x=feature, y="area", hue="addressregion", ax=ax5, palette=region_palette)

    ax6 = fig.add_subplot(grid[1, 2])
    ax6.set_title(f"6. Cost per Sqft vs {feature}")
    sns.lineplot(data=df_filtered, x=feature, y="costpersqft", hue="addressregion", ax=ax6, palette=region_palette)

    # Improve readability
    for ax in [ax1, ax2, ax3, ax4, ax5 ,ax6, ax7, ax8]:  
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.show()

def num_two_chart_plot_high_card_categorical(df, feature, top_n=10):
    """
    The plots include:
    - Distribution Plot
    - Scatter Plot
    """
    # Get the top N categories based on count
    top_categories = df[feature].value_counts().nlargest(top_n).index.tolist()
    df_filtered = df[df[feature].isin(top_categories) & df['addressregion'].notna()]

    # Sort categories based on count
    df_filtered[feature] = pd.Categorical(df_filtered[feature], categories=top_categories, ordered=True)
    df_filtered = df_filtered.sort_values(feature)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Distribution Plot
    sns.histplot(df_filtered[feature], kde=True, ax=axes[0])
    axes[0].set_title('Distribution Plot', fontsize=12)
    axes[0].set_xlabel(feature, fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].tick_params(axis='both', labelsize=10)
    axes[0].tick_params(axis='x', rotation=90)  # Apply rotation to x-axis

    # Scatter Plot
    sns.scatterplot(x=df_filtered[feature], y=df_filtered['price'], ax=axes[1])
    axes[1].set_title('Scatter Plot', fontsize=12)
    axes[1].set_xlabel(feature, fontsize=12)
    axes[1].set_ylabel('Price', fontsize=12)
    axes[1].tick_params(axis='both', labelsize=10)
    axes[1].tick_params(axis='x', rotation=90)  # Apply rotation to x-axis

    # Tight layout to reduce overlapping
    plt.tight_layout()
    plt.show()

def summarize_properties_high_card_categorical(df, feature, top_n=10):
    """
    Returns a summary DataFrame containing overall metrics and region-specific counts 
    and median prices for properties, grouped by the specified feature.
    """

    # Get the top N categories based on count
    top_categories = df[feature].value_counts().nlargest(top_n).index.tolist()
    df_filtered = df[df[feature].isin(top_categories) & df['addressregion'].notna()]

    # Sort categories based on count
    df_filtered[feature] = pd.Categorical(df_filtered[feature], categories=top_categories, ordered=True)
    df_filtered = df_filtered.sort_values(feature)
    
    # Define regions of interest.
    regions = ['mumbai', 'navi mumbai', 'thane', 'palghar']
    
    # Group by the feature column to compute overall metrics.
    overall = df_filtered.groupby(feature).agg(
        price_mean=('price', 'mean'),
        price_median=('price', 'median'),
        count=('price', 'count')
    ).reset_index().rename(columns={feature: 'feature_value'})
    
    # Prepare lists to collect region-specific metrics for each unique feature value.
    region_counts = {region: [] for region in regions}
    region_median_prices = {region: [] for region in regions}
    
    unique_values = overall['feature_value'].unique()
    for val in unique_values:
        df_val = df_filtered[df_filtered[feature] == val]
        for region in regions:
            # Use a case-insensitive match on 'addressregion'
            region_df = df_val[df_val['addressregion'].str.lower() == region]
            region_counts[region].append(len(region_df))
            median_price = region_df['price'].median() if len(region_df) > 0 else None
            region_median_prices[region].append(median_price)
    
    # Add region-specific columns to the overall DataFrame.
    for region in regions:
        overall[region] = region_counts[region]
        overall[f"{region}_median_price"] = region_median_prices[region]
    
    return overall

def plot_log_median_subplots_high_card_categorical(df, feature, top_n=10):
    # Get the top N categories based on count
    top_categories = df[feature].value_counts().nlargest(top_n).index.tolist()
    df_filtered = df[df[feature].isin(top_categories) & df['addressregion'].notna()]

    # Sort categories based on count
    df_filtered[feature] = pd.Categorical(df_filtered[feature], categories=top_categories, ordered=True)
    df_filtered = df_filtered.sort_values(feature)
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharex=True, sharey=True)
    
    regions = ['mumbai', 'navi mumbai', 'palghar', 'thane']
    labels = ['Log Median costpersqft', 'Log Median area', 'Log Median price']
    columns = ['costpersqft', 'area', 'price']
    
    for i, region in enumerate(regions):
        ax = axes[i]
        subset = df_filtered[df_filtered['addressregion'] == region]
        
        # Compute median values
        median_values = subset.groupby(feature)[columns].median()
        
        # Ensure all values are positive before applying log
        median_values = median_values.replace(0, np.nan).dropna()
        
        log_median_values = np.log1p(median_values.dropna())

        for j, col in enumerate(columns):
            if log_median_values[col].isna().all():
                continue  # Skip if all values are NaN
            
            sns.lineplot(
                x=log_median_values.index,
                y=log_median_values[col],
                ax=ax,
                label=f"{labels[j]} for {region}",
                marker='o',
                linewidth=2
            )
            
            # Annotate each point
            for x_val, y_val in zip(log_median_values.index, log_median_values[col]):
                if np.isfinite(y_val):  # Ensure finite values before annotation
                    ax.text(x_val, y_val, f'{y_val:.2f}', ha='right', va='bottom', fontsize=12)
        
        ax.set_title(region.title(), fontsize=14, fontweight='bold')
        ax.set_xlabel(feature, fontsize=12)
        ax.set_ylabel("Log Median Values", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(fontsize=12, frameon=True)

        # Rotate x-axis labels properly
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

def median_plot_high_card_categorical(df, feature, top_n=10):
    # Get the top N categories based on count
    top_categories = df[feature].value_counts().nlargest(top_n).index.tolist()
    df_filtered = df[df[feature].isin(top_categories) & df['addressregion'].notna()]

    # Sort categories based on count
    df_filtered[feature] = pd.Categorical(df_filtered[feature], categories=top_categories, ordered=True)
    df_filtered = df_filtered.sort_values(feature)
    
    # Drop rows with missing values for relevant columns
    df_filtered = df_filtered.dropna(subset=[feature, 'costpersqft', 'area', 'price'])
    
    # Compute median values for each category in the feature
    median_cost = df_filtered.groupby(feature)['costpersqft'].median()
    median_area = df_filtered.groupby(feature)['area'].median()
    median_price = df_filtered.groupby(feature)['price'].median()
    
    # Apply log transformation
    log_median_cost = np.log1p(median_cost)
    log_median_area = np.log1p(median_area)
    log_median_price = np.log1p(median_price)
    
    # Combine data for visualization
    median_df = (
        pd.DataFrame({
            feature: log_median_cost.index, 
            'Log Median Cost per Sqft': log_median_cost.values, 
            'Log Median Area': log_median_area.values, 
            'Log Median Price': log_median_price.values
        })
        .melt(id_vars=[feature], var_name='Metric', value_name='Log Value')
    )
    
    # Plot the graph
    plt.figure(figsize=(10, 6))
    ax = sns.lineplot(data=median_df, x=feature, y='Log Value', hue='Metric', marker='o')
    
    # Annotate each point
    for line in ax.lines:
        for x, y in zip(line.get_xdata(), line.get_ydata()):
            ax.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Set labels and title
    plt.xlabel(feature, fontsize=12)
    plt.ylabel("Log Median Values", fontsize=12)
    plt.title("Log Median Cost per Sqft, Area, and Price by " + feature + " (Line Plot)", fontsize=14)
    plt.xticks(rotation=45)
    
    # Move legend outside the plot
    plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, frameon=True)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    
    # Create a sorted DataFrame (from expensive to cheapest)
    sorted_df = pd.DataFrame({
        feature: log_median_cost.index, 
        'log_median_costpersqft_values': log_median_cost.values  # Renamed column
    }).sort_values(by='log_median_costpersqft_values', ascending=False)
    
    # Print the sorted table
    print(sorted_df.to_string(index=False))
