import polars as pl
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az

from typing import Optional

#############################################################################################################

def plot_overlap_distribution(overlap_df):
    """
    Plot the distribution of weighted_overlap values for Man and Zone coverage.
    For use with the original overlap_df dataframe.

    Parameters:
    overlap_df (polars.DataFrame): DataFrame containing 'weighted_overlap' and 'pff_manZone' columns
    """
    # Convert to pandas for easier plotting with seaborn
    man_data = overlap_df.filter(pl.col("pff_manZone") == "Man").select("weighted_overlap").to_pandas()
    zone_data = overlap_df.filter(pl.col("pff_manZone") == "Zone").select("weighted_overlap").to_pandas()

    # Set the style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot both distributions using kernel density estimation
    sns.kdeplot(data=man_data["weighted_overlap"], label="Man Coverage", fill=True, alpha=0.3)
    sns.kdeplot(data=zone_data["weighted_overlap"], label="Zone Coverage", fill=True, alpha=0.3)

    # Customize the plot
    plt.title("Distribution of Weighted Overlap by Coverage Type", pad=20)
    plt.xlabel("Weighted Overlap")
    plt.ylabel("Density")
    plt.legend()

    # Add grid for better readability
    plt.grid(True, alpha=0.3)

    # Show the plot
    plt.show()

#############################################################################################################

def plot_changes_distribution(overlap_changes, column_name="absolute_volatility", handle_lists=False):
    """
    Plot the distribution of change/volatility metrics for Man and Zone coverage.
    For use with the overlap_changes dataframe.

    Parameters:
    overlap_changes (polars.DataFrame): DataFrame containing coverage change metrics
    column_name (str): Name of the column to plot
    handle_lists (bool): Whether to handle list-type columns with potential null values
    """
    # Handle list-type columns if specified
    if handle_lists:
        # Explode the list column and handle nulls
        man_data = (overlap_changes
                   .filter(pl.col("pff_manZone") == "Man")
                   .select(pl.col(column_name))
                   .explode(column_name)
                   .drop_nulls()
                   .to_pandas())

        zone_data = (overlap_changes
                    .filter(pl.col("pff_manZone") == "Zone")
                    .select(pl.col(column_name))
                    .explode(column_name)
                    .drop_nulls()
                    .to_pandas())
    else:
        # Regular columns
        man_data = (overlap_changes
                   .filter(pl.col("pff_manZone") == "Man")
                   .select(pl.col(column_name))
                   .drop_nulls()
                   .to_pandas())

        zone_data = (overlap_changes
                    .filter(pl.col("pff_manZone") == "Zone")
                    .select(pl.col(column_name))
                    .drop_nulls()
                    .to_pandas())

    # Set the style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot both distributions using kernel density estimation
    sns.kdeplot(data=man_data[column_name], label="Man Coverage", fill=True, alpha=0.3)
    sns.kdeplot(data=zone_data[column_name], label="Zone Coverage", fill=True, alpha=0.3)

    # Customize the plot
    title_map = {
        "overlap_delta": "Change in Overlap",
        "absolute_overlap_delta": "Absolute Change in Overlap",
        "directional_volatility": "Directional Volatility",
        "absolute_volatility": "Absolute Volatility"
    }

    plot_title = title_map.get(column_name, column_name.replace('_', ' ').title())
    plt.title(f"Distribution of {plot_title} by Coverage Type", pad=20)
    plt.xlabel(plot_title)
    plt.ylabel("Density")
    plt.legend()

    # Add grid for better readability
    plt.grid(True, alpha=0.3)

    # Show the plot
    plt.show()

#############################################################################################################

def plot_volatility_distributions(overlap_changes):
    """
    Plot regular and log-transformed volatility distributions side by side.
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Get data
    man_data = overlap_changes.filter(pl.col("pff_manZone") == "Man")["absolute_volatility"].to_numpy()
    zone_data = overlap_changes.filter(pl.col("pff_manZone") == "Zone")["absolute_volatility"].to_numpy()

    # Plot regular distribution
    sns.kdeplot(data=man_data, label="Man Coverage", ax=ax1, fill=True, alpha=0.3)
    sns.kdeplot(data=zone_data, label="Zone Coverage", ax=ax1, fill=True, alpha=0.3)
    ax1.set_title("Distribution of Absolute Volatility")
    ax1.set_xlabel("Absolute Volatility")
    ax1.set_ylabel("Density")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot log-transformed distribution
    sns.kdeplot(data=np.log(man_data), label="Man Coverage", ax=ax2, fill=True, alpha=0.3)
    sns.kdeplot(data=np.log(zone_data), label="Zone Coverage", ax=ax2, fill=True, alpha=0.3)
    ax2.set_title("Distribution of Log Absolute Volatility")
    ax2.set_xlabel("Log Absolute Volatility")
    ax2.set_ylabel("Density")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

#############################################################################################################

def filter_and_sort_overlaps(df: pl.DataFrame,
                           coverage_type: str,
                           sort_col: str,
                           ascending: bool = True,
                           n_rows: int = 3) -> pl.DataFrame:
    """
    Filter overlap data by coverage type and sort by specified column.

    Args:
        df: Input DataFrame
        coverage_type: Either "Man" or "Zone"
        sort_col: Column name to sort by
        ascending: Sort order
        n_rows: Number of rows to return

    Returns:
        Filtered and sorted DataFrame
    """
    filtered = df.filter(pl.col('pff_manZone') == coverage_type)

    if ascending:
        return filtered.sort(sort_col).head(n_rows)
    else:
        return filtered.sort(sort_col, descending=True).head(n_rows)

#############################################################################################################

def calculate_total_delta(df: pl.DataFrame,
                         coverage_type: Optional[str] = None,
                         ascending: bool = True,
                         n_rows: int = 3) -> pl.DataFrame:
    """
    Calculate total delta for overlap values per gameId and playId, optionally filtering by coverage type.

    Args:
        df: Input DataFrame
        coverage_type: Optional filter for "Man" or "Zone"
        ascending: Sort order for total_delta

    Returns:
        DataFrame with total_delta column calculated per gameId and playId
    """
    result = df

    if coverage_type:
        result = result.filter(pl.col('pff_manZone') == coverage_type)

    result = result.with_columns([
        pl.col('overlap_delta')
        .explode()
        .drop_nulls()
        .sum()
        .over(['gameId', 'playId'])
        .alias('total_delta')
    ])

    return result.sort('total_delta', descending=not ascending).head(n_rows)