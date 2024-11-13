import polars as pl
import random

#############################################################################################################

def find_overlapping_plays_efficient(df1, df2):
    """
    Memory-efficient way to find overlapping plays between two dataframes.
    Only processes unique gameId-playId combinations and shows random examples.

    Parameters:
    df1, df2: Polars DataFrames with gameId and playId columns

    Returns:
    tuple: (overlap_count, overlap_percentage)
    """
    # Get unique combinations from each dataframe
    unique_plays1 = (df1
        .select(['gameId', 'playId'])
        .unique()
        .to_numpy()
    )

    unique_plays2 = (df2
        .select(['gameId', 'playId'])
        .unique()
        .to_numpy()
    )

    # Convert to sets of tuples for efficient comparison
    plays_set1 = {tuple(row) for row in unique_plays1}
    plays_set2 = {tuple(row) for row in unique_plays2}

    # Find overlaps using set intersection
    overlaps = plays_set1.intersection(plays_set2)

    # Calculate statistics
    total_overlaps = len(overlaps)
    total_unique1 = len(plays_set1)
    total_unique2 = len(plays_set2)
    overlap_percentage = (total_overlaps / min(total_unique1, total_unique2)) * 100

    print(f"\nOverlap Statistics:")
    print(f"Total unique plays in first DataFrame: {total_unique1}")
    print(f"Total unique plays in second DataFrame: {total_unique2}")
    print(f"Number of overlapping plays: {total_overlaps}")
    print(f"Overlap percentage: {overlap_percentage:.2f}%")

    # Print random example overlaps if any exist
    if overlaps:
        print("\nRandom sample of overlapping plays (up to 5):")
        # Convert overlaps to list for random sampling
        overlaps_list = list(overlaps)
        # Get min of 5 or total overlaps
        sample_size = min(5, len(overlaps_list))
        # Get random sample
        random_samples = random.sample(overlaps_list, sample_size)
        for play in random_samples:
            print(f"GameId: {play[0]}, PlayId: {play[1]}")

    return total_overlaps, overlap_percentage

#############################################################################################################

def sample_man_zone_plays(df, n_samples=25):
    """
    Randomly sample n_samples unique plays for both Man and Zone coverage.
    
    Parameters:
    df: Polars DataFrame with columns [gameId, playId, pff_manZone]
    n_samples: Number of plays to sample for each coverage type (default: 25)
    
    Returns:
    Dictionary containing two lists of (gameId, playId) tuples for Man and Zone plays
    """
    # Get unique combinations of gameId and playId for each coverage type
    unique_plays = (
        df.select(['gameId', 'playId', 'pff_manZone'])
        .unique(subset=['gameId', 'playId', 'pff_manZone'])
    )
    
    # Separate Man and Zone plays
    man_plays = (
        unique_plays
        .filter(pl.col('pff_manZone') == 'Man')
        .select(['gameId', 'playId'])
        .sample(n=n_samples, shuffle=True)
        .to_pandas()
        .values.tolist()
    )
    
    zone_plays = (
        unique_plays
        .filter(pl.col('pff_manZone') == 'Zone')
        .select(['gameId', 'playId'])
        .sample(n=n_samples, shuffle=True)
        .to_pandas()
        .values.tolist()
    )
    
    return {
        'man_plays': man_plays,
        'zone_plays': zone_plays
    }

#############################################################################################################

def filter_sampled_plays(existing_manzone, sampled_plays):
    """
    Remove plays from sampled_plays that exist in existing_manzone DataFrame.
    This is used in the Model Predictions file to ensure no duplicate predictions
    are made.
    
    Parameters:
    existing_manzone (pd.DataFrame): DataFrame with gameId and playId columns
    sampled_plays (dict): Dictionary of lists containing [gameId, playId] pairs
    
    Returns:
    dict: Filtered sampled_plays dictionary
    """
    # Convert existing data to set of tuples for O(1) lookup
    existing_pairs = set(zip(existing_manzone['gameId'], existing_manzone['playId']))
    
    # Create new dictionary with filtered lists
    filtered_plays = {}
    
    for key, plays in sampled_plays.items():
        # Keep only plays that don't exist in existing_manzone
        filtered_list = [
            play for play in plays 
            if tuple(play) not in existing_pairs
        ]
        filtered_plays[key] = filtered_list
    
    return filtered_plays
