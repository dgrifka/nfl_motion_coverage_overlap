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