# Core data manipulation
import polars as pl
from pathlib import Path
from typing import Optional, Union, List, Literal

# Math/computing
import numpy as np

google_drive_path = '/content/drive/My Drive/NFL'

#############################################################################################################

def read_nfl_csv(csv_name):
  # Import CSV files using Polars, since this is faster
  # 'NA' values will be treated as null/missing data
  df = pl.read_csv(f'{google_drive_path}/{csv_name}.csv', null_values=['NA'])
  return df

#############################################################################################################

def process_nfl_plays(plays: pl.DataFrame, games: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    """
    Process NFL play-by-play data by adding team information, score differentials, and time-related features.

    Args:
        plays (pl.DataFrame): The plays DataFrame containing play-by-play data
        games (pl.DataFrame, optional): The games DataFrame containing game information.
                                      If None, score differential and team info won't be added.

    Returns:
        pl.DataFrame: DataFrame with additional columns:
            - homeTeamAbbr: Home team abbreviation
            - visitorTeamAbbr: Visitor team abbreviation
            - offense_score_diff: Score differential from offense's perspective
            - game_time_left_mins: Total minutes remaining in the game
            - quarter_mins_left: Minutes remaining in current quarter

    Raises:
        ValueError: If required columns are missing from input DataFrames
    """
    # Verify required columns in plays DataFrame
    required_plays_cols = {'gameId', 'quarter', 'gameClock'}
    if not all(col in plays.columns for col in required_plays_cols):
        missing_cols = required_plays_cols - set(plays.columns)
        raise ValueError(f"Plays DataFrame missing required columns: {missing_cols}")

    # Create a copy of plays to avoid modifying the input
    processed_plays = plays.clone()

    def convert_gameclock_to_minutes(clock_str: str) -> float:
        """Convert game clock from MM:SS format to minutes as a decimal number"""
        try:
            minutes, seconds = map(int, clock_str.split(':'))
            return minutes + seconds / 60
        except (ValueError, TypeError):
            return 0.0  # Return 0 for invalid formats

    # Add team information and score differential if games DataFrame is provided
    if games is not None:
        # Verify required columns in games DataFrame
        required_games_cols = {'gameId', 'homeTeamAbbr', 'visitorTeamAbbr'}
        if not all(col in games.columns for col in required_games_cols):
            missing_cols = required_games_cols - set(games.columns)
            raise ValueError(f"Games DataFrame missing required columns: {missing_cols}")

        # Additional required columns for score differential
        score_cols = {'possessionTeam', 'preSnapHomeScore', 'preSnapVisitorScore'}
        has_score_cols = all(col in plays.columns for col in score_cols)

        # Join with games DataFrame
        processed_plays = (
            processed_plays
            .join(
                games.select(['gameId', 'homeTeamAbbr', 'visitorTeamAbbr']),
                on=["gameId"],
                how="left"
            )
        )

        # Add score differential if all required columns are present
        if has_score_cols:
            processed_plays = processed_plays.with_columns([
                pl.when(pl.col('possessionTeam') == pl.col('homeTeamAbbr'))
                .then(pl.col('preSnapHomeScore') - pl.col('preSnapVisitorScore'))
                .otherwise(pl.col('preSnapVisitorScore') - pl.col('preSnapHomeScore'))
                .alias('offense_score_diff')
            ])

    # Add time-related columns
    processed_plays = processed_plays.with_columns([
        # Total game time remaining
        (15 * 4 -
         ((pl.col('quarter') - 1) * 15 +
          (15 - pl.col('gameClock').map_elements(convert_gameclock_to_minutes, return_dtype=pl.Float64)))
        ).alias('game_time_left_mins'),

        # Time remaining in current quarter
        pl.col('gameClock')
        .map_elements(convert_gameclock_to_minutes, return_dtype=pl.Float64)
        .alias('quarter_mins_left')
    ])

    return processed_plays

#############################################################################################################

def process_tracking_data(
    tracking_path: Union[str, Path],
    snap_type: Literal["pre", "post"],
    weeks: List[int] = list(range(1, 10)),
    file_pattern: str = "tracking_week_{}.csv"
) -> pl.DataFrame:
    """
    Process NFL tracking data by combining multiple weeks and filtering for either pre-snap or post-snap frames.
    For post-snap, identifies pass plays. For pre-snap, identifies plays with motion.

    Args:
        tracking_path (str or Path): Base path where tracking CSV files are stored
        snap_type (str): Either "pre" or "post" to determine filtering logic
        weeks (List[int]): List of week numbers to process (default: weeks 1-9)
        file_pattern (str): Pattern for tracking file names (default: "tracking_week_{}.csv")

    Returns:
        pl.DataFrame: Processed tracking data filtered according to snap_type,
                     sorted by gameId, playId, nflId, and frameId

    Raises:
        FileNotFoundError: If any of the tracking files are not found
        ValueError: If the tracking data is missing required columns or invalid snap_type
    """
    if snap_type not in ["pre", "post"]:
        raise ValueError("snap_type must be either 'pre' or 'post'")

    # Convert path to Path object for better path handling
    base_path = Path(tracking_path)

    # Initialize empty DataFrame
    filtered_data = None

    # Required columns for validation
    required_cols = {'gameId', 'playId', 'nflId', 'frameId', 'frameType', 'event'}

    # Process each week's data
    for week in weeks:
        file_path = base_path / file_pattern.format(week)

        # Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(f"Tracking file not found for week {week}: {file_path}")

        # Read week's data
        week_df = pl.read_csv(file_path, null_values=['NA'])

        # Validate columns
        if not all(col in week_df.columns for col in required_cols):
            missing_cols = required_cols - set(week_df.columns)
            raise ValueError(f"Week {week} data missing required columns: {missing_cols}")

        if snap_type == "post":
            # Post-snap processing
            week_df = (
                week_df
                .filter(pl.col("frameType") == "AFTER_SNAP")
                .sort(["gameId", "playId", "nflId", "frameId"])
            )
        else:
            # Pre-snap processing
            # Find minimum frameId for line_set events
            line_set_frames = (
                week_df
                .filter(pl.col("event") == "line_set")
                .select(["gameId", "playId", "nflId", "frameId"])
                .group_by(["gameId", "playId", "nflId"])
                .agg(pl.col("frameId").min().alias("line_set_frameId"))
            )

            # Find plays with motion
            motion_plays = (
                week_df
                .filter(pl.col("event") == "man_in_motion")
                .select(["gameId", "playId"])
                .unique()
            )

            # Filter for pre-snap frames after line_set
            week_df = (
                week_df
                .join(
                    line_set_frames,
                    on=["gameId", "playId", "nflId"],
                    how="inner"
                )
                .join(
                    motion_plays,
                    on=["gameId", "playId"],
                    how="inner"
                )
                .filter(pl.col("frameId") >= pl.col("line_set_frameId"))
                .filter((pl.col("frameType") == "BEFORE_SNAP") | (pl.col("frameType") == "SNAP"))
                .sort(["gameId", "playId", "nflId", "frameId"])
                .drop("line_set_frameId")
            )

        # Concatenate with existing results
        if filtered_data is None:
            filtered_data = week_df
        else:
            filtered_data = pl.concat([filtered_data, week_df])

    # Final sorting
    return filtered_data.sort(["gameId", "playId", "nflId", "frameId"])

#############################################################################################################

def calculate_mean_ttt(post_snap: pl.DataFrame, frame_rate: float = 0.1) -> float:
    """
    Calculates mean time to throw from snap to pass.

    Parameters:
    -----------
    post_snap : pl.DataFrame
        Data with gameId, playId, frameId, and event columns
    frame_rate : float, default=0.1
        Time between frames (NFL = 10 fps = 0.1s)

    Returns:
    --------
    float
        Mean time to throw in seconds
    """

    # Calculate time to throw using frameId
    time_to_throw = (
        post_snap
        .group_by(["gameId", "playId"])
        .agg([
            # Get first frameId for play
            pl.col("frameId").first().alias("start_frame"),
            # Get frameId at pass_forward
            pl.col("frameId")
            .filter(pl.col("event").str.contains("pass_forward"))
            .first()
            .alias("pass_frame")
        ])
        # Calculate time difference
        .with_columns([
            ((pl.col("pass_frame") - pl.col("start_frame")) * frame_rate)
            .alias("time_to_throw_seconds")
        ])
        # Remove null values
        .filter(
            pl.col("time_to_throw_seconds").is_not_null()
        )
    )

    # Return mean time to throw
    return time_to_throw.select(pl.col("time_to_throw_seconds").mean()).item()

#############################################################################################################

def add_team_column(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add a 'team' column to the DataFrame based on position values ('offense' or 'defense').
    Returns a new DataFrame with the additional column.

    Args:
        df: Polars DataFrame containing a 'position' column

    Returns:
        Polars DataFrame with new 'team' column
    """
    defensive_positions = ['CB', 'NT', 'DE', 'FS', 'SS', 'LB', 'DB', 'DT', 'MLB', 'OLB', 'ILB']
    offensive_positions = ['C', 'T', 'FB', 'G', 'WR', 'TE', 'QB', 'RB']

    return df.with_columns(
        pl.when(pl.col('position').is_in(defensive_positions))
        .then(pl.lit('defense'))
        .when(pl.col('position').is_in(offensive_positions))
        .then(pl.lit('offense'))
        .otherwise(pl.lit('unknown'))
        .alias('team')
    )

#############################################################################################################

def normalize_direction(df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalizes play direction by standardizing coordinates and angles.
    
    Parameters:
    -----------
    df : pl.DataFrame
        Data with x, y, o (orientation), dir (direction), and playDirection
    
    Returns:
    --------
    pl.DataFrame
        Original data plus normalized columns:
        - x_normalized: Flips x-coord if play goes left
        - y_normalized: Y-coord (unchanged)
        - o_normalized: Mirrors orientation if play goes left
        - dir_normalized: Mirrors direction if play goes left
    """
    
    return df.with_columns([
        # Flip x-coordinate for left-moving plays (field is 120 yards)
        pl.when(pl.col("playDirection") == "left")
        .then(120 - pl.col("x"))
        .otherwise(pl.col("x"))
        .alias("x_normalized"),

        # Y-coordinate remains unchanged
        pl.col("y").alias("y_normalized"),

        # Mirror orientation across y-axis for left-moving plays
        pl.when(pl.col("playDirection") == "left")
        .then((360 - pl.col("o")) % 360)
        .otherwise(pl.col("o"))
        .alias("o_normalized"),

        # Mirror direction across y-axis for left-moving plays
        pl.when(pl.col("playDirection") == "left")
        .then((360 - pl.col("dir")) % 360)
        .otherwise(pl.col("dir"))
        .alias("dir_normalized")
    ])

#############################################################################################################

def merge_tracking_data(
    tracking_data: pl.DataFrame,
    plays: pl.DataFrame,
    players: pl.DataFrame,
    add_team_column_func: callable
) -> pl.DataFrame:
    """
    Merges player tracking data with play and player information, and adds team designation.

    This function performs a series of left joins to combine tracking data with play-level
    and player-level information, then adds offensive/defensive team designation. Finally,
    it filters for only dropbacks.

    Parameters:
    -----------
    tracking_data : pl.DataFrame
        Either pre-snap or post-snap player tracking data
    plays : pl.DataFrame
        Play-by-play data containing game and play level information
    players : pl.DataFrame
        Player information including position data
    add_team_column_func : callable
        Function that adds offense/defense designation to the DataFrame

    Returns:
    --------
    pl.DataFrame
        Merged DataFrame containing tracking, play, and player information with team designation
        that is filtered for dropbacks only
    """

    # Select relevant columns from plays DataFrame
    play_columns = [
        'gameId',
        'playId',
        'quarter',
        'quarter_mins_left',
        'offense_score_diff',
        'game_time_left_mins',
        'down',
        'yardsToGo',
        'absoluteYardlineNumber',
        'offenseFormation',
        'isDropback',
        'pff_passCoverage',
        'pff_manZone'
    ]

    # Merge tracking data with plays and players information
    merged_data = (
        tracking_data
        # Merge with plays data
        .join(
            plays.select(play_columns),
            on=['gameId', 'playId'],
            how='left'
        )
        # Merge with players data
        .join(
            players.select(['nflId', 'position']),
            on=['nflId'],
            how='left'
        )
    )

    # Add offense/defense designation
    merged_data = add_team_column_func(merged_data)

    # Filter only for dropbacks
    merged_data = merged_data.filter(pl.col("isDropback") == True)

    return merged_data

#############################################################################################################

def calculate_shortened_passing_plays(
    tracking_data: pl.DataFrame,
    mean_ttt: float,
    frame_rate: float = 0.1
) -> pl.DataFrame:
    """
    Calculates time elapsed for passing plays and filters to mean time to throw.

    Parameters:
    -----------
    tracking_data : pl.DataFrame
        Tracking data to process (requires gameId, playId, frameId)
    mean_ttt : float
        Mean time to throw (in seconds) to use as cutoff
    frame_rate : float, default=0.1
        Time between frames (NFL = 10 fps = 0.1s)

    Returns:
    --------
    pl.DataFrame
        Tracking data filtered to mean time to throw
    """

    return (
        tracking_data
        # Get starting frame for each play
        .with_columns(
            pl.col("frameId")
            .first()
            .over(["gameId", "playId"])
            .alias("start_frame")
        )
        # Calculate time since play start
        .with_columns(
            ((pl.col("frameId") - pl.col("start_frame")) * frame_rate)
            .alias("time_elapsed")
        )
        # Remove null times
        .filter(
            pl.col("time_elapsed").is_not_null()
        )
        # Keep only frames before mean time to throw
        .filter(
            pl.col("time_elapsed") <= mean_ttt
        )
        # Sort for consistency
        .sort(["gameId", "playId", "frameId"])
    )

#############################################################################################################

def get_initial_ball_position(tracking_data: pl.DataFrame) -> pl.DataFrame:
    """
    Gets initial ball position for each play from tracking data.
    This is used for various calculations and visuals.
    
    Parameters:
    -----------
    tracking_data : pl.DataFrame
        Tracking data with normalized coordinates and displayName
    
    Returns:
    --------
    pl.DataFrame
        Initial ball position for each play:
        - gameId, playId
        - x_normalized, y_normalized
    """
    
    return (
        tracking_data
        # Sort to ensure correct order
        .sort(["gameId", "playId", "nflId", "frameId"])
        # Filter to only football tracking
        .filter(pl.col("displayName") == "football")
        # Get initial position for each play
        .group_by(['gameId', 'playId'])
        .agg([
            pl.col('x_normalized').first().alias('x_normalized'),
            pl.col('y_normalized').first().alias('y_normalized')
        ])
    )

#############################################################################################################

def filter_players_by_position(
    tracking_data: pl.DataFrame,
    game_ids: pl.DataFrame,
    position_group: str = 'defense'
) -> pl.DataFrame:
    """
    Filters tracking data for defensive or offensive players.
    
    Parameters:
    -----------
    tracking_data : pl.DataFrame
        Player tracking data with position column
    game_ids : pl.DataFrame
        DataFrame with gameId column to filter by
    position_group : str, default='defense'
        'defense' for defensive players or 'offense' for receivers
    
    Returns:
    --------
    pl.DataFrame
        Filtered tracking data for specified position group
    """
    
    # Define position groups
    DEFENSIVE_POSITIONS = ['LB', 'MLB', 'ILB', 'OLB', 'DB', 'CB', 'SS', 'FS']
    OFFENSIVE_POSITIONS = ['WR', 'TE', 'RB']
    
    # Filter by game IDs
    filtered_df = tracking_data.filter(
        pl.col('gameId').is_in(game_ids['gameId'])
    )
    
    # Filter by position group
    if position_group.lower() == 'defense':
        return filtered_df.filter(
            pl.col('position').is_in(DEFENSIVE_POSITIONS)
        )
    elif position_group.lower() == 'offense':
        return filtered_df.filter(
            pl.col('position').is_in(OFFENSIVE_POSITIONS)
        )
    else:
        raise ValueError("position_group must be 'defense' or 'offense'")

#############################################################################################################

def add_distance_from_football(db_df, ball_df):
    """
    Add distance_from_football and relative_position features to the defensive players dataframe.

    Parameters:
    -----------
    db_df : polars.DataFrame
        DataFrame containing defensive players' positions and other features
    ball_df : polars.DataFrame
        DataFrame containing football positions for each play
        Must have columns: gameId, playId, x_normalized, y_normalized

    Returns:
    --------
    polars.DataFrame
        Original dataframe with new distance_from_football and relative_position columns
        relative_position will be (facing the ball):
        - positive: player is to the right of the football (y is smaller)
        - negative: player is to the left of the football (y is larger)
        - 0: player is exactly aligned with football on y-axis
    """
    # Get initial football positions for each play
    ball_positions = ball_df.group_by(['gameId', 'playId'])\
        .agg([
            pl.col('x_normalized').first().alias('football_x'),
            pl.col('y_normalized').first().alias('football_y')
        ])

    # Join ball positions with player positions
    db_with_ball = db_df.join(
        ball_positions,
        on=['gameId', 'playId'],
        how='left'
    )

    # Calculate Euclidean distance and relative position
    db_with_features = db_with_ball.with_columns([
        # Euclidean distance calculation
        ((pl.col('x_normalized') - pl.col('football_x'))**2 +
         (pl.col('y_normalized') - pl.col('football_y'))**2).pow(0.5).alias('distance_from_football'),

        # Relative position calculation (positive = right, negative = left)
        (pl.col('y_normalized') - pl.col('football_y')).alias('relative_y_position')
    ])

    return db_with_features
