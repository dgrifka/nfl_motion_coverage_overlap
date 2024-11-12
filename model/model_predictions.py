# Data manipulation and scientific computing
import numpy as np
import pandas as pd
import polars as pl

# Visualization
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Machine learning and preprocessing components
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Animation and image handling
import imageio
from IPython.display import Image, display
import tempfile
import os
#############################################################################################################

def get_actual_routes(receivers_df, game_id, play_id):
    """
    Extract the actual routes run by players in a specific play from tracking data.

    Parameters:
        receivers_df (polars.DataFrame): DataFrame containing player tracking data
        game_id (int): The ID of the game to analyze
        play_id (int): The ID of the specific play within the game

    Returns:
        tuple: (play_data, actual_routes) where:
            - play_data: Polars DataFrame containing all tracking data for the play
            - actual_routes: List of numpy arrays, each array containing (x,y) coordinates 
              representing a player's route during the play
    """
    # Filter the dataframe for the specific play
    play_data = receivers_df.filter(
        (pl.col('gameId') == game_id) &
        (pl.col('playId') == play_id)
    )

    # Convert play_data to pandas for easier groupby operations
    play_data_pd = play_data.to_pandas()

    # Extract actual routes - one route per player (nflId)
    actual_routes = []
    for _, group in play_data_pd.groupby('nflId'):
        # Create array of (x,y) coordinates for each player's route
        route = group[['x_normalized', 'y_normalized']].values
        actual_routes.append(route)

    return play_data, actual_routes

#############################################################################################################

def analyze_specific_play(receivers_df, model, scalers, scaler_y, game_id, play_id):
    """
    Analyze and visualize player routes for a specific play, handling data preparation,
    making predictions, and creating visualizations.

    Parameters:
        receivers_df (polars.DataFrame): DataFrame containing player tracking data
        model: Trained ML model for route prediction
        scalers: Tuple of (position_scaler, context_scaler) for feature normalization
        scaler_y: Scaler for denormalizing model predictions
        game_id (int): The ID of the game to analyze
        play_id (int): The ID of the specific play within the game

    Returns:
        tuple: (predictions, actual_routes) where:
            - predictions: Model's predicted routes for each player
            - actual_routes: List of actual routes run by players
    """
    # Get the actual route data for the play
    play_data, actual_routes = get_actual_routes(receivers_df, game_id, play_id)

    # Get initial position for each player (first frame of the play)
    initial_positions = play_data.group_by('nflId').first().to_pandas()

    # List of all required features for the model
    required_columns = [
        # Player position and orientation features
        'x_normalized', 'y_normalized', 'o_normalized', 'dir_normalized',
        # Game situation features
        'yardsToGo', 'absoluteYardlineNumber', 'offense_score_diff', 
        'down', 'game_time_left_mins', 'quarter', 'quarter_mins_left',
        # Formation features
        'formation_SHOTGUN', 'formation_UNDER_CENTER', 'formation_PISTOL',
        'formation_NO_HUDDLE_SHOTGUN', 'formation_NO_HUDDLE', 'formation_WILDCAT',
        # Player-specific features
        'position', 'distance_from_football', 'relative_y_position'
    ]

    # Fill missing columns with zeros
    for col in required_columns:
        if col not in initial_positions.columns:
            initial_positions[col] = 0

    # Prepare separate column sets for modeling and plotting
    modeling_columns = [col for col in required_columns if col != 'position']  # position used differently in model
    plot_columns = modeling_columns + ['position'] if 'position' in initial_positions.columns else modeling_columns

    # Create dataframes for model input and plotting
    initial_positions_for_model = initial_positions[modeling_columns]
    initial_positions_for_plot = initial_positions[plot_columns]

    # Generate predictions and create visualization
    predictions = predict_and_create_heatmap(model, initial_positions_for_model, scalers, scaler_y)
    plot_route_heatmap_with_actual(initial_positions_for_plot, predictions, actual_routes, game_id, play_id)
    
    return predictions, actual_routes

#############################################################################################################

def predict_and_create_heatmap(model, initial_positions, scalers, scaler_y, side='offense', n_predictions=200, prediction_steps=10):
    """
    Create route predictions with position-specific behaviors and game-situation awareness.
    
    Parameters:
        model: Trained ML model for route prediction
        initial_positions (pd.DataFrame): Initial player positions and game context
        scalers: Tuple of (position_scaler, context_scaler) for feature normalization
        scaler_y: Scaler for denormalizing predictions
        side (str): 'offense' or 'defense' - determines position groups and behaviors
        n_predictions (int): Number of route variations to generate
        prediction_steps (int): Number of steps to predict in each route
        
    Returns:
        np.array: Array of shape (n_predictions, n_players, prediction_steps, 2) containing predicted routes
    """
    position_scaler, context_scaler = scalers
    
    # 1. FEATURE EXTRACTION AND SCALING
    # Extract and scale position-related features
    position_features = initial_positions[['x_normalized', 'y_normalized', 'o_normalized',
                                         'yardsToGo', 'absoluteYardlineNumber', 'distance_from_football',
                                         'relative_y_position']].values
    position_features_scaled = position_scaler.transform(position_features)
    
    # 2. NOISE GENERATION FOR ROUTE VARIATION
    # Base noise with field position influence
    base_noise_std = 0.08
    field_position_factor = np.abs(position_features[:, 0] - 60) / 60  # Distance from midfield
    noise = np.random.normal(0, base_noise_std, (n_predictions, *position_features_scaled.shape))
    
    # Apply field position and temporal factors
    field_position_factor = field_position_factor.reshape(1, -1, 1)
    noise = noise * (1 + field_position_factor)
    temporal_factor = np.linspace(1, 2, position_features_scaled.shape[1])
    noise = noise * temporal_factor[np.newaxis, np.newaxis, :]
    
    # 3. FEATURE PREPARATION
    # Add noise to positions
    noisy_positions = position_features_scaled[np.newaxis, :, :] + noise
    noisy_positions = noisy_positions.reshape(-1, position_features_scaled.shape[1])
    
    # Prepare game context features
    context_features = np.zeros((position_features.shape[0], 5))
    context_columns = ['offense_score_diff', 'down', 'game_time_left_mins', 'quarter', 'quarter_mins_left']
    for i, col in enumerate(context_columns):
        if col in initial_positions.columns:
            context_features[:, i] = initial_positions[col].values
    context_features_scaled = context_scaler.transform(context_features)
    
    # 4. FORMATION AND POSITION ENCODING
    # Setup formation features
    formation_categories = ['SHOTGUN', 'UNDER_CENTER', 'PISTOL',
                          'NO_HUDDLE_SHOTGUN', 'NO_HUDDLE', 'WILDCAT']
    formation_features = np.zeros((position_features.shape[0], len(formation_categories)))
    for i, formation in enumerate(formation_categories):
        col_name = f'formation_{formation}'
        if col_name in initial_positions.columns:
            formation_features[:, i] = initial_positions[col_name].values
    
    # 5. POSITION-SPECIFIC ENCODING
    # Define position groups and encode positions
    position_groups = {
        'offense': ['WR', 'TE', 'RB'],
        'defense': ['LB', 'MLB', 'ILB', 'OLB', 'DB', 'CB', 'SS', 'FS']
    }
    positions = position_groups[side]
    position_encoding = np.zeros((position_features.shape[0], len(positions)))
    for i, pos in enumerate(positions):
        col_name = f'position_{pos}'
        if col_name in initial_positions.columns:
            position_encoding[:, i] = initial_positions[col_name].values
            
    # 6. BATCH PREPARATION
    # Repeat features for batch prediction
    context_features_repeated = np.repeat(context_features_scaled[np.newaxis, :, :], n_predictions, axis=0)
    formation_features_repeated = np.repeat(formation_features[np.newaxis, :, :], n_predictions, axis=0)
    position_encoding_repeated = np.repeat(position_encoding[np.newaxis, :, :], n_predictions, axis=0)
    
    # Reshape all features for batch prediction
    context_features_batch = context_features_repeated.reshape(-1, context_features_scaled.shape[1])
    formation_features_batch = formation_features_repeated.reshape(-1, formation_features.shape[1])
    position_encoding_batch = position_encoding_repeated.reshape(-1, position_encoding.shape[1])

            
    # 7. MODEL PREDICTION
    pred_scaled = model.predict(
        [noisy_positions.astype(np.float32),
         context_features_batch.astype(np.float32),
         formation_features_batch.astype(np.float32),
         position_encoding_batch.astype(np.float32)],
        batch_size=32,
        verbose=0
    )
    
    # 8. POST-PROCESSING
    # Transform predictions and reshape
    predictions = scaler_y.inverse_transform(pred_scaled)
    predictions = predictions.reshape(n_predictions, -1, prediction_steps, 2)
    
    # Ensure predictions stay within field boundaries
    predictions[:, :, :, 0] = np.clip(predictions[:, :, :, 0], 0, 120)  # x-coordinates
    predictions[:, :, :, 1] = np.clip(predictions[:, :, :, 1], 0, 53.3)  # y-coordinates
    
    return predictions

#############################################################################################################

def analyze_specific_play_by_side(receivers_df, model, scalers, scaler_y, game_id, play_id, side='defense', plot=True):
    """Modified to include plot parameter and avoid unnecessary plotting during animation"""
    position_groups = {
        'offense': ['WR', 'TE', 'RB'],
        'defense': ['LB', 'MLB', 'ILB', 'OLB', 'DB', 'CB', 'SS', 'FS']
    }

    play_data = receivers_df.filter(
        (pl.col('gameId') == game_id) &
        (pl.col('playId') == play_id) &
        pl.col('position').is_in(position_groups[side])
    )

    if play_data.height == 0:
        print(f"No {side} players found for game_id={game_id}, play_id={play_id}")
        available_positions = receivers_df.filter(
            (pl.col('gameId') == game_id) &
            (pl.col('playId') == play_id)
        ).select('position').unique()
        print("Available positions in this play:")
        print(available_positions.to_pandas())
        return None, None

    play_data = play_data.sort('frameId')
    football_pos = play_data.select(['football_x', 'football_y']).head(1).to_pandas()

    play_data_pd = play_data.to_pandas()
    actual_routes = []
    for _, group in play_data_pd.groupby('nflId'):
        route = group[['x_normalized', 'y_normalized']].values
        actual_routes.append(route)

    initial_positions = play_data.group_by('nflId').head(1).to_pandas()

    required_columns = [
        'x_normalized', 'y_normalized', 'o_normalized', 'dir_normalized',
        'yardsToGo', 'absoluteYardlineNumber',
        'offense_score_diff', 'down', 'game_time_left_mins', 'quarter', 'quarter_mins_left',
        'formation_SHOTGUN', 'formation_UNDER_CENTER', 'formation_PISTOL',
        'formation_NO_HUDDLE_SHOTGUN', 'formation_NO_HUDDLE', 'formation_WILDCAT',
        'position', 'distance_from_football', 'relative_y_position'
    ]

    for col in required_columns:
        if col not in initial_positions.columns and col != 'position':
            initial_positions[col] = 0

    invalid_positions = initial_positions[~initial_positions['position'].isin(position_groups[side])]
    if not invalid_positions.empty:
        print(f"Warning: Found {side} players with invalid positions:")
        print(invalid_positions[['position']])
        initial_positions = initial_positions[initial_positions['position'].isin(position_groups[side])]

    for pos in position_groups[side]:
        col_name = f'position_{pos}'
        initial_positions[col_name] = (initial_positions['position'] == pos).astype(int)

    modeling_columns = [col for col in required_columns if col != 'position']
    modeling_columns.extend([f'position_{pos}' for pos in position_groups[side]])
    plot_columns = modeling_columns + ['position']

    initial_positions_for_model = initial_positions[modeling_columns]
    initial_positions_for_plot = initial_positions[modeling_columns + ['position']]

    predictions = predict_and_create_heatmap(
        model,
        initial_positions_for_model,
        scalers,
        scaler_y,
        side=side
    )

    if plot:
        side_suffix = "\nOffensive" if side == 'offense' else "\nDefensive"
        plot_route_heatmap_with_actual(
            initial_positions_for_plot,
            predictions,
            actual_routes,
            game_id,
            play_id,
            title_suffix=f" - {side_suffix} Routes",
            football_pos=football_pos
        )

    return predictions, actual_routes

#############################################################################################################

def plot_route_heatmap_with_actual(initial_positions, predictions, actual_routes=None,
                                 game_id=None, play_id=None, title_suffix="",
                                 football_pos=None):
    """
    Create a visualization of predicted and actual routes on a football field.
    
    Parameters:
        initial_positions (pd.DataFrame): Initial player positions and game context
        predictions (np.array): Predicted route probabilities
        actual_routes (list, optional): List of actual routes taken by players
        game_id (int, optional): Game identifier for title
        play_id (int, optional): Play identifier for title
        title_suffix (str, optional): Additional text to append to plot title
        football_pos (pd.DataFrame, optional): Football x,y coordinates
        
    Creates a matplotlib figure showing:
        - Football field with yard lines
        - Heatmap of predicted routes
        - Initial player positions with labels
        - Actual routes (if provided)
        - Football position (if provided)
        - Game situation information in title
    """
    # Create figure and set up football field
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 120)  # Field length
    ax.set_ylim(0, 53.3)  # Field width
    ax.set_facecolor('white')

    # Draw yard lines and numbers
    yard_numbers = {
        10: "G",   # Goal line
        20: "10",  # 10 yard line
        30: "20",  # 20 yard line
        40: "30",  # 30 yard line
        50: "40",  # 40 yard line
        60: "50",  # 50 yard line (midfield)
        70: "40",  # 40 yard line
        80: "30",  # 30 yard line
        90: "20",  # 20 yard line
        100: "10", # 10 yard line
        110: "G"   # Goal line
    }

    # Draw yard lines and numbers
    for yard, number in yard_numbers.items():
        ax.axvline(yard, color='black', alpha=0.2)
        if yard != 60:  # Don't label the 50 yard line
            ax.text(yard, 5, number, color='black', alpha=0.5, ha='center',
                   fontsize=14, fontweight='bold')

    # Clean up axes
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Create heatmap from predictions
    all_points = []
    for pred_set in predictions:
        for route in pred_set:
            all_points.extend(route)
    all_points = np.array(all_points)

    # Set up kernel density estimation
    kde = gaussian_kde(all_points.T, bw_method='scott')
    kde.set_bandwidth(kde.factor * 2)  # Adjust smoothing

    # Create grid for heatmap
    x_grid, y_grid = np.mgrid[0:120:100j, 0:53.3:100j]
    positions = np.vstack([x_grid.ravel(), y_grid.ravel()])

    # Calculate and normalize density
    density = kde(positions).reshape(x_grid.shape)
    density_normalized = density / density.max()

    # Remove very low probability areas
    threshold = 0.05
    density_normalized[density_normalized < threshold] = 0

    # Create custom colormap for heatmap
    colors = [(1, 1, 1, 0)]  # Start with transparent white
    n_colors = 256
    yellows = plt.cm.YlOrRd(np.linspace(0.1, 1, n_colors))
    yellows[:, 3] = np.linspace(0.3, 0.8, n_colors)  # Adjust transparency
    colors.extend(yellows)
    custom_cmap = plt.matplotlib.colors.ListedColormap(colors)

    # Plot the heatmap
    contour = ax.contourf(x_grid, y_grid, density_normalized, levels=20,
                         cmap=custom_cmap)

    # Plot initial player positions
    ax.scatter(initial_positions['x_normalized'],
              initial_positions['y_normalized'],
              c='blue', s=100)

    # Add player position labels
    for _, row in initial_positions.iterrows():
        ax.annotate(row['position'] if 'position' in row else 'NA',
                   (row['x_normalized'], row['y_normalized']),
                   xytext=(0, 10), textcoords='offset points',
                   ha='center', va='bottom',
                   color='black', fontweight='bold')

    # Plot actual routes if provided
    if actual_routes is not None:
        for route in actual_routes:
            ax.plot(route[:, 0], route[:, 1], color='blue', linewidth=2)

    # Plot football if position provided
    if football_pos is not None and 'football_x' in football_pos and 'football_y' in football_pos:
        # Plot football with brown fill
        ax.scatter(football_pos['football_x'], football_pos['football_y'],
                  c='brown', s=200, marker='o', label='Football')
        # Add black border around football
        ax.scatter(football_pos['football_x'], football_pos['football_y'],
                  facecolors='none', edgecolors='black', s=200, marker='o')

    # Create title with game information
    game_info = []
    if game_id:
        game_info.append(f"Game: {game_id}")
    if play_id:
        game_info.append(f"Play: {play_id}")
    if 'game_time_left_mins' in initial_positions.columns:
        time_left = round(initial_positions['game_time_left_mins'].iloc[0], 1)
        game_info.append(f"Minutes Left: {time_left}min")
    if 'down' in initial_positions.columns:
        game_info.append(f"Down: {int(initial_positions['down'].iloc[0])}")
    if 'yardsToGo' in initial_positions.columns:
        game_info.append(f"Yards to Go: {int(initial_positions['yardsToGo'].iloc[0])}")
    if 'offense_score_diff' in initial_positions.columns:
        game_info.append(f"Offense Score Difference: {int(initial_positions['offense_score_diff'].iloc[0])}")

    # Set plot title
    title = ' | '.join(game_info) + title_suffix
    ax.set_title(title, pad=20, color='black', fontsize=12)

    ax.grid(False)
    plt.tight_layout()
    plt.show()

#############################################################################################################

def calculate_kde_density(predictions, x_grid, y_grid, bandwidth_factor=1.75):
    """
    Calculate kernel density estimate for route predictions, creating a smooth probability
    distribution over the field.
    
    Parameters:
        predictions (np.array): Array of predicted routes, shape (n_predictions, n_players, n_steps, 2)
        x_grid (np.array): Grid of x-coordinates to evaluate density on
        y_grid (np.array): Grid of y-coordinates to evaluate density on
        bandwidth_factor (float): Factor to adjust KDE bandwidth (higher = smoother)
    
    Returns:
        np.array: Normalized density values over the specified grid, range [0,1]
    
    Notes:
        - Uses Scott's rule for initial bandwidth estimation, then adjusts by bandwidth_factor
        - Higher bandwidth_factor creates smoother, more spread out distributions
        - Lower bandwidth_factor creates more concentrated, peaked distributions
    """
    # Combine all predicted points into a single array
    all_points = []
    for pred_set in predictions:        # Loop through each set of predictions
        for route in pred_set:          # Loop through each player's route
            all_points.extend(route)     # Add all points from the route
    all_points = np.array(all_points)   # Convert to numpy array for efficiency

    # Create kernel density estimate
    # - Uses Scott's rule for initial bandwidth estimation
    # - Scott's rule: optimal for normal distributions
    kde = gaussian_kde(all_points.T, bw_method='scott')
    
    # Adjust bandwidth by factor
    # - Larger factor = smoother distribution
    # - Smaller factor = more detailed, possibly noisier distribution
    kde.set_bandwidth(kde.factor * bandwidth_factor)

    # Calculate density on the provided grid
    positions = np.vstack([x_grid.ravel(), y_grid.ravel()])  # Reshape grid for KDE
    density = kde(positions).reshape(x_grid.shape)           # Calculate and reshape density

    # Normalize density to [0, 1] range
    # This makes the densities comparable across different predictions
    return density / density.max()

#############################################################################################################

def calculate_overlap_metrics(density1, density2, threshold=0.01):
    """
    Calculate various metrics to quantify the overlap between two density distributions,
    typically used to compare offensive and defensive route coverages.
    
    Parameters:
        density1 (np.array): First density map (e.g., offensive routes)
        density2 (np.array): Second density map (e.g., defensive coverage)
        threshold (float): Minimum density value to be considered significant (0.01 = 1%)
    
    Returns:
        dict: Dictionary containing three overlap metrics:
            - iou: Intersection over Union (0-1, higher = more overlap)
            - overlap_coefficient: Overlap coefficient (0-1, higher = more overlap)
            - weighted_overlap: Cosine similarity of densities (0-1, higher = more similar distributions)
    
    Notes:
        - IoU is sensitive to the total area covered by both distributions
        - Overlap coefficient is more sensitive to the smaller distribution
        - Weighted overlap considers the actual density values, not just overlap area
    """
    # Create binary masks for significant density regions
    # Filters out very low density noise
    mask1 = density1 > threshold
    mask2 = density2 > threshold

    # Calculate Intersection over Union (IoU)
    # - Uses element-wise min for intersection
    # - Uses element-wise max for union
    # - Range: [0,1] where 1 = perfect overlap
    intersection = np.sum(np.minimum(density1, density2))
    union = np.sum(np.maximum(density1, density2))
    iou = intersection / union if union > 0 else 0

    # Calculate Overlap Coefficient
    # - Similar to IoU but normalized by smaller distribution
    # - Less sensitive to size differences between distributions
    # - Range: [0,1] where 1 = smaller distribution completely overlapped
    overlap_coef = intersection / min(np.sum(density1), np.sum(density2))

    # Calculate Weighted Overlap (Cosine Similarity)
    # - Considers actual density values, not just binary overlap
    # - Normalized by distribution magnitudes
    # - Range: [0,1] where 1 = identical distributions
    weighted_overlap = np.sum(density1 * density2) / np.sqrt(np.sum(density1**2) * np.sum(density2**2))

    return {
        'iou': iou,                           # Standard overlap metric
        'overlap_coefficient': overlap_coef,   # Size-invariant overlap metric
        'weighted_overlap': weighted_overlap   # Density-weighted similarity
    }

#############################################################################################################

def plot_route_overlap(initial_positions_off, initial_positions_def,
                      off_predictions, def_predictions,
                      off_routes=None, def_routes=None,
                      game_id=None, play_id=None, frame_id=None):
    """
    Create a side-by-side visualization comparing offensive and defensive route distributions
    and their overlap.
    
    Parameters:
        initial_positions_off (pd.DataFrame): Initial positions of offensive players
        initial_positions_def (pd.DataFrame): Initial positions of defensive players
        off_predictions (np.array): Predicted offensive routes
        def_predictions (np.array): Predicted defensive routes
        off_routes (list, optional): Actual offensive routes
        def_routes (list, optional): Actual defensive routes
        game_id (int, optional): Game identifier
        play_id (int, optional): Play identifier
        frame_id (int, optional): Frame identifier for animation purposes
        
    Returns:
        dict: Overlap metrics between offensive and defensive distributions
        
    Creates a figure with two subplots:
        Left: Offense (red) and defense (blue) route distributions
        Right: Coverage overlap intensity (purple)
    """
    # Create two-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Set up field dimensions and compute density grid
    x_grid, y_grid = np.mgrid[0:120:100j, 0:53.3:100j]  # Standard football field dimensions

    # Calculate route density distributions
    off_density = calculate_kde_density(off_predictions, x_grid, y_grid)
    def_density = calculate_kde_density(def_predictions, x_grid, y_grid)
    overlap_metrics = calculate_overlap_metrics(off_density, def_density)

    # Configure both subplots with field markings
    for ax in [ax1, ax2]:
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 53.3)
        ax.set_facecolor('white')

        # Draw yard lines and numbers
        yard_numbers = {
            10: "G", 20: "10", 30: "20", 40: "30", 50: "40",
            60: "50", 70: "40", 80: "30", 90: "20", 100: "10", 110: "G"
        }
        for yard, number in yard_numbers.items():
            ax.axvline(yard, color='black', alpha=0.2)
            if yard != 60:  # Skip 50 yard line number
                ax.text(yard, 5, number, color='black', alpha=0.5,
                       ha='center', fontsize=14, fontweight='bold')

        # Clean up axes
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.grid(False)

    # Left subplot: Offense and Defense distributions
    threshold = 0.01  # Minimum density to display
    off_density_masked = np.ma.masked_where(off_density < threshold, off_density)
    def_density_masked = np.ma.masked_where(def_density < threshold, def_density)

    # Create transparent colormaps
    off_cmap = plt.cm.Reds
    def_cmap = plt.cm.Blues
    off_cmap.set_bad(alpha=0)
    def_cmap.set_bad(alpha=0)

    # Plot offensive and defensive heatmaps
    ax1.contourf(x_grid, y_grid, off_density_masked, levels=20, cmap=off_cmap, alpha=0.5)
    ax1.contourf(x_grid, y_grid, def_density_masked, levels=20, cmap=def_cmap, alpha=0.5)

    # Right subplot: Coverage overlap
    overlap = np.minimum(off_density, def_density)
    overlap_masked = np.ma.masked_where(overlap < threshold, overlap)
    overlap_cmap = plt.cm.Purples
    overlap_cmap.set_bad(alpha=0)
    ax2.contourf(x_grid, y_grid, overlap_masked, levels=20, cmap=overlap_cmap)

    # Plot player positions and routes on both subplots
    for ax in [ax1, ax2]:
        # Plot player positions
        ax.scatter(initial_positions_off['x_normalized'], initial_positions_off['y_normalized'],
                  c='red', s=100, label='Offense')
        ax.scatter(initial_positions_def['x_normalized'], initial_positions_def['y_normalized'],
                  c='blue', s=100, label='Defense')

        # Plot football if available
        if 'football_x' in initial_positions_off.columns and 'football_y' in initial_positions_off.columns:
            football_pos = initial_positions_off[['football_x', 'football_y']].iloc[0]
            ax.scatter(football_pos['football_x'], football_pos['football_y'],
                      c='brown', s=200, marker='o')
            ax.scatter(football_pos['football_x'], football_pos['football_y'],
                      facecolors='none', edgecolors='black', s=200, marker='o')

        # Add position labels
        for _, row in initial_positions_off.iterrows():
            ax.annotate(row['position'] if 'position' in row else 'NA',
                       (row['x_normalized'], row['y_normalized']),
                       xytext=(0, 10), textcoords='offset points',
                       ha='center', va='bottom', color='red', fontweight='bold')

        for _, row in initial_positions_def.iterrows():
            ax.annotate(row['position'] if 'position' in row else 'NA',
                       (row['x_normalized'], row['y_normalized']),
                       xytext=(0, 10), textcoords='offset points',
                       ha='center', va='bottom', color='blue', fontweight='bold')

    # Add titles and metrics
    ax1.set_title('Separate Offense (Red) and Defense (Blue) Heatmaps', pad=20)
    ax2.set_title('Coverage Overlap Intensity (Purple)', pad=20)

    # Display overlap metrics
    metrics_text = (f"Overlap Metrics:\n"
                   f"IoU: {overlap_metrics['iou']:.3f}\n"
                   f"Overlap Coef: {overlap_metrics['overlap_coefficient']:.3f}\n"
                   f"Weighted Overlap: {overlap_metrics['weighted_overlap']:.3f}")
    
    if frame_id is not None:
        metrics_text += f"\nFrame: {frame_id}"

    # Add metrics text with white background
    plt.figtext(0.02, 0.02, metrics_text, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8))

    # Add game/play information if provided
    if game_id and play_id:
        title = f"Game: {game_id} | Play: {play_id}"
        if frame_id is not None:
            title += f" | Frame: {frame_id}"
        plt.suptitle(title, y=1.02)

    plt.tight_layout()
    return overlap_metrics

#############################################################################################################

def create_route_overlap_animation(off_presnap_df, def_presnap_df, off_model, def_model,
                                 off_scaler_X, def_scaler_X, off_scaler_y, def_scaler_y,
                                 game_id, play_id, fps=4):
    """
    Create an animated GIF showing route overlap predictions frame by frame and return metrics.
    
    Parameters:
        [previous parameters remain the same]
        
    Returns:
        pd.DataFrame: DataFrame containing overlap metrics for each frame
    """
    # Filter data for specific game and play
    off_play_data = off_presnap_df.filter(
        (pl.col('gameId') == game_id) &
        (pl.col('playId') == play_id)
    )
    def_play_data = def_presnap_df.filter(
        (pl.col('gameId') == game_id) &
        (pl.col('playId') == play_id)
    )
    
    # Get unique frame IDs
    frame_ids = off_play_data['frameId'].unique().sort()
    
    # Create temporary directory for frames
    temp_dir = tempfile.mkdtemp()
    frame_files = []
    
    # Store metrics for each frame
    metrics_data = []
    
    # Generate frames
    for frame_id in frame_ids:
        # Get data for current frame
        off_frame_data = off_play_data.filter(pl.col('frameId') == frame_id)
        def_frame_data = def_play_data.filter(pl.col('frameId') == frame_id)
        
        # Get predictions for current frame (without plotting individual heatmaps)
        off_predictions, _ = analyze_specific_play_by_side(
            off_frame_data,
            off_model,
            off_scaler_X,
            off_scaler_y,
            game_id,
            play_id,
            side='offense',
            plot=False  # Don't plot individual heatmaps
        )
        
        def_predictions, _ = analyze_specific_play_by_side(
            def_frame_data,
            def_model,
            def_scaler_X,
            def_scaler_y,
            game_id,
            play_id,
            side='defense',
            plot=False  # Don't plot individual heatmaps
        )
        
        if off_predictions is not None and def_predictions is not None:
            # Create frame visualization
            overlap_metrics = plot_route_overlap(
                initial_positions_off=off_frame_data.group_by('nflId').first().to_pandas(),
                initial_positions_def=def_frame_data.group_by('nflId').first().to_pandas(),
                off_predictions=off_predictions,
                def_predictions=def_predictions,
                game_id=game_id,
                play_id=play_id,
                frame_id=frame_id
            )
            
            # Store metrics with frame information
            metrics_data.append({
                'gameId': game_id,
                'playId': play_id,
                'frameId': frame_id,
                'iou': overlap_metrics['iou'],
                'overlap_coefficient': overlap_metrics['overlap_coefficient'],
                'weighted_overlap': overlap_metrics['weighted_overlap']
            })
            
            # Save frame
            frame_file = os.path.join(temp_dir, f'frame_{frame_id}.png')
            plt.savefig(frame_file, bbox_inches='tight', dpi=100)
            plt.close()
            frame_files.append(frame_file)
    
    # Create GIF
    if frame_files:
        images = []
        for file in frame_files:
            images.append(imageio.imread(file))
        
        output_file = os.path.join(temp_dir, 'route_overlap.gif')
        imageio.mimsave(output_file, images, fps=fps)
        
        # Display animation in notebook
        display(Image(filename=output_file))
        
        # Cleanup temporary files
        for file in frame_files:
            os.remove(file)
        os.remove(output_file)
        os.rmdir(temp_dir)
    else:
        print("No valid frames were generated.")
    
    # Create and return metrics DataFrame
    metrics_df = pd.DataFrame(metrics_data)
    return metrics_df
