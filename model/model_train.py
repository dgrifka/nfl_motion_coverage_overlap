# Data manipulation libraries
import numpy as np
import polars as pl
import pandas as pd
import pickle

# Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Deep Learning - TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Add, BatchNormalization, Concatenate, Dense, Dropout,
    GRU, Input, LSTM, Lambda, Multiply
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Progress tracking
from tqdm import tqdm

#############################################################################################################

def prepare_sequences(df, prediction_steps=10, side='offense'):
    """
    Prepare sequences for training with position encoding and side-specific filtering
    """
    print("Preparing sequences for training...")

    # Define position groups
    position_groups = {
        'offense': ['WR', 'TE', 'RB'],
        'defense': ['LB', 'MLB', 'ILB', 'OLB', 'DB', 'CB', 'SS', 'FS']
    }

    # Filter by position group
    df_filtered = df.filter(pl.col('position').is_in(position_groups[side]))
    df_pandas = df_filtered.sort('frameId').to_pandas()

    X = []  # Input features
    y = []  # Output sequences

    # One-hot encode offensive formation
    formation_categories = [
        'SHOTGUN', 'UNDER_CENTER', 'PISTOL',
        'NO_HUDDLE_SHOTGUN', 'NO_HUDDLE', 'WILDCAT'
    ]
    formation_dummies = pd.get_dummies(
        df_pandas['offenseFormation'],
        prefix='formation',
        columns=formation_categories
    )

    # Ensure all formation columns exist
    for formation in formation_categories:
        col_name = f'formation_{formation}'
        if col_name not in formation_dummies.columns:
            print(f'Adding missing formation column: {col_name}')
            formation_dummies[col_name] = 0

    formation_dummies = formation_dummies[
        [f'formation_{f}' for f in formation_categories]
    ]

    # One-hot encode positions based on side
    position_categories = position_groups[side]
    position_dummies = pd.get_dummies(
        df_pandas['position'],
        prefix='position',
        columns=position_categories
    )

    # Ensure all position columns exist
    for position in position_categories:
        col_name = f'position_{position}'
        if col_name not in position_dummies.columns:
            print(f'Adding missing position column: {col_name}')
            position_dummies[col_name] = 0

    position_dummies = position_dummies[
        [f'position_{p}' for p in position_categories]
    ]

    for _, group in tqdm(df_pandas.groupby(['gameId', 'playId', 'nflId'])):
        # Get position features
        position_features = group[['x_normalized', 'y_normalized', 'o_normalized',
                                 'yardsToGo', 'absoluteYardlineNumber', 'distance_from_football',
                                 'relative_y_position']].values[0]

        # Get game context features
        context_features = [
            group['offense_score_diff'].values[0],
            group['down'].values[0],
            group['game_time_left_mins'].values[0],
            group['quarter'].values[0],
            group['quarter_mins_left'].values[0]
        ]

        # Get formation one-hot features
        formation_features = formation_dummies.loc[group.index[0]].values

        # Get position one-hot features
        position_one_hot = position_dummies.loc[group.index[0]].values

        # Combine all features
        input_features = np.concatenate([
            position_features,
            context_features,
            formation_features,
            position_one_hot
        ])

        # Get future positions for output
        future_positions = group[['x_normalized', 'y_normalized']].values[1:prediction_steps+1]

        # Skip if trajectory is too short
        if len(future_positions) < prediction_steps:
            continue

        X.append(input_features)
        y.append(future_positions)

    return np.array(X), np.array(y)

#############################################################################################################

def build_improved_model(input_shapes, output_shape):
    """
    Build an improved model with position encoding
    """
    # Position features branch
    position_input = Input(shape=(input_shapes[0],))
    position_dense = Dense(64, activation='relu')(position_input)
    position_bn = BatchNormalization()(position_dense)

    # Context features branch
    context_input = Input(shape=(input_shapes[1],))
    context_dense = Dense(32, activation='relu')(context_input)
    context_bn = BatchNormalization()(context_dense)

    # Formation features branch
    formation_input = Input(shape=(input_shapes[2],))
    formation_dense = Dense(32, activation='relu')(formation_input)
    formation_bn = BatchNormalization()(formation_dense)

    # Position encoding branch
    position_encoding_input = Input(shape=(input_shapes[3],))
    position_encoding_dense = Dense(32, activation='relu')(position_encoding_input)
    position_encoding_bn = BatchNormalization()(position_encoding_dense)

    # Merge all features
    merged = Concatenate()([position_bn, context_bn, formation_bn, position_encoding_bn])

    # Enhanced network architecture
    x = Dense(512, activation='relu')(merged)  # Increased capacity
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # First residual block
    residual = x
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, residual])

    # Second residual block
    residual = x
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, residual])

    # Sequence modeling
    x_reshaped = tf.keras.layers.Reshape((1, 512))(x)
    lstm = LSTM(256, return_sequences=True)(x_reshaped)  # Increased units
    gru = GRU(256)(lstm)  # Increased units

    # Final layers
    x = Dense(1024, activation='relu')(gru)  # Increased capacity
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # Movement direction bias for offensive players
    outputs = Dense(output_shape, activation='linear')(x)

    model = Model(
        inputs=[position_input, context_input, formation_input, position_encoding_input],
        outputs=outputs
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')

    return model

#############################################################################################################

def build_improved_offensive_model(input_shapes, output_shape, prediction_steps=10):
    """
    Build a position-aware model for offensive players with separate branches
    """
    # Position features branch
    position_input = Input(shape=(input_shapes[0],))
    position_dense = Dense(64, activation='relu')(position_input)
    position_bn = BatchNormalization()(position_dense)

    # Context features branch
    context_input = Input(shape=(input_shapes[1],))
    context_dense = Dense(32, activation='relu')(context_input)
    context_bn = BatchNormalization()(context_dense)

    # Formation features branch
    formation_input = Input(shape=(input_shapes[2],))
    formation_dense = Dense(32, activation='relu')(formation_input)
    formation_bn = BatchNormalization()(formation_dense)

    # Position encoding branch (WR, TE, RB)
    position_encoding_input = Input(shape=(input_shapes[3],))

    # Separate branches for each position
    # WR Branch
    wr_mask = Lambda(lambda x: x[:, 0:1])(position_encoding_input)
    wr_branch = Dense(64, activation='relu')(position_bn)
    wr_branch = Dense(64, activation='relu')(wr_branch)
    wr_branch = Multiply()([wr_branch, wr_mask])

    # TE Branch
    te_mask = Lambda(lambda x: x[:, 1:2])(position_encoding_input)
    te_branch = Dense(64, activation='relu')(position_bn)
    te_branch = Dense(64, activation='relu')(te_branch)
    te_branch = Multiply()([te_branch, te_mask])

    # RB Branch
    rb_mask = Lambda(lambda x: x[:, 2:3])(position_encoding_input)
    rb_branch = Dense(64, activation='relu')(position_bn)
    rb_branch = Dense(64, activation='relu')(rb_branch)
    rb_branch = Multiply()([rb_branch, rb_mask])

    # Merge position-specific branches
    position_merged = Add()([wr_branch, te_branch, rb_branch])

    # Merge with context and formation
    merged = Concatenate()([
        position_merged,
        context_bn,
        formation_bn
    ])

    # Main network
    x = Dense(512, activation='relu')(merged)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # Route-specific residual blocks
    # Deep routes
    deep_branch = Dense(256, activation='relu')(x)
    deep_branch = BatchNormalization()(deep_branch)
    deep_branch = Dense(256, activation='relu')(deep_branch)

    # Medium routes
    medium_branch = Dense(256, activation='relu')(x)
    medium_branch = BatchNormalization()(medium_branch)
    medium_branch = Dense(256, activation='relu')(medium_branch)

    # Short routes
    short_branch = Dense(256, activation='relu')(x)
    short_branch = BatchNormalization()(short_branch)
    short_branch = Dense(256, activation='relu')(short_branch)

    # Merge route branches
    route_merged = Concatenate()([deep_branch, medium_branch, short_branch])

    # Final layers
    x = Dense(512, activation='relu')(route_merged)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # Add directional bias for offensive movement
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)

    outputs = Dense(output_shape, activation='linear')(x)

    model = Model(
        inputs=[position_input, context_input, formation_input, position_encoding_input],
        outputs=outputs
    )

    # Define the custom loss function that was used in training
    def route_aware_loss(y_true, y_pred, prediction_steps=10):
        """
        Custom loss function that penalizes unrealistic movements
        """
        # Calculate MSE manually
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

        # Extract predicted coordinates
        pred_sequences = tf.reshape(y_pred, (-1, prediction_steps, 2))

        # Penalize backwards movement for offensive players
        x_movements = pred_sequences[:, 1:, 0] - pred_sequences[:, :-1, 0]
        backwards_penalty = tf.reduce_mean(tf.maximum(0.0, -x_movements)) * 0.5

        # Penalize excessive lateral movement
        y_movements = tf.abs(pred_sequences[:, 1:, 1] - pred_sequences[:, :-1, 1])
        lateral_penalty = tf.reduce_mean(tf.maximum(0.0, y_movements - 2.0)) * 0.3

        # Add route coherence loss
        route_variance = tf.reduce_mean(tf.math.reduce_variance(pred_sequences, axis=1))
        coherence_penalty = route_variance * 0.2

        return mse_loss + backwards_penalty + lateral_penalty + coherence_penalty

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=route_aware_loss)

    return model

#############################################################################################################

def train_improved_route_predictor(df, prediction_steps=10, side='offense'):
    """
    Train the improved model with position encoding and side-specific filtering
    """
    # Prepare sequences
    X, y = prepare_sequences(df, prediction_steps=prediction_steps, side=side)
    X = X.astype(np.float32)

    # Define feature splits
    n_position_features = 7
    n_context_features = 5
    n_formation_features = 6

    # Calculate number of position features based on side
    n_position_encoding_features = 3 if side == 'offense' else 8  # Number of possible positions

    # Split features
    X_position = X[:, :n_position_features]
    start_idx = n_position_features
    X_context = X[:, start_idx:start_idx + n_context_features]
    start_idx += n_context_features
    X_formation = X[:, start_idx:start_idx + n_formation_features]
    start_idx += n_formation_features
    X_position_encoding = X[:, start_idx:]

    # Scale features
    position_scaler = StandardScaler()
    context_scaler = StandardScaler()

    X_position_scaled = position_scaler.fit_transform(X_position)
    X_context_scaled = context_scaler.fit_transform(X_context)

    # One-hot encodings don't need scaling
    X_formation = X_formation.astype(np.float32)
    X_position_encoding = X_position_encoding.astype(np.float32)

    # Prepare output
    y_flat = y.reshape(y.shape[0], -1)
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y_flat)

    # Split data
    train_idx, test_idx = train_test_split(
        np.arange(len(X_position_scaled)),
        test_size=0.2,
        random_state=42
    )

    # Create training and test sets
    train_position = X_position_scaled[train_idx]
    train_context = X_context_scaled[train_idx]
    train_formation = X_formation[train_idx]
    train_position_encoding = X_position_encoding[train_idx]
    train_y = y_scaled[train_idx]

    test_position = X_position_scaled[test_idx]
    test_context = X_context_scaled[test_idx]
    test_formation = X_formation[test_idx]
    test_position_encoding = X_position_encoding[test_idx]
    test_y = y_scaled[test_idx]

    # Build model with correct input shapes
    input_shapes = (
        train_position.shape[1],
        train_context.shape[1],
        train_formation.shape[1],
        train_position_encoding.shape[1]
    )

    # Calculate output shape from y data
    output_shape = train_y.shape[1]

    # Select appropriate model based on side
    if side == 'offense':
        model = build_improved_offensive_model(input_shapes, output_shape, prediction_steps)
    else:
        model = build_improved_model(input_shapes, output_shape)

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6
        )
    ]

    history = model.fit(
        [train_position, train_context, train_formation, train_position_encoding],
        train_y,
        validation_data=([test_position, test_context, test_formation, test_position_encoding], test_y),
        epochs=150,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    return model, (position_scaler, context_scaler), y_scaler, history

#############################################################################################################

def route_aware_loss(y_true, y_pred, prediction_steps=10):
    """Custom loss function that penalizes unrealistic movements"""
    # Calculate MSE manually
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

    # Extract predicted coordinates
    pred_sequences = tf.reshape(y_pred, (-1, prediction_steps, 2))

    # Penalize backwards movement for offensive players
    x_movements = pred_sequences[:, 1:, 0] - pred_sequences[:, :-1, 0]
    backwards_penalty = tf.reduce_mean(tf.maximum(0.0, -x_movements)) * 0.5

    # Penalize excessive lateral movement
    y_movements = tf.abs(pred_sequences[:, 1:, 1] - pred_sequences[:, :-1, 1])
    lateral_penalty = tf.reduce_mean(tf.maximum(0.0, y_movements - 2.0)) * 0.3

    # Add route coherence loss
    route_variance = tf.reduce_mean(tf.math.reduce_variance(pred_sequences, axis=1))
    coherence_penalty = route_variance * 0.2

    return mse_loss + backwards_penalty + lateral_penalty + coherence_penalty

#############################################################################################################

def save_models_and_scalers(model_name, model, scaler_X, scaler_y, history):
    """Save a model, its scalers, and training history as individual files."""
    # First register the custom loss function
    tf.keras.utils.get_custom_objects()['route_aware_loss'] = route_aware_loss

    # Save the model
    model.save(f'{model_name}_model.keras')

    # Save the scalers
    position_scaler, context_scaler = scaler_X
    scalers = {
        'position_scaler': position_scaler,
        'context_scaler': context_scaler,
        'y_scaler': scaler_y
    }
    with open(f'{model_name}_scalers.pkl', 'wb') as f:
        pickle.dump(scalers, f)

    # Save the training history
    with open(f'{model_name}_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)

#############################################################################################################

def load_model_and_scalers(model_name):
    """Load a saved model and its scalers from individual files."""
    # Register the custom loss function
    tf.keras.utils.get_custom_objects()['route_aware_loss'] = route_aware_loss

    # Enable unsafe deserialization
    tf.keras.config.enable_unsafe_deserialization()

    # Load the model
    model = tf.keras.models.load_model(f'{model_name}_model.keras')

    # Load the scalers
    with open(f'{model_name}_scalers.pkl', 'rb') as f:
        scalers = pickle.load(f)

    # Load the history
    with open(f'{model_name}_history.pkl', 'rb') as f:
        history = pickle.load(f)

    return (
        model,
        (scalers['position_scaler'], scalers['context_scaler']),
        scalers['y_scaler'],
        history
    )
