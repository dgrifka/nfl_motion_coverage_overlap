# nfl_motion_coverage_overlap
NFL Big Data Bowl 2025 Code - NFL Pre-Snap Motion Analysis

https://www.kaggle.com/code/derekgrifka/overlap-how-can-pre-snap-motion-exploit-it

A machine learning approach to analyzing how pre-snap motion impacts Defensive coverage in the NFL using player tracking data.

# Overview

This project uses neural networks to generate heat maps of likely player positions in the first ~2.6 seconds after snap, analyzing how pre-snap motion affects the spatial relationship between offense and defense. The analysis compares Man vs. Zone coverage responses to motion using a Bayesian approach.

# Key Features

•	Neural network model for predicting player movement probabilities
•	Monte Carlo sampling for movement prediction
•	Heat map generation for Offensive and Defensive player positioning
•	Comparative analysis of Man vs. Zone coverage responses
•	Bayesian statistical modeling for coverage comparison

# Data

The model uses 2022 NFL player tracking data, focusing on:
•	Offensive players: WR, TE, RB
•	Defensive players: LB, MLB, ILB, OLB, DB, CB, SS, FS
•	~2.6 seconds of post-snap passing data
•	Pre-snap motion plays

# Methodology

1.	Data Preparation 
o	Standardized play directions
o	Calculated relative player positioning
o	Incorporated game context features

2.	Heat Map Generation 
o	Monte Carlo sampling (200 variations per player)
o	Position-dependent noise modeling
o	Custom loss function for realistic movements
o	Kernel density estimation

3.	Analysis 
o	Weighted overlap scoring using cosine similarity
o	Frame-by-frame coverage analysis
o	Bayesian comparative modeling

# Key Findings

•	Man coverage shows higher overlap coefficients (0.51 vs 0.38)
•	Man coverage exhibits more volatility during motion sequences
•	Zone coverage more likely (~91%) to experience negative deltas in overlap
•	Neither coverage type is inherently superior; effectiveness depends on Offensive strategy

# Applications

•	Real-time coverage gap analysis
•	Defensive strategy adjustment
•	Player development and evaluation
•	Coverage pattern recognition

# Future Development

•	Incorporate player speed/acceleration profiles
•	Expand Defensive scheme identification
•	Extend temporal analysis window
•	Include environmental factors
•	Add run play analysis

# Installation & Usage

This project uses Google Colab for analysis. The `Model_Training.ipynb`, `Model_Predictions.ipynb`, and `Overlap_Analysis.ipynb` files in the `colab` folder contain information and instructions on cloning the repo in Colab for your own purposes. These files are as described, but reference the `data_cleaning`, `model`, and `analysis` folders.

# Project Structure

├── data_cleaning/

│   ├── data_cleaning.py         # Data cleaning and preparation

│   └── utils.py                 # Helper functions

├── model/

│   ├── model_predictions.py       # Heat map predictions

│   ├── model_train.py            # Neural network architecture

│   └── zone_man_overlap/         # Frame-by-frame predictions

├── Colab/

│   ├── Model_Predictions.ipynb       # Prediction notebook

│   ├── Model_Training.ipynb         # Training notebook

│   └── Overlap_Analysis.ipynb       # Probability programming notebook

├── analysis/

│   └── analysis.py       # Distribution plots

├── setup.py                    # Package configuration

├── img/  # Folder of images referenced in Kaggle notebook

└── README.md                   # Data documentation

# Contact

Derek Grifka – dmgrifka@gmail.com - https://dgrifka.github.io/
