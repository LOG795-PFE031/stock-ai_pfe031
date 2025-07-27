"""
Module containing constants used in the data processing service
"""

# ML/AI models that need scaling in preprocessing
SCALABLE_MODELS = {"lstm"}

# Types of scalers used for preprocessing different parts of the data
FEATURES_SCALER_TYPE = "features"
TARGETS_SCALER_TYPE = "targets"
VALID_SCALER_TYPES = {FEATURES_SCALER_TYPE, TARGETS_SCALER_TYPE}

# ML pipeline phases that require scalers.
# Only training and prediction/production phases use scalers.
# The evaluation phase reuses the scaler from the prediction/production phase.
TRAINING_PHASE = "training"
PREDICTION_PHASE = "prediction"
VALID_PHASES = [PREDICTION_PHASE, TRAINING_PHASE]
