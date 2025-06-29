from services.data_processing.scaler_manager import ScalerManager
from services.data_processing.abstract import BaseDataProcessor
from core.types import PreprocessedData

import numpy as np


class DataNormalizer(BaseDataProcessor):
    """
    Class that normalizes data for different model types using
    model-specific scalers during training and prediction.
    """

    def __init__(self, symbol, model_type: str, phase: str):
        self.symbol = symbol
        self.model_type = model_type
        self.scaler_manager = ScalerManager(model_type, symbol, phase)

    def process(self, data: PreprocessedData, fit=False) -> PreprocessedData:
        """
        Process the data by applying normalization if required.

        Args:
            data (PreprocessedData): Features and targets to normalize.
            fit (bool): If True, fit a new scaler.

        Returns:
            PreprocessedData: Normalized features and targets.
        """
        try:

            if self.scaler_manager.model_requires_scaling():

                # Extract features and targets
                X = data.X
                y = data.y

                original_X_shape = X.shape
                original_y_shape = y.shape

                # Reshape for scaler
                if isinstance(X, np.ndarray):
                    if X.ndim == 3:
                        X = X.reshape(-1, X.shape[-1])
                    if X.ndim == 1:
                        X = X.reshape(-1, 1)

                if isinstance(y, np.ndarray):
                    if y.ndim == 3:
                        y = y.reshape(-1, y.shape[-1])
                    if y.ndim == 1:
                        y = y.reshape(-1, 1)

                # Load (or create) the scaler
                X_scaler = self._load_or_fit_scaler("features", fit=fit, data=X)
                y_scaler = self._load_or_fit_scaler("targets", fit=fit, data=y)

                # Scale the data
                scaled_X = X_scaler.transform(X)
                scaled_y = y_scaler.transform(y)

                # RESHAPE BACK AFTER SCALING
                if len(original_X_shape) == 3:
                    scaled_X = scaled_X.reshape(original_X_shape)

                if len(original_y_shape) == 3:
                    scaled_y = scaled_y.reshape(original_y_shape)
                elif len(original_y_shape) == 1:
                    scaled_y = scaled_y.reshape(-1)  # Flatten y back to 1D if it was 1D

                return PreprocessedData(X=scaled_X, y=scaled_y)

            else:
                # No Normalization needed
                return data

        except Exception as e:
            raise RuntimeError(f"Error while scaling data.") from e

    def unprocess(self, data: PreprocessedData):
        """
        Reverse the normalization applied to the targets included in the data.

        This method uses the stored scaler to apply inverse transformation
        to the given target data, restoring it to its original scale.

        Args:
            data (PreprocessedData): Features and targets.

        Returns:
            PreprocessedData: Unnormalized target values in the PreprocessedData format.
        """

        if self.scaler_manager.model_requires_scaling():

            # Extract the targets
            y = data.y

            original_y_shape = y.shape

            # Reshape for scaler
            if isinstance(y, np.ndarray):
                if y.ndim == 3:
                    y = y.reshape(-1, y.shape[-1])
                if y.ndim == 1:
                    y = y.reshape(-1, 1)

            # Load the targets scaler
            y_scaler = self.scaler_manager.load_scaler(
                ScalerManager.TARGETS_SCALER_TYPE
            )

            # Unscale the targets
            unscaled_y = y_scaler.inverse_transform(y)

            if len(original_y_shape) == 3:
                unscaled_y = unscaled_y.reshape(original_y_shape)
            elif len(original_y_shape) == 1:
                unscaled_y = unscaled_y.reshape(-1)  # Flatten y back to 1D if it was 1D

            return PreprocessedData(y=unscaled_y)
        else:
            # No Unnormalize needed
            return data

    def _load_or_fit_scaler(self, scaler_type: str, fit: bool, data):
        """
        Returns the appropriate scaler based on the model type.
        """
        try:
            if fit:
                # Create it
                scaler = self.scaler_manager.create_scaler()
                # Fit and save the scaler
                scaler.fit(data)
                self.scaler_manager.save_scaler(scaler, scaler_type)
            else:
                scaler = self.scaler_manager.load_scaler(scaler_type)

            return scaler

        except Exception as e:
            raise RuntimeError(f"Error preparing the scaler.") from e
