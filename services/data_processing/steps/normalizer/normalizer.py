from services.data_processing.scaler_manager import ScalerManager
from services.data_processing.abstract import BaseDataProcessor
from core.types import ProcessedData

import numpy as np


class DataNormalizer(BaseDataProcessor):
    """
    Class that normalizes data for different model types using
    model-specific scalers during training and prediction.
    """

    def __init__(self, symbol, model_type: str, phase: str):
        self.symbol = symbol
        self.model_type = model_type
        self.phase = phase
        self.scaler_manager = ScalerManager(model_type, symbol)

    def process(self, data: ProcessedData, fit=False) -> ProcessedData:
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

                results = []

                for scaler_type, inputs in {"features": X, "targets": y}.items():

                    if inputs is not None:
                        # Keep the origianl shape of the data
                        original_shape = inputs.shape

                        # Reshape for scaler
                        if inputs.ndim == 3:
                            inputs = inputs.reshape(-1, inputs.shape[-1])
                        if inputs.ndim == 1:
                            inputs = inputs.reshape(-1, 1)

                        # Load (or create) the scaler
                        scaler = self._load_or_fit_scaler(
                            scaler_type, fit=fit, data=inputs
                        )

                        # Scale the data
                        scaled_data = scaler.transform(inputs)

                        # RESHAPE BACK AFTER SCALING
                        if len(original_shape) == 3:
                            scaled_data = scaled_data.reshape(original_shape)

                        elif len(original_shape) == 1:
                            scaled_data = scaled_data.reshape(
                                -1
                            )  # Flatten y back to 1D if it was 1D

                        results.append(scaled_data)
                    else:
                        results.append(None)

                return ProcessedData(
                    X=results[0], y=results[1], feature_index_map=data.feature_index_map
                )

            else:
                # No Normalization needed
                return data

        except Exception as e:
            raise RuntimeError(f"Error while scaling data.") from e

    def unprocess(self, data: ProcessedData):
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
                self.phase, ScalerManager.TARGETS_SCALER_TYPE
            )

            # Unscale the targets
            unscaled_y = y_scaler.inverse_transform(y)

            if len(original_y_shape) == 3:
                unscaled_y = unscaled_y.reshape(original_y_shape)
            elif len(original_y_shape) == 1:
                unscaled_y = unscaled_y.reshape(-1)  # Flatten y back to 1D if it was 1D

            return ProcessedData(y=unscaled_y)
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
                self.scaler_manager.save_scaler(scaler, self.phase, scaler_type)
            else:
                scaler = self.scaler_manager.load_scaler(self.phase, scaler_type)

            return scaler

        except Exception as e:
            raise RuntimeError(f"Error preparing the scaler.") from e
