from ...scaler_manager import ScalerManager
from ..abstract import BaseDataProcessor
from core.types import ProcessedData

import numpy as np
from typing import Optional


class DataNormalizer(BaseDataProcessor):
    """
    Class that normalizes data for different model types using
    model-specific scalers during training and prediction.
    """

    def __init__(self, symbol, model_type: str):
        self.symbol = symbol
        self.model_type = model_type
        self.scaler_manager = ScalerManager(model_type, symbol)

    def process(
        self,
        data: ProcessedData,
        phase="prediction",
        fit=False,
        scaler_dates: Optional[tuple[str, str]] = None,
    ) -> ProcessedData:
        """
        Processes the given data by applying scaling to features and targets if required
        based on the current phase.

        Args:
            data (ProcessedData): The input data containing features (X) and targets (y) to be processed.
            phase (str, optional): The current phase of the pipeline (e.g., "training", "prediction").
                                Determines which scaler to use. Defaults to "prediction".
            fit (bool, optional): Whether to fit a new scaler using the input data.
                                If False, an existing scaler is loaded. Defaults to False.
            scaler_dates (Optional[tuple[str, str]], optional): Tuple containing start and end dates
                                                                for fitting the scaler, if applicable.
                                                                Defaults to None.

        Returns:
            ProcessedData: Preprocessed data (scaled features and targets).
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

                        if fit:
                            scaler = self._create_and_fit_scaler(
                                scaler_type=scaler_type,
                                data=inputs,
                                scaler_dates=scaler_dates,
                            )
                        else:
                            scaler = self.scaler_manager.load_scaler(
                                scaler_type=scaler_type, phase=phase
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
            raise RuntimeError(f"Error while scaling data : {str(e)}") from e

    def unprocess(self, data: ProcessedData, phase: str):
        """
        Reverse the normalization applied to the targets included in the data.

        This method uses the stored scaler to apply inverse transformation
        to the given target data, restoring it to its original scale.

        Args:
            data (PreprocessedData): Features and targets.
            phase (str): The phase (e.g., "training", "prediction").

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
                scaler_type=ScalerManager.TARGETS_SCALER_TYPE, phase=phase
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

    def _create_and_fit_scaler(
        self, scaler_type: str, data, scaler_dates: tuple[str, str]
    ):
        """
        Creates, fits, and saves a scaler for the given data and type.

        Args:
            scaler_type (str): A string indicating whether the scaler is for "features" or "targets".
            data: The input data to be used for fitting the scaler.
            scaler_dates (tuple[str, str]): A tuple representing the start and end dates
                                            used to label/version the fitted scaler.

        Returns:
            sklearn.preprocessing.MinMaxScaler: A fitted scaler.
        """
        try:
            if not scaler_dates:
                raise ValueError("You must provide scaler_dates when fitting a scaler.")

            # Create it
            scaler = self.scaler_manager.create_scaler()

            # Fit it
            scaler = scaler.fit(data)

            # Save it
            self.scaler_manager.save_scaler(
                scaler=scaler, scaler_type=scaler_type, scaler_dates=scaler_dates
            )

            return scaler

        except Exception as e:
            raise RuntimeError(f"Error creating and fitting the scaler.") from e
