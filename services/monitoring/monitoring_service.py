import asyncio
import time
from collections import deque
import contextlib
from typing import Dict, Tuple

import httpx
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from core import BaseService
from core.logging import logger
from core.config import config
from core.prometheus_metrics import evaluation_mae
from .schemas import ModelListMlflowResponse
from httpx import ReadTimeout, ConnectTimeout


class MonitoringService(BaseService):
    def __init__(
        self,
        check_interval_seconds: int = 24 * 60 * 60,  # default: once per day
        data_interval_seconds: int = 7 * 24 * 60 * 60,  # once per week
        max_drift_samples: int = 500,
        drift_stat_threshold: float = 0.1,
        data_drift_history_length: int = 8,
        data_drift_zscore: float = 2.0,
    ):
        super().__init__()

        self.logger = logger["monitoring"]
        self.check_interval_seconds = check_interval_seconds
        self.data_interval = data_interval_seconds
        self.max_drift_samples = max_drift_samples
        self.drift_stat_threshold = drift_stat_threshold
        self._last_mae: Dict[Tuple[str, str], float] = {}
        self._drift_history_length = data_drift_history_length
        self._drift_zscore = data_drift_zscore
        self._drift_history: Dict[Tuple[str, str], deque] = {}

        # Background tasks for performance and data loops
        self._perf_task: asyncio.Task | None = None
        self._data_task: asyncio.Task | None = None
        self.logger.info("Initializing MonitoringService")

    async def initialize(self) -> None:
        self._initialized = True
        # Start background loops
        self._perf_task = asyncio.create_task(self._performance_loop())
        self._data_task = asyncio.create_task(self._data_loop())
        self.logger.info("MonitoringService initialized and loops started")

    async def cleanup(self) -> None:
        """Stop both monitoring loops and clean up."""
        self._initialized = False
        # Cancel performance and data tasks if running
        for task in (self._perf_task, self._data_task):
            if task:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self.logger.info("MonitoringService cleaned up successfully.")

    async def _performance_loop(self) -> None:
        """Performance monitoring loop with monitoring config interval."""
        while True:
            try:
                await self._run_performance_check()
                await asyncio.sleep(config.monitoring.PERFORMANCE_CHECK_INTERVAL)
            except Exception as e:
                self.logger.error(f"Error in performance loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying

    async def _data_loop(self) -> None:
        """Data drift monitoring loop with monitoring config interval."""
        while True:
            try:
                await self._run_data_drift_check()
                await asyncio.sleep(config.monitoring.DATA_DRIFT_CHECK_INTERVAL)
            except Exception as e:
                self.logger.error(f"Error in data drift loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying

    async def _run_performance_check(self) -> None:
        """
        1. List all live models (e.g. 'lstm_AAPL')
        2. For each: run evaluation, read MAE, compare to last run, retrain if worse.
        """
        self.logger.info("Starting daily drift check")
        try:
            # URL to the endpoint to preprocess the data
            url = f"http://{config.deployment_service.HOST}:{config.deployment_service.PORT}/deployment/models"
            async with httpx.AsyncClient(timeout=None) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                payload = resp.json()

            # Parse with Pydantic to get .models list
            mlflow_list = ModelListMlflowResponse(**payload)
            live_models = mlflow_list.models

            for model in live_models:
                full_name = model.name
                if not full_name or "_" not in full_name:
                    self.logger.warning(f"Skipping unexpected model entry: {model}")
                    continue

                # now split on the first underscore
                model_type, symbol = full_name.split("_", 1)
                symbol = symbol.upper()

                # 1) Run evaluation pipeline (updates the evaluation_mae gauge)
                await self._run_evaluation_pipeline(model_type, symbol)

                # self.logger.info(f"Evaluation pipeline result is: {result}")

                # 2) Grab the just-set MAE
                current = evaluation_mae.labels(
                    model_type=model_type, symbol=symbol
                )._value.get()
                if current is None:
                    self.logger.warning(f"No MAE for {model}, skipping drift check")
                    continue

                key = (model_type, symbol)
                prev = self._last_mae.get(key)
                self._last_mae[key] = current

                # 3) Compare and retrain if MAE worsened
                if prev is not None and current > prev:
                    self.logger.info(
                        f"MAE increased for {model}: {prev:.4f} → {current:.4f}; retraining..."
                    )
                    await self._run_training_pipeline(model_type, symbol)
                else:
                    self.logger.info(
                        f"No retraining needed for {model}, prev MAE: {prev} vs current MAE: {current}"
                    )
        except Exception as e:
            self.logger.error(f"Error in daily drift check: {e}")
        finally:
            self.logger.info("🏁 Daily drift check complete")

    async def _fetch_stock_data(self, symbol: str, days_back: int) -> pd.DataFrame:
        """
        Fetch stock data from the API endpoint and convert to DataFrame.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            days_back: Number of days to look back

        Returns:
            DataFrame with stock data
        """
        start_time = time.time()
        try:
            # API endpoint URL
            api_url = f"http://{config.data.HOST}:{config.data.PORT}/data/stock/recent"
            params = {"symbol": symbol, "days_back": days_back}

            # Use monitoring config timeout settings
            timeout = httpx.Timeout(
                connect=config.monitoring.CONNECT_TIMEOUT,
                read=config.monitoring.DATA_FETCH_TIMEOUT,
                write=config.monitoring.CONNECT_TIMEOUT,
                pool=config.monitoring.CONNECT_TIMEOUT,
            )

            async with httpx.AsyncClient(timeout=timeout) as client:
                self.logger.info(
                    f"Fetching stock data for {symbol} (looking back {days_back} days)"
                )
                response = await client.get(api_url, params=params)
                response.raise_for_status()
                data = response.json()

            # Extract prices from the response
            prices = data.get("prices", [])
            if not prices:
                raise ValueError(f"No price data found for {symbol}")

            # Convert to DataFrame
            df = pd.DataFrame(prices)

            # Log the original columns for debugging
            self.logger.debug(f"Original columns for {symbol}: {list(df.columns)}")

            # Check for duplicate columns and handle them
            if "Date" in df.columns and "date" in df.columns:
                # If both exist, drop the lowercase one
                df = df.drop(columns=["date"])
            elif "date" in df.columns:
                # Rename lowercase to uppercase
                df = df.rename(columns={"date": "Date"})

            # Convert date column to datetime
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
            else:
                raise ValueError(
                    f"No 'Date' column found for {symbol}. Available columns: {list(df.columns)}"
                )

            # Sort by date
            df = df.sort_values("Date")

            elapsed = time.time() - start_time
            self.logger.info(
                f"Successfully fetched {len(df)} rows for {symbol} in {elapsed:.2f} seconds"
            )

            return df

        except ReadTimeout as e:
            elapsed = time.time() - start_time
            self.logger.error(
                f"ReadTimeout for {symbol} after {elapsed:.2f} seconds: {e}"
            )
            raise
        except ConnectTimeout as e:
            elapsed = time.time() - start_time
            self.logger.error(
                f"ConnectTimeout for {symbol} after {elapsed:.2f} seconds: {e}"
            )
            raise
        except Exception as e:
            elapsed = time.time() - start_time
            self.logger.error(
                f"Error fetching stock data for {symbol} after {elapsed:.2f} seconds: {e}"
            )
            raise

    async def _preprocess_data(
        self, symbol: str, model_type: str, phase: str, df: pd.DataFrame
    ):
        """
        Preprocess data with monitoring config timeout settings.
        """
        start_time = time.time()
        try:
            # Convert the dates into strings (JSON serializable)
            df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

            # URL to the endpoint to preprocess the data
            url = f"http://{config.data_processing_service.HOST}:{config.data_processing_service.PORT}/processing/preprocess"

            # Define the payload
            payload = {
                "data": df.where(pd.notnull(df), None).to_dict(orient="records"),
            }

            # Define the query parameters
            params = {"symbol": symbol, "model_type": model_type, "phase": phase}

            # Use monitoring config timeout settings
            timeout = httpx.Timeout(
                connect=config.monitoring.CONNECT_TIMEOUT,
                read=config.monitoring.PREPROCESSING_TIMEOUT,
                write=config.monitoring.CONNECT_TIMEOUT,
                pool=config.monitoring.CONNECT_TIMEOUT,
            )

            async with httpx.AsyncClient(timeout=timeout) as client:
                # Send POST request to FastAPI endpoint
                response = await client.post(url, params=params, json=payload)

                # Check if the response is successful
                response.raise_for_status()

                # Retrieve the data from the json response
                data = response.json()["data"]
                preprocessed_features = (
                    data["train"]["X"] if phase == "training" else data["X"]
                )

                if isinstance(preprocessed_features, Dict):
                    preprocessed_features = pd.DataFrame(preprocessed_features)
                else:
                    preprocessed_features = np.array(preprocessed_features)

                elapsed = time.time() - start_time
                self.logger.info(
                    f"Preprocessed data for {symbol} ({phase}) in {elapsed:.2f} seconds"
                )

                return preprocessed_features

        except Exception as e:
            elapsed = time.time() - start_time
            self.logger.error(
                f"Error preprocessing data for {symbol} after {elapsed:.2f} seconds: {e}"
            )
            raise

    async def _run_evaluation_pipeline(self, model_type, symbol):
        """
        Run the full evaluation pipeline for the specified model and symbol.

        Args:
            model_type (str): The type of model to be used (e.g., 'LSTM', 'Prophet').
            symbol (str): The stock symbol for which the model is being evaluated.

        Returns:
            metrics: The result of the evaluation pipeline execution.
        """
        try:
            # URL to the endpoint to train the model
            url = f"http://{config.orchestration_service.HOST}:{config.orchestration_service.PORT}/orchestration/evaluate"

            async with httpx.AsyncClient(timeout=None) as client:
                # Define the query parameters
                params = {
                    "symbol": symbol,
                    "model_type": model_type,
                }

                # Send POST request to FastAPI endpoint
                response = await client.post(url, params=params)

                # Check if the response is successful
                response.raise_for_status()

                # Return the response as is
                return response.json()
        except Exception as e:
            self.logger.error(
                f"Error running evaluation pipeline for model {model_type} and symbol {symbol} : {e}"
            )

    async def _run_training_pipeline(self, model_type, symbol):
        """
        Run the full training pipeline for the specified model and symbol.

        Args:
            model_type (str): The type of model to be used (e.g., 'LSTM', 'Prophet').
            symbol (str): The stock symbol for which the model is being trained.

        Returns:
            result: The result of the training pipeline execution.
        """
        try:
            # URL to the endpoint to train the model
            url = f"http://{config.orchestration_service.HOST}:{config.orchestration_service.PORT}/orchestration/train"

            async with httpx.AsyncClient(timeout=None) as client:
                # Define the query parameters
                params = {
                    "symbol": symbol,
                    "model_type": model_type,
                }

                # Send POST request to FastAPI endpoint
                response = await client.post(url, params=params)

                # Check if the response is successful
                response.raise_for_status()

                # Return the response as is
                return response.json()
        except Exception as e:
            self.logger.error(
                f"Error running training pipeline for model {model_type} and symbol {symbol} : {e}"
            )
            raise

    async def _run_data_drift_check(self) -> None:
        """
        Run data drift check with monitoring config intervals.
        """
        self.logger.info("Starting weekly data drift check")
        try:
            # URL to the endpoint to preprocess the data
            url = f"http://{config.deployment_service.HOST}:{config.deployment_service.PORT}/deployment/models"
            async with httpx.AsyncClient(timeout=None) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                payload = resp.json()

            # Parse with Pydantic to get .models list
            mlflow_list = ModelListMlflowResponse(**payload)
            live_models = mlflow_list.models

            self.logger.debug(f"Found {len(live_models)} live models: {live_models}")
            for model in live_models:
                full_name = model.name
                if not full_name or "_" not in full_name:
                    self.logger.warning(f"Skipping unexpected model entry: {model}")
                    continue

                # now split on the first underscore
                model_type, symbol = full_name.split("_", 1)
                symbol = symbol.upper()

                await self._check_data_drift(model_type, symbol)
        except Exception as e:
            self.logger.error(f"Error in data drift check: {e}", exc_info=True)
        finally:
            self.logger.info("✅ Weekly data drift check complete")

    async def _check_data_drift(self, model_type: str, symbol: str) -> None:
        """
        Check data drift with monitoring config settings.
        """
        self.logger.debug(f"Starting data drift check for {model_type}_{symbol}")
        start_time = time.time()

        try:
            # Load data windows with monitoring config timeouts
            days_back = config.data.LOOKBACK_PERIOD_DAYS

            # Fetch training data
            self.logger.info(f"Fetching training data for {symbol}")
            train_df = await self._fetch_stock_data(symbol, days_back)

            # Evaluation window must cover at least the model's sequence length
            seq_len = getattr(config.model, "SEQUENCE_LENGTH", 60)
            days_needed = max(seq_len, days_back)

            # Fetch recent data
            self.logger.info(f"Fetching recent data for {symbol}")
            recent_df = await self._fetch_stock_data(symbol, days_needed)

            # Skip if no full sequence available
            if len(recent_df) < seq_len:
                self.logger.warning(
                    f"Insufficient recent data ({len(recent_df)}) for sequence, skip"
                )
                return

            # Preprocess with monitoring config settings
            self.logger.info(f"Preprocessing training data for {symbol}")
            train_proc = await self._preprocess_data(
                symbol, model_type, "training", train_df
            )

            self.logger.info(f"Preprocessing recent data for {symbol}")
            recent_proc = await self._preprocess_data(
                symbol, model_type, "evaluation", recent_df
            )

            # Extract features
            t_arr = np.asarray(getattr(train_proc, "X", []))
            r_arr = np.asarray(getattr(recent_proc, "X", []))
            if t_arr.size == 0 or r_arr.size == 0:
                self.logger.warning("No features, skip drift")
                return

            # Flatten & numeric
            # meaning turn them into a simple 1D list and keep only the numbers and Convert everything to floats, because compare two sets of numbers which only works on clean numeric data
            t_vals = np.array(
                [float(v) for v in t_arr.ravel() if isinstance(v, (int, float))]
            )
            r_vals = np.array(
                [float(v) for v in r_arr.ravel() if isinstance(v, (int, float))]
            )
            if t_vals.size == 0 or r_vals.size == 0:
                self.logger.warning("No numeric data, skip drift")
                return

            # Subsample for reliability (if too many numberse, randomly pick smaller, equally from each set (but no dupes))
            n = min(len(t_vals), len(r_vals), self.max_drift_samples)
            if len(t_vals) > n:
                t_vals = np.random.choice(t_vals, n, replace=False)
            if len(r_vals) > n:
                r_vals = np.random.choice(r_vals, n, replace=False)

            # Compute KS statistic
            stat, _ = ks_2samp(t_vals, r_vals)

            # Update rolling history & compute dynamic threshold (key is model_type,symbol)
            hist = self._drift_history.setdefault(
                (model_type, symbol), deque(maxlen=self._drift_history_length)
            )
            hist.append(stat)

            # Compute dynamic threshold
            if len(hist) >= 3:
                m = np.mean(hist)
                s = np.std(hist)
                threshold = m + self._drift_zscore * s
                self.logger.debug(
                    f"Dynamic threshold for {symbol}: {threshold:.4f} (mean={m:.4f},std={s:.4f})"
                )
            else:
                threshold = m if (hist and (m := np.mean(hist))) else stat * 1.0
            self.logger.info(f"KS stat={stat:.4f}, threshold={threshold:.4f}")

            # Decision and retraining if necessary
            if stat > threshold:
                self.logger.info(
                    f"Drift detected for {model_type}_{symbol}, retraining"
                )
                await self._run_training_pipeline(model_type, symbol)
            else:
                self.logger.debug(f"No drift for {model_type}_{symbol}")

            elapsed = time.time() - start_time
            self.logger.info(
                f"Completed data drift check for {model_type}_{symbol} in {elapsed:.2f} seconds"
            )

        except Exception as e:
            elapsed = time.time() - start_time
            self.logger.error(
                f"Error in data drift for {model_type}_{symbol} after {elapsed:.2f} seconds: {e}",
                exc_info=True,
            )
