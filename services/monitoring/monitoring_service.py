import asyncio
from collections import deque
import contextlib
from typing import Dict, Tuple

import numpy as np
from scipy.stats import ks_2samp

from core import BaseService
from core.logging import logger
from core.config import config
from core.prometheus_metrics import evaluation_mae
from services import DeploymentService, DataProcessingService, DataService
from services.orchestration import OrchestrationService


class MonitoringService(BaseService):
    def __init__(
        self,
        deployment_service: DeploymentService,
        orchestration_service: OrchestrationService,
        data_service: DataService,
        preprocessing_service: DataProcessingService,
        check_interval_seconds: int = 24 * 60 * 60,  # default: once per day
        data_interval_seconds: int = 7 * 24 * 60 * 60,  # once per week
        max_drift_samples: int = 500,
        drift_stat_threshold: float = 0.1,
        data_drift_history_length: int = 8,
        data_drift_zscore: float = 2.0,
    ):
        super().__init__()
        self.logger = logger["monitoring"]
        self.deployment_service = deployment_service
        self.orchestration_service = orchestration_service
        self.data_service = data_service
        self.preprocessing_service = preprocessing_service
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
        """Run daily performance drift checks."""
        try:
            while self._initialized:
                await self._run_performance_check()
                await asyncio.sleep(self.check_interval_seconds)
        except asyncio.CancelledError:
            pass

    async def _data_loop(self) -> None:
        """Run weekly data drift checks."""
        try:
            while self._initialized:
                await self._run_data_drift_check()
                await asyncio.sleep(self.data_interval)
        except asyncio.CancelledError:
            pass

    async def _run_performance_check(self) -> None:
        """
        1. List all live models (e.g. 'lstm_AAPL')
        2. For each: run evaluation, read MAE, compare to last run, retrain if worse.
        """
        self.logger.info("Starting daily drift check")
        try:
            live_models = await self.deployment_service.list_models()
            for model in live_models:
                # grab the name string from the dict
                full_name = model.get("name")
                if not full_name or "_" not in full_name:
                    self.logger.warning(f"Skipping unexpected model entry: {model}")
                    continue

                # now split on the first underscore
                model_type, symbol = full_name.split("_", 1)
                symbol = symbol.upper()

                # 1) Run evaluation pipeline (updates the evaluation_mae gauge)
                result = await self.orchestration_service.run_evaluation_pipeline(
                    model_type, symbol
                )

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
                    await self.orchestration_service.run_training_pipeline(
                        model_type, symbol
                    )
                else:
                    self.logger.info(
                        f"No retraining needed for {model}, prev MAE: {prev} vs current MAE: {current}"
                    )
        except Exception as e:
            self.logger.error(f"Error in daily drift check: {e}")
        finally:
            self.logger.info("🏁 Daily drift check complete")

    async def _run_data_drift_check(self) -> None:
        """
        For each live model, fetch recent and historical data,
        run KS-test on 'Close' distributions weekly, retrain on drift.
        """
        self.logger.info("Starting weekly data drift check")
        try:
            models = await self.deployment_service.list_models()
            self.logger.debug(f"Found {len(models)} live models: {models}")
            for model in models:
                full_name = model.get("name")
                if not full_name or "_" not in full_name:
                    self.logger.warning(f"Skipping unexpected model entry: {model}")
                    continue

                # now split on the first underscore
                model_type, symbol = full_name.split("_", 1)
                symbol = symbol.upper()

                await self._check_data_drift(model_type, symbol)
        except Exception as e:
            self.logger.error(f"Error in weekly data drift check: {e}", exc_info=True)
        finally:
            self.logger.info("✅ Weekly data drift check complete")

    async def _check_data_drift(self, model_type: str, symbol: str) -> None:
        """
        Fetch training and recent data, preprocess&filter both, then run KS-test on the 2 distributions.
        Maintain a rolling history of KS stats to compute a dynamic threshold = mean + zscore * std.
        Retrain only if the current KS stat exceeds dynamic threshold.
        """
        self.logger.debug(f"Starting data drift for {model_type}_{symbol}")
        try:
            # Load data windows
            days_back = config.data.LOOKBACK_PERIOD_DAYS
            train_df, _ = await self.data_service.get_recent_data(
                symbol, days_back=days_back
            )
            # Evaluation window must cover at least the model's sequence length
            seq_len = getattr(config.model, "SEQUENCE_LENGTH", 60)
            days_needed = max(seq_len, days_back)
            recent_df, _ = await self.data_service.get_recent_data(
                symbol, days_back=days_needed
            )
            # Skip if no full sequence available
            if len(recent_df) < seq_len:
                self.logger.warning(
                    f"Insufficient recent data ({len(recent_df)}) for sequence, skip"
                )
                return

            # Preprocess
            # Perform training-phase preprocessing to get baseline features
            train_res = await self.preprocessing_service.preprocess_data(
                symbol, train_df, model_type, phase="training"
            )
            train_proc = train_res[0] if isinstance(train_res, tuple) else train_res
            # Perform evaluation-phase preprocessing to get current features?
            recent_res = await self.preprocessing_service.preprocess_data(
                symbol, recent_df, model_type, phase="evaluation"
            )
            recent_proc = recent_res[0] if isinstance(recent_res, tuple) else recent_res

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
                await self.orchestration_service.run_training_pipeline(
                    model_type, symbol
                )
            else:
                self.logger.debug(f"No drift for {model_type}_{symbol}")
        except Exception as e:
            self.logger.error(
                f"Error in data drift for {model_type}_{symbol}: {e}", exc_info=True
            )
