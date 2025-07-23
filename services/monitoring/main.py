import os
import asyncio
from monitoring_service import MonitoringService

import httpx

class APIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=base_url)

    async def list_models(self) -> list[str]:
        resp = await self.client.get("/api/models")
        resp.raise_for_status()
        payload = resp.json()
        return payload.get("models", [])

    async def run_evaluation(self, model_type: str, symbol: str) -> dict:
        resp = await self.client.post(
            "/api/train/evaluate",
            json={"model_type": model_type, "symbol": symbol}
        )
        resp.raise_for_status()
        return resp.json()

    async def run_training(self, model_type: str, symbol: str) -> dict:
        resp = await self.client.post(
            "/api/train/train",
            json={"model_type": model_type, "symbol": symbol}
        )
        resp.raise_for_status()
        return resp.json()

    async def get_recent_data(self, symbol: str, days_back: int) -> list[dict]:
        resp = await self.client.get(
            "/api/data/stock/historical",
            params={"symbol": symbol, "days_back": days_back}
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload.get("data", [])

    async def preprocess(self, symbol: str, data: list[dict], model_type: str, phase: str) -> dict:
        resp = await self.client.post(
            "/api/data/process",
            json={"symbol": symbol, "data": data, "model_type": model_type, "phase": phase}
        )
        resp.raise_for_status()
        return resp.json()


async def main():
    # Base URL of API gateway
    API_URL = os.getenv("API_GATEWAY_URL", "http://api-gateway:8000")

    client = APIClient(API_URL)

    # Instantiate MonitoringService with HTTP-backed clients
    monitoring_service = MonitoringService(
        deployment_service=client,
        orchestration_service=client,
        data_service=client,
        preprocessing_service=client,
        check_interval_seconds=int(os.getenv("CHECK_INTERVAL_SECONDS", 86400)),
        data_interval_seconds=int(os.getenv("DATA_INTERVAL_SECONDS", 604800)),
    )

    # Initialize and start background loops
    await monitoring_service.initialize()
    try:
        # Keep running until killed
        while True:
            await asyncio.sleep(60)
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        # Clean up tasks
        await monitoring_service.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
