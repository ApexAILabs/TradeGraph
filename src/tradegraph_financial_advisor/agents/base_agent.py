from abc import ABC, abstractmethod
from typing import Any, Dict
from datetime import datetime
import asyncio
from loguru import logger

from ..config.settings import settings


class BaseAgent(ABC):
    def __init__(self, name: str, description: str, **kwargs):
        self.name = name
        self.description = description
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.is_active = False
        self.config = kwargs

    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        pass

    async def start(self) -> None:
        self.is_active = True
        self.last_activity = datetime.now()
        logger.info(f"Agent {self.name} started")

    async def stop(self) -> None:
        self.is_active = False
        logger.info(f"Agent {self.name} stopped")

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
        }

    async def health_check(self) -> bool:
        try:
            await asyncio.wait_for(
                self._health_check_impl(), timeout=settings.analysis_timeout_seconds
            )
            return True
        except asyncio.TimeoutError:
            logger.warning(f"Health check timeout for agent {self.name}")
            return False
        except Exception as e:
            logger.error(f"Health check failed for agent {self.name}: {str(e)}")
            return False

    async def _health_check_impl(self) -> None:
        pass
