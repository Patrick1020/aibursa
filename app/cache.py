import json
from datetime import datetime, timedelta, timezone
from .database import get_db
from .models import APICache
from .logger import logger

CACHE_DURATION_HOURS = 6


def get_cached_response(api_name: str, request_key: str):
    with get_db() as db:
        cached = (
            db.query(APICache)
            .filter(
                APICache.api_name == api_name,
                APICache.request_key == request_key,
                APICache.timestamp
                >= datetime.now(timezone.utc) - timedelta(hours=CACHE_DURATION_HOURS),
            )
            .first()
        )
        if cached:
            logger.info(f"Using cached response for {api_name}:{request_key}")
            return json.loads(cached.response_data)
        return None


def set_cached_response(api_name: str, request_key: str, data):
    with get_db() as db:
        cached = APICache(
            api_name=api_name,
            request_key=request_key,
            response_data=json.dumps(data),
            timestamp=datetime.now(timezone.utc),
        )
        db.add(cached)
        db.commit()
        logger.info(f"Cached response saved for {api_name}:{request_key}")
