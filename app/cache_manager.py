"""
Semantics Service - Cache Manager
Redis-based caching with content-based keys
"""

import hashlib
import json
import os
from typing import Optional, Dict, Any
import redis.asyncio as aioredis
import logging

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages Redis caching for semantics analysis jobs"""

    def __init__(self, redis_url: str, module: str = "semantics", ttl: int = 3600):
        """
        Initialize cache manager

        Args:
            redis_url: Redis connection string
            module: Module name for key namespacing
            ttl: Default TTL in seconds
        """
        self.redis_url = redis_url
        self.module = module
        self.ttl = ttl
        self.redis: Optional[aioredis.Redis] = None

    async def connect(self):
        """Establish Redis connection"""
        if not self.redis:
            self.redis = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            logger.info(f"Connected to Redis: {self.redis_url}")

    async def disconnect(self):
        """Close Redis connection"""
        if self.redis:
            await self.redis.close()
            self.redis = None
            logger.info("Disconnected from Redis")

    def generate_cache_key(self, video_path: str, params: Dict[str, Any]) -> str:
        """
        Generate deterministic cache key from video + parameters

        Args:
            video_path: Absolute path to video file
            params: Analysis parameters

        Returns:
            SHA-256 hash (hex string)
        """
        try:
            # Get file modification time (invalidates cache if video changes)
            if os.path.exists(video_path):
                video_mtime = os.path.getmtime(video_path)
            else:
                logger.warning(f"Video not found for cache key: {video_path}")
                video_mtime = 0

            # Sort params keys for deterministic JSON
            params_str = json.dumps(params, sort_keys=True)

            # Combine into cache key string
            cache_str = f"{video_path}:{video_mtime}:{self.module}:{params_str}"

            # Hash to fixed-length key
            cache_key = hashlib.sha256(cache_str.encode()).hexdigest()

            return cache_key
        except Exception as e:
            logger.error(f"Error generating cache key: {e}")
            # Fallback to simpler key without mtime
            fallback_str = f"{video_path}:{self.module}:{json.dumps(params, sort_keys=True)}"
            return hashlib.sha256(fallback_str.encode()).hexdigest()

    async def get_cached_job_id(self, cache_key: str) -> Optional[str]:
        """
        Check if results exist for this cache key

        Args:
            cache_key: Content-based cache key

        Returns:
            Job ID if cached, None otherwise
        """
        if not self.redis:
            await self.connect()

        try:
            mapping_key = f"{self.module}:cache:{cache_key}"
            job_id = await self.redis.get(mapping_key)

            if job_id:
                # Verify the job's metadata still exists; if it was cleared
                # the mapping is stale — treat as a cache miss and clean up.
                metadata_key = f"{self.module}:job:{job_id}:metadata"
                if not await self.redis.exists(metadata_key):
                    logger.warning(f"Stale cache mapping for key: {cache_key[:16]}... → job: {job_id} (metadata gone), removing")
                    await self.redis.delete(mapping_key)
                    return None
                logger.info(f"Cache hit for key: {cache_key[:16]}... → job: {job_id}")
            else:
                logger.debug(f"Cache miss for key: {cache_key[:16]}...")

            return job_id
        except Exception as e:
            logger.error(f"Error checking cache: {e}")
            return None

    async def cache_job_metadata(
        self,
        job_id: str,
        cache_key: str,
        metadata: Dict[str, Any],
        ttl: Optional[int] = None
    ):
        """
        Store job metadata

        Args:
            job_id: Job identifier
            cache_key: Content-based cache key
            metadata: Job metadata dict
            ttl: Time to live (seconds), uses default if None
        """
        if not self.redis:
            await self.connect()

        ttl = ttl or self.ttl

        try:
            metadata_key = f"{self.module}:job:{job_id}:metadata"

            # Store metadata
            await self.redis.setex(
                metadata_key,
                ttl,
                json.dumps(metadata)
            )

            logger.debug(f"Cached metadata for job: {job_id}")
        except Exception as e:
            logger.error(f"Error caching metadata: {e}")

    async def cache_job_results(
        self,
        job_id: str,
        cache_key: str,
        results: Dict[str, Any],
        ttl: Optional[int] = None
    ):
        """
        Store job results with cache mapping

        Args:
            job_id: Job identifier
            cache_key: Content-based cache key
            results: Job results dict
            ttl: Time to live (seconds), uses default if None
        """
        if not self.redis:
            await self.connect()

        ttl = ttl or self.ttl

        try:
            results_key = f"{self.module}:job:{job_id}:results"
            mapping_key = f"{self.module}:cache:{cache_key}"

            # Store results
            await self.redis.setex(
                results_key,
                ttl,
                json.dumps(results)
            )

            # Create cache mapping (bidirectional lookup)
            await self.redis.setex(
                mapping_key,
                ttl,
                job_id
            )

            logger.info(f"Cached results for job: {job_id} (key: {cache_key[:16]}...)")
        except Exception as e:
            logger.error(f"Error caching results: {e}")

    async def create_job_alias(self, alias_job_id: str, target_job_id: str, ttl: Optional[int] = None):
        """
        Create an alias from one job_id to another (for cache hit scenarios).

        Args:
            alias_job_id: The job_id to create an alias for
            target_job_id: The actual job_id that has the cached data
            ttl: Time to live (seconds), uses default if None
        """
        if not self.redis:
            await self.connect()

        ttl = ttl or self.ttl

        try:
            alias_key = f"{self.module}:job:{alias_job_id}:alias"
            await self.redis.setex(alias_key, ttl, target_job_id)
            logger.debug(f"Created job alias: {alias_job_id} → {target_job_id}")
        except Exception as e:
            logger.error(f"Error creating job alias: {e}")

    async def resolve_job_alias(self, job_id: str) -> str:
        """
        Resolve job alias to actual job_id, or return original if no alias.

        Args:
            job_id: Job identifier (may be an alias)

        Returns:
            The actual job_id (resolved if alias, original if not)
        """
        if not self.redis:
            await self.connect()

        try:
            alias_key = f"{self.module}:job:{job_id}:alias"
            target = await self.redis.get(alias_key)
            if target:
                logger.debug(f"Resolved job alias: {job_id} → {target}")
                return target
            return job_id
        except Exception as e:
            logger.error(f"Error resolving job alias: {e}")
            return job_id

    async def get_job_metadata(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve job metadata by job ID

        Args:
            job_id: Job identifier (may be an alias)

        Returns:
            Metadata dict if found, None otherwise
        """
        if not self.redis:
            await self.connect()

        try:
            # Resolve alias first
            resolved_id = await self.resolve_job_alias(job_id)
            metadata_key = f"{self.module}:job:{resolved_id}:metadata"
            metadata_json = await self.redis.get(metadata_key)

            if metadata_json:
                return json.loads(metadata_json)
            return None
        except Exception as e:
            logger.error(f"Error retrieving metadata: {e}")
            return None

    async def get_job_results(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve job results by job ID

        Args:
            job_id: Job identifier (may be an alias)

        Returns:
            Results dict if found, None otherwise
        """
        if not self.redis:
            await self.connect()

        try:
            # Resolve alias first
            resolved_id = await self.resolve_job_alias(job_id)
            results_key = f"{self.module}:job:{resolved_id}:results"
            results_json = await self.redis.get(results_key)

            if results_json:
                return json.loads(results_json)
            return None
        except Exception as e:
            logger.error(f"Error retrieving results: {e}")
            return None

    async def update_job_status(
        self,
        job_id: str,
        status: str,
        progress: float = 0.0,
        stage: Optional[str] = None,
        message: Optional[str] = None,
        error: Optional[str] = None
    ):
        """
        Update job status in cache

        Args:
            job_id: Job identifier
            status: Job status (queued, processing, completed, failed)
            progress: Progress fraction (0.0-1.0)
            stage: Current processing stage
            message: Status message
            error: Error message if failed
        """
        if not self.redis:
            await self.connect()

        try:
            metadata = await self.get_job_metadata(job_id)
            if metadata:
                metadata['status'] = status
                metadata['progress'] = progress
                if stage:
                    metadata['stage'] = stage
                if message:
                    metadata['message'] = message
                if error:
                    metadata['error'] = error

                await self.cache_job_metadata(job_id, metadata.get('cache_key', ''), metadata)
                logger.debug(f"Updated status for job {job_id}: {status} ({progress:.0%})")
        except Exception as e:
            logger.error(f"Error updating job status: {e}")
