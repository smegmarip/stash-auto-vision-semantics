"""
Redis-backed job queue with exclusive single-job processing.

Uses an atomic Lua script to enforce that only one job is active at a time.
The active lock serves dual purpose as both state (a job is running) and
identity (which job is running, on which worker).

Contract: the worker MUST call acquire_next_job() to get a job_id. No other
code path can pull from the pending queue. The Lua script guarantees that
only one acquire can succeed while a lock is held.
"""

import json
import logging
import os
import socket
import time
import uuid
from typing import Any, Dict, Optional

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

# Lua script: atomically acquire next job if no job is currently active.
# Returns the job_id (with worker_id stored in active lock) or nil.
#
# KEYS[1] = pending list, KEYS[2] = active lock key
# ARGV[1] = lock value (worker_id:job_id format set after RPOP), ARGV[2] = TTL seconds
#
# Note: ARGV[1] is constructed inside the script using the popped job_id.
ACQUIRE_SCRIPT = """
if redis.call('EXISTS', KEYS[2]) == 1 then
  return nil
end
local job_id = redis.call('RPOP', KEYS[1])
if not job_id then
  return nil
end
local lock_value = ARGV[1] .. ':' .. job_id
redis.call('SET', KEYS[2], lock_value, 'EX', tonumber(ARGV[2]))
return job_id
"""

# Lua script: release the active lock only if it belongs to us.
# Prevents accidental release of another worker's lock (e.g., after TTL expiry
# and reacquisition by a different worker).
#
# KEYS[1] = active lock key
# ARGV[1] = expected lock value prefix (worker_id:job_id)
RELEASE_SCRIPT = """
local current = redis.call('GET', KEYS[1])
if current == ARGV[1] then
  redis.call('DEL', KEYS[1])
  return 1
end
return 0
"""

# Lua script: requeue the active job back to the front of pending and clear lock.
# Used on worker startup to recover orphans from a previous crash.
#
# KEYS[1] = active lock key, KEYS[2] = pending list
# ARGV[1] = expected worker_id prefix (so we only recover OUR previous instance)
RECOVER_SCRIPT = """
local current = redis.call('GET', KEYS[1])
if not current then
  return nil
end
-- Extract job_id from "worker_id:job_id" format
local sep = string.find(current, ':')
if not sep then
  return nil
end
local locked_worker = string.sub(current, 1, sep - 1)
local locked_job = string.sub(current, sep + 1)
-- Only recover if it's our worker's previous lock
if locked_worker ~= ARGV[1] then
  return nil
end
redis.call('RPUSH', KEYS[2], locked_job)
redis.call('DEL', KEYS[1])
return locked_job
"""


class JobQueue:
    """Redis-backed job queue with atomic exclusive job acquisition.

    The queue maintains:
    - pending list: FIFO of job_ids waiting to be processed
    - active lock: single key holding "{worker_id}:{job_id}" while a job runs
    - job request hash: full request payload, source of truth for processing

    The active lock is the resource permit. Only one job can hold it at a time,
    enforced by a Redis Lua script.
    """

    def __init__(
        self,
        redis: aioredis.Redis,
        module: str = "semantics",
        worker_id: Optional[str] = None,
        lock_ttl_seconds: int = 3600,
    ):
        self.redis = redis
        self.module = module
        # Default worker_id to hostname so it's stable across container restarts.
        # Falls back to a random id only if hostname is unavailable.
        self.worker_id = (
            worker_id
            or os.getenv("SEMANTICS_WORKER_ID")
            or socket.gethostname()
            or f"worker-{uuid.uuid4().hex[:8]}"
        )
        self.lock_ttl = lock_ttl_seconds

        # Key names
        self.pending_key = f"{module}:queue:pending"
        self.active_key = f"{module}:queue:active"

        # Pre-loaded scripts (Redis caches by SHA after first call)
        self._acquire_script = self.redis.register_script(ACQUIRE_SCRIPT)
        self._release_script = self.redis.register_script(RELEASE_SCRIPT)
        self._recover_script = self.redis.register_script(RECOVER_SCRIPT)

    # ----- Job request payload -----

    def _request_key(self, job_id: str) -> str:
        return f"{self.module}:job:{job_id}:request"

    async def store_request(self, job_id: str, payload: Dict[str, Any], ttl: int = 86400) -> None:
        """Store the full request payload for a job. TTL defaults to 24h."""
        await self.redis.set(self._request_key(job_id), json.dumps(payload), ex=ttl)

    async def load_request(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Load a stored request payload by job_id."""
        raw = await self.redis.get(self._request_key(job_id))
        if not raw:
            return None
        return json.loads(raw)

    # ----- Queue operations -----

    async def enqueue(self, job_id: str) -> int:
        """Push a job to the pending queue. Returns new queue length."""
        return await self.redis.lpush(self.pending_key, job_id)

    async def acquire_next_job(self) -> Optional[str]:
        """Atomically acquire the next job from pending if no job is active.

        Returns the job_id and sets the active lock to "{worker_id}:{job_id}".
        Returns None if a job is already active or the queue is empty.
        """
        result = await self._acquire_script(
            keys=[self.pending_key, self.active_key],
            args=[self.worker_id, str(self.lock_ttl)],
        )
        return result if result else None

    async def release_job(self, job_id: str) -> bool:
        """Atomically release the active lock if it belongs to us.

        Returns True if released, False if the lock was held by someone else
        (e.g., it expired and was reacquired) — indicates a logic error or
        a worker that ran longer than the TTL.
        """
        lock_value = f"{self.worker_id}:{job_id}"
        result = await self._release_script(
            keys=[self.active_key],
            args=[lock_value],
        )
        return bool(result)

    async def recover_orphan(self) -> Optional[str]:
        """On startup, requeue any in-flight job from a previous instance of this worker.

        Only recovers jobs locked by OUR worker_id (set via SEMANTICS_WORKER_ID
        or defaulting to hostname). Returns the recovered job_id or None.
        """
        result = await self._recover_script(
            keys=[self.active_key, self.pending_key],
            args=[self.worker_id],
        )
        return result if result else None

    async def force_recover_any_orphan(self) -> Optional[str]:
        """Force-recover an orphan locked by ANY worker_id.

        Intended for single-worker deployments where a restart may have
        changed the worker_id (e.g., transition from random id to hostname).
        Requeues the locked job to the front of pending and clears the lock.
        Returns the recovered job_id or None.
        """
        raw = await self.redis.get(self.active_key)
        if not raw:
            return None
        # Extract job_id from "worker_id:job_id" format
        if ":" in raw:
            _, job_id = raw.split(":", 1)
        else:
            job_id = raw
        # Atomically requeue and clear (best effort — race with a live worker
        # that holds the lock would be prevented by the release script, but
        # we're doing this on startup before any worker loop runs)
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.rpush(self.pending_key, job_id)
            pipe.delete(self.active_key)
            await pipe.execute()
        return job_id

    # ----- Inspection -----

    async def pending_count(self) -> int:
        """Return the number of jobs waiting in the pending queue."""
        return await self.redis.llen(self.pending_key)

    async def active_job(self) -> Optional[Dict[str, str]]:
        """Return the currently active job and worker, or None if idle."""
        raw = await self.redis.get(self.active_key)
        if not raw:
            return None
        if ":" in raw:
            worker_id, job_id = raw.split(":", 1)
            return {"worker_id": worker_id, "job_id": job_id}
        return {"worker_id": "unknown", "job_id": raw}

    async def queue_stats(self) -> Dict[str, Any]:
        """Return queue inspection data for the health endpoint."""
        return {
            "worker_id": self.worker_id,
            "pending": await self.pending_count(),
            "active": await self.active_job(),
        }
