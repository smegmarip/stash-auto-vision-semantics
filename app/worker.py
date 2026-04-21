"""
Semantics worker — single async task that processes jobs from the Redis queue.

The worker is the only consumer of the job queue. It acquires jobs via the
JobQueue's atomic Lua script (which serves as the resource permit) and runs
the pipeline. Only one job runs at a time, enforced by Redis.
"""

import asyncio
import logging
from typing import Awaitable, Callable, Optional

from .job_queue import JobQueue

logger = logging.getLogger(__name__)


class SemanticsWorker:
    """Long-running async task that pulls jobs from the queue and processes them.

    The worker contract:
    - Calls JobQueue.acquire_next_job() — atomic, exclusive
    - Runs the pipeline only between acquire and release
    - Always releases the lock in finally
    - Only one job in flight at any time, enforced by Redis Lua script
    """

    def __init__(
        self,
        queue: JobQueue,
        process_fn: Callable[[str, dict], Awaitable[None]],
        poll_interval: float = 1.0,
    ):
        self.queue = queue
        self.process_fn = process_fn
        self.poll_interval = poll_interval
        self._task: Optional[asyncio.Task] = None
        self._shutdown = asyncio.Event()

    async def start(self) -> None:
        """Recover any orphan from a previous run, then start the main loop.

        Recovery strategy (single-worker deployment):
        1. Try to recover a job locked by OUR worker_id (clean restart with stable id)
        2. If that fails and there's still a stale lock, force-recover it
           (e.g., previous instance used a different worker_id)
        """
        recovered = await self.queue.recover_orphan()
        if recovered:
            logger.warning(f"Recovered orphan job {recovered} from previous instance (matching worker_id)")
        else:
            # Check for a stale lock from a previous instance with a different worker_id
            active = await self.queue.active_job()
            if active:
                stale_id = active.get("worker_id", "unknown")
                stale_job = active.get("job_id")
                logger.warning(
                    f"Found stale active lock held by worker_id={stale_id} "
                    f"(our worker_id={self.queue.worker_id}). Force-recovering job {stale_job}."
                )
                forced = await self.queue.force_recover_any_orphan()
                if forced:
                    logger.warning(f"Force-recovered orphan job {forced}")

        self._shutdown.clear()
        self._task = asyncio.create_task(self._run_loop(), name="semantics-worker")
        logger.info(f"Worker started (worker_id={self.queue.worker_id})")

    async def stop(self) -> None:
        """Signal shutdown and wait for the loop to exit."""
        self._shutdown.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=10)
            except asyncio.TimeoutError:
                logger.warning("Worker did not stop within 10s, cancelling")
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
        logger.info("Worker stopped")

    async def _run_loop(self) -> None:
        """Main worker loop. Runs until shutdown is signalled."""
        logger.info("Worker loop started")
        while not self._shutdown.is_set():
            try:
                job_id = await self.queue.acquire_next_job()
            except Exception as e:
                logger.error(f"Failed to acquire job: {e}", exc_info=True)
                await asyncio.sleep(self.poll_interval)
                continue

            if not job_id:
                # Either no pending jobs, or another worker holds the lock.
                # Sleep briefly and try again.
                try:
                    await asyncio.wait_for(self._shutdown.wait(), timeout=self.poll_interval)
                except asyncio.TimeoutError:
                    pass
                continue

            logger.info(f"Acquired job {job_id}")
            try:
                request_payload = await self.queue.load_request(job_id)
                if not request_payload:
                    logger.error(f"Job {job_id} has no stored request payload, skipping")
                    continue
                await self.process_fn(job_id, request_payload)
            except Exception as e:
                logger.error(f"Job {job_id} processing failed: {e}", exc_info=True)
            finally:
                released = await self.queue.release_job(job_id)
                if not released:
                    logger.warning(f"Job {job_id} lock was not held by us at release time")
                logger.info(f"Released job {job_id}")
        logger.info("Worker loop exiting")
