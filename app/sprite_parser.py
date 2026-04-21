"""
Sprite Parser — internalized from frame-server for standalone deployment.
Parse WebVTT sprite files and extract tiles from sprite grids.
"""

import re
import httpx
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)


class SpriteParser:
    """Parse sprite sheets and VTT files"""

    def __init__(self, stash_api_key: str = ""):
        self.temp_dir = Path("/tmp/sprites")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self._headers = {"ApiKey": stash_api_key} if stash_api_key else {}

    async def download_file(self, url: str, output_path: Path) -> bool:
        """
        Download file from URL

        Args:
            url: File URL
            output_path: Local path to save file

        Returns:
            True if successful, False otherwise
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=30.0, headers=self._headers)
                response.raise_for_status()

                with open(output_path, 'wb') as f:
                    f.write(response.content)

                logger.info(f"Downloaded {url} to {output_path}")
                return True

        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return False

    def parse_vtt(self, vtt_content: str) -> List[Tuple[float, float, int, int, int, int]]:
        """
        Parse WebVTT sprite file

        Args:
            vtt_content: VTT file content

        Returns:
            List of (start_time, end_time, x, y, width, height)
        """
        sprites = []

        try:
            # VTT format:
            # WEBVTT
            #
            # 00:00:01.000 --> 00:00:02.000
            # sprite.jpg#xywh=0,0,160,90

            # Split into cue blocks
            blocks = vtt_content.strip().split('\n\n')

            for block in blocks:
                lines = block.strip().split('\n')

                # Skip header and empty blocks
                if len(lines) < 2 or lines[0].startswith('WEBVTT'):
                    continue

                # Parse timestamp line
                timestamp_line = lines[0]
                timestamp_match = re.match(
                    r'(\d{2}):(\d{2}):(\d{2}\.\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}\.\d{3})',
                    timestamp_line
                )

                if not timestamp_match:
                    continue

                # Convert timestamps to seconds
                start_h, start_m, start_s, end_h, end_m, end_s = timestamp_match.groups()
                start_time = int(start_h) * 3600 + int(start_m) * 60 + float(start_s)
                end_time = int(end_h) * 3600 + int(end_m) * 60 + float(end_s)

                # Parse coordinate line
                coord_line = lines[1]
                coord_match = re.search(r'#xywh=(\d+),(\d+),(\d+),(\d+)', coord_line)

                if not coord_match:
                    continue

                x, y, w, h = map(int, coord_match.groups())

                sprites.append((start_time, end_time, x, y, w, h))

            logger.info(f"Parsed {len(sprites)} sprite coordinates from VTT")
            return sprites

        except Exception as e:
            logger.error(f"Error parsing VTT: {e}")
            return []

    def extract_sprite_tiles(
        self,
        sprite_image_path: Path,
        coordinates: List[Tuple[float, float, int, int, int, int]],
        job_id: str,
        output_format: str = "jpeg",
        quality: int = 95
    ) -> List[Tuple[int, float, str, int, int]]:
        """
        Extract tiles from sprite grid image

        Args:
            sprite_image_path: Path to sprite grid image
            coordinates: List of (start_time, end_time, x, y, w, h)
            job_id: Job identifier for output naming
            output_format: jpeg or png
            quality: Image quality (1-100)

        Returns:
            List of (index, timestamp, file_path, width, height)
        """
        frames = []

        try:
            # Load sprite grid
            sprite_img = cv2.imread(str(sprite_image_path))

            if sprite_img is None:
                raise ValueError(f"Failed to load sprite image: {sprite_image_path}")

            logger.info(f"Loaded sprite image: {sprite_img.shape}")

            # Create job directory
            job_dir = self.temp_dir / job_id
            job_dir.mkdir(parents=True, exist_ok=True)

            # Extract each tile
            for idx, (start_time, end_time, x, y, w, h) in enumerate(coordinates):
                # Extract tile from grid
                tile = sprite_img[y:y+h, x:x+w]

                if tile.size == 0:
                    logger.warning(f"Empty tile at index {idx}, coords ({x},{y},{w},{h})")
                    continue

                # Save tile
                ext = "jpg" if output_format == "jpeg" else "png"
                tile_path = job_dir / f"sprite_tile_{idx:06d}.{ext}"

                if output_format == "jpeg":
                    cv2.imwrite(
                        str(tile_path),
                        tile,
                        [cv2.IMWRITE_JPEG_QUALITY, quality]
                    )
                else:
                    cv2.imwrite(str(tile_path), tile)

                # Use midpoint timestamp
                timestamp = (start_time + end_time) / 2

                frames.append((idx, timestamp, str(tile_path), w, h))

                if (idx + 1) % 50 == 0:
                    logger.info(f"Extracted {idx + 1}/{len(coordinates)} sprite tiles")

            logger.info(f"Sprite extraction complete: {len(frames)} tiles")
            return frames

        except Exception as e:
            logger.error(f"Error extracting sprite tiles: {e}")
            raise

    async def process_sprites(
        self,
        sprite_vtt_url: str,
        sprite_image_url: str,
        job_id: str,
        output_format: str = "jpeg",
        quality: int = 95
    ) -> List[Tuple[int, float, str, int, int]]:
        """
        Complete sprite processing pipeline

        Args:
            sprite_vtt_url: URL to VTT file
            sprite_image_url: URL to sprite grid image
            job_id: Job identifier
            output_format: jpeg or png
            quality: Image quality (1-100)

        Returns:
            List of (index, timestamp, file_path, width, height)
        """
        try:
            # Create job directory
            job_dir = self.temp_dir / job_id
            job_dir.mkdir(parents=True, exist_ok=True)

            # Download VTT file
            vtt_path = job_dir / "sprites.vtt"
            if not await self.download_file(sprite_vtt_url, vtt_path):
                raise ValueError(f"Failed to download VTT file: {sprite_vtt_url}")

            # Download sprite image
            sprite_image_path = job_dir / "sprite_grid.jpg"
            if not await self.download_file(sprite_image_url, sprite_image_path):
                raise ValueError(f"Failed to download sprite image: {sprite_image_url}")

            # Parse VTT
            with open(vtt_path, 'r', encoding='utf-8') as f:
                vtt_content = f.read()

            coordinates = self.parse_vtt(vtt_content)

            if not coordinates:
                raise ValueError("No sprite coordinates found in VTT")

            # Extract tiles
            frames = self.extract_sprite_tiles(
                sprite_image_path,
                coordinates,
                job_id,
                output_format,
                quality
            )

            return frames

        except Exception as e:
            logger.error(f"Error processing sprites: {e}")
            raise

    def cleanup_job(self, job_id: str):
        """
        Clean up temporary files for a job

        Args:
            job_id: Job identifier
        """
        try:
            job_dir = self.temp_dir / job_id
            if job_dir.exists():
                import shutil
                shutil.rmtree(job_dir)
                logger.info(f"Cleaned up sprite files for job: {job_id}")
        except Exception as e:
            logger.error(f"Error cleaning up job {job_id}: {e}")
