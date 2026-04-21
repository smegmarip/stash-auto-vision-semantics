"""
Taxonomy Builder - Build classifier taxonomy from Stash GraphQL API or raw tag data.

Converts Stash findTags output (nested parent references) into the flat taxonomy
structure expected by the tag classifier training pipeline. Handles DAG topologies
where tags may have multiple parents, producing multiple paths per tag.

Output matches the taxonomy.json format:
    metadata  - summary stats (total_tags, leaf_tags, max_depth, root_tag_id)
    tags      - list of enriched tag dicts with paths, depths, parent info
    by_id     - lookup dict keyed by tag ID
    by_name   - lookup dict keyed by tag name
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# GraphQL query that fetches all tags with nested parent chain (5 levels deep)
# and immediate children. Stash returns parents recursively; we walk the chain
# to reconstruct the full path from root.
FIND_TAGS_QUERY = """
query FindTags($filter: FindFilterType, $tag_filter: TagFilterType) {
  findTags(filter: $filter, tag_filter: $tag_filter) {
    count
    tags {
      id
      name
      description
      aliases
      parents {
        id
        name
        parents {
          id
          name
          parents {
            id
            name
            parents {
              id
              name
              parents {
                id
                name
              }
            }
          }
        }
      }
      children {
        id
        name
      }
    }
  }
}
""".strip()

# Page size for Stash GraphQL pagination
_PAGE_SIZE = 500


class TaxonomyBuilder:
    """Builds a classifier-ready taxonomy dict from Stash tag data."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    async def build_from_stash(
        stash_url: str,
        stash_api_key: str = "",
        root_tag_id: str | None = None,
    ) -> dict[str, Any]:
        """Fetch tags from Stash GraphQL API and build taxonomy.

        Args:
            stash_url: Base URL of the Stash instance (e.g. ``http://stash:9999``).
            stash_api_key: Optional API key for authentication.
            root_tag_id: If set, only include tags that descend from this root.
                The root tag itself is excluded from the output.

        Returns:
            Taxonomy dict matching the ``taxonomy.json`` schema.

        Raises:
            httpx.HTTPStatusError: On non-2xx responses from Stash.
            RuntimeError: If the GraphQL response contains errors or no data.
        """
        raw_tags = await TaxonomyBuilder._fetch_all_tags(stash_url, stash_api_key)
        logger.info("Fetched %d tags from Stash at %s", len(raw_tags), stash_url)

        if root_tag_id is not None:
            raw_tags = TaxonomyBuilder._filter_descendants(raw_tags, root_tag_id)
            logger.info("Filtered to %d tags under root %s", len(raw_tags), root_tag_id)

        return TaxonomyBuilder.build_from_tags(raw_tags, root_tag_id=root_tag_id)

    @staticmethod
    def build_from_tags(
        tags: list[dict[str, Any]],
        root_tag_id: str | None = None,
    ) -> dict[str, Any]:
        """Build taxonomy from a raw ``findTags.tags`` array.

        Args:
            tags: List of tag dicts as returned by the Stash ``findTags`` query.
                Each dict must have at minimum ``id``, ``name``, and ``parents``.
            root_tag_id: Optional root tag ID. When provided, the root tag is
                excluded from the output and paths are relative to it.

        Returns:
            Taxonomy dict matching the ``taxonomy.json`` schema.
        """
        tags_by_id, children_map = TaxonomyBuilder._build_indices(tags)

        # Auto-detect root if not provided: a tag with no parents that is an
        # ancestor of other tags.
        if root_tag_id is None:
            root_tag_id = TaxonomyBuilder._detect_root(tags_by_id)

        root_name: str | None = None
        if root_tag_id and root_tag_id in tags_by_id:
            root_name = tags_by_id[root_tag_id]["name"]

        enriched: list[dict[str, Any]] = []
        for tag in tags:
            tag_id = str(tag["id"])

            # Skip the root tag itself -- it is organizational, not a real label
            if tag_id == root_tag_id:
                continue

            entry = TaxonomyBuilder._enrich_tag(tag, tags_by_id, children_map, root_tag_id, root_name)
            enriched.append(entry)

        # Sort by depth (ascending), then name for deterministic output
        enriched.sort(key=lambda t: (t["depth"], t["name"]))

        by_id: dict[str, dict[str, Any]] = {t["id"]: t for t in enriched}
        by_name: dict[str, dict[str, Any]] = {t["name"]: t for t in enriched}

        max_depth = max((t["depth"] for t in enriched), default=0)
        leaf_count = sum(1 for t in enriched if t["is_leaf"])

        metadata: dict[str, Any] = {
            "total_tags": len(enriched),
            "leaf_tags": leaf_count,
            "branch_tags": len(enriched) - leaf_count,
            "max_depth": max_depth,
            "root_tag_id": root_tag_id or "",
        }

        logger.info(
            "Taxonomy built: %d tags (%d leaf, %d branch), max depth %d",
            metadata["total_tags"],
            metadata["leaf_tags"],
            metadata["branch_tags"],
            metadata["max_depth"],
        )

        return {
            "metadata": metadata,
            "tags": enriched,
            "by_id": by_id,
            "by_name": by_name,
        }

    # ------------------------------------------------------------------
    # GraphQL fetching
    # ------------------------------------------------------------------

    @staticmethod
    async def _fetch_all_tags(
        stash_url: str,
        stash_api_key: str,
    ) -> list[dict[str, Any]]:
        """Fetch all tags from Stash, paginating as needed.

        Returns the combined ``tags`` array across all pages.
        """
        graphql_url = f"{stash_url.rstrip('/')}/graphql"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if stash_api_key:
            headers["ApiKey"] = stash_api_key

        all_tags: list[dict[str, Any]] = []
        page = 1

        async with httpx.AsyncClient(timeout=30.0) as client:
            while True:
                variables: dict[str, Any] = {
                    "filter": {
                        "page": page,
                        "per_page": _PAGE_SIZE,
                        "sort": "name",
                        "direction": "ASC",
                    }
                }

                payload = {"query": FIND_TAGS_QUERY, "variables": variables}
                resp = await client.post(graphql_url, json=payload, headers=headers)
                resp.raise_for_status()

                body = resp.json()
                if "errors" in body and body["errors"]:
                    raise RuntimeError(f"Stash GraphQL errors: {body['errors']}")

                find_tags = body.get("data", {}).get("findTags")
                if find_tags is None:
                    raise RuntimeError("Unexpected GraphQL response: missing data.findTags")

                page_tags = find_tags.get("tags", [])
                total_count = find_tags.get("count", 0)
                all_tags.extend(page_tags)

                logger.debug(
                    "Fetched page %d: %d tags (total so far: %d / %d)",
                    page,
                    len(page_tags),
                    len(all_tags),
                    total_count,
                )

                if len(all_tags) >= total_count or len(page_tags) < _PAGE_SIZE:
                    break

                page += 1

        return all_tags

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_descendants(
        tags: list[dict[str, Any]],
        root_tag_id: str,
    ) -> list[dict[str, Any]]:
        """Keep only tags that are descendants of *root_tag_id*.

        A tag is a descendant if walking up its parent chain eventually reaches
        *root_tag_id*. The root tag itself is included in the returned list
        so that ``build_from_tags`` can reference it (it gets excluded later).
        """
        tags_by_id: dict[str, dict[str, Any]] = {str(t["id"]): t for t in tags}
        descendant_ids: set[str] = {root_tag_id}
        checked: set[str] = set()

        def _is_descendant(tag_id: str) -> bool:
            if tag_id in descendant_ids:
                return True
            if tag_id in checked:
                return False
            checked.add(tag_id)

            tag = tags_by_id.get(tag_id)
            if tag is None:
                return False

            for parent_id in TaxonomyBuilder._extract_direct_parent_ids(tag):
                if _is_descendant(parent_id):
                    descendant_ids.add(tag_id)
                    return True

            return False

        for tag in tags:
            _is_descendant(str(tag["id"]))

        return [t for t in tags if str(t["id"]) in descendant_ids]

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    @staticmethod
    def _build_indices(
        tags: list[dict[str, Any]],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
        """Build lookup dicts from raw tag list.

        Returns:
            tags_by_id: ``{tag_id: tag_dict}``
            children_map: ``{parent_id: [child_id, ...]}``
        """
        tags_by_id: dict[str, dict[str, Any]] = {}
        children_map: dict[str, list[str]] = {}

        for tag in tags:
            tag_id = str(tag["id"])
            tags_by_id[tag_id] = tag

            # Build children_map from explicit ``children`` field
            for child in tag.get("children", []):
                children_map.setdefault(tag_id, []).append(str(child["id"]))

            # Also infer children from parent references (in case children
            # field is missing or incomplete)
            for parent_id in TaxonomyBuilder._extract_direct_parent_ids(tag):
                children_map.setdefault(parent_id, []).append(tag_id)

        # Deduplicate children lists
        for parent_id in children_map:
            children_map[parent_id] = list(dict.fromkeys(children_map[parent_id]))

        return tags_by_id, children_map

    @staticmethod
    def _detect_root(tags_by_id: dict[str, dict[str, Any]]) -> str | None:
        """Auto-detect root tag: a tag with no parents whose subtree is largest.

        Falls back to ``None`` if no parentless tag exists.
        """
        parentless: list[str] = []
        for tag_id, tag in tags_by_id.items():
            parent_ids = TaxonomyBuilder._extract_direct_parent_ids(tag)
            if not parent_ids:
                parentless.append(tag_id)

        if not parentless:
            logger.warning("No parentless tag found; cannot auto-detect root")
            return None

        if len(parentless) == 1:
            logger.info(
                "Auto-detected root tag: %s (%s)",
                parentless[0],
                tags_by_id[parentless[0]]["name"],
            )
            return parentless[0]

        # Multiple roots -- pick the one with the most descendants
        def _count_descendants(root_id: str) -> int:
            visited: set[str] = set()
            queue: deque[str] = deque([root_id])
            while queue:
                tid = queue.popleft()
                if tid in visited:
                    continue
                visited.add(tid)
                tag = tags_by_id.get(tid)
                if tag:
                    for child in tag.get("children", []):
                        queue.append(str(child["id"]))
            return len(visited)

        best = max(parentless, key=_count_descendants)
        logger.info(
            "Multiple parentless tags found (%d); selected %s (%s) as root",
            len(parentless),
            best,
            tags_by_id[best]["name"],
        )
        return best

    # ------------------------------------------------------------------
    # Tag enrichment
    # ------------------------------------------------------------------

    @staticmethod
    def _enrich_tag(
        tag: dict[str, Any],
        tags_by_id: dict[str, dict[str, Any]],
        children_map: dict[str, list[str]],
        root_tag_id: str | None,
        root_name: str | None,
    ) -> dict[str, Any]:
        """Compute all derived fields for a single tag."""
        tag_id = str(tag["id"])

        all_paths = TaxonomyBuilder._compute_all_paths(tag, tags_by_id, root_tag_id, root_name)

        # Primary path is the shortest one (closest to root)
        all_paths.sort(key=len)
        primary_path = all_paths[0] if all_paths else []

        # All direct parent IDs (root included -- it appears in parent refs)
        direct_parent_ids = TaxonomyBuilder._extract_direct_parent_ids(tag)
        all_parent_ids = list(direct_parent_ids)

        # Primary parent
        primary_parent_id = ""
        primary_parent_name = ""
        if all_parent_ids:
            primary_parent_id = all_parent_ids[0]
            parent_tag = tags_by_id.get(primary_parent_id)
            primary_parent_name = parent_tag["name"] if parent_tag else ""

        # Depth: length of primary path (root children are depth 1)
        depth = len(primary_path)

        # Leaf: has no children (that are in the taxonomy)
        child_ids = children_map.get(tag_id, [])
        is_leaf = len(child_ids) == 0

        path_strings = [" > ".join(p) for p in all_paths]

        return {
            "id": tag_id,
            "name": tag["name"],
            "description": tag.get("description") or "",
            "aliases": tag.get("aliases", []) or [],
            "path": primary_path,
            "path_string": " > ".join(primary_path) if primary_path else "",
            "all_paths": all_paths,
            "all_path_strings": path_strings,
            "parent_id": primary_parent_id,
            "parent_name": primary_parent_name,
            "all_parent_ids": all_parent_ids,
            "depth": depth,
            "is_leaf": is_leaf,
            "has_multiple_parents": len(all_parent_ids) > 1,
            "path_count": len(all_paths),
        }

    # ------------------------------------------------------------------
    # Path computation (DAG-aware)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_all_paths(
        tag: dict[str, Any],
        tags_by_id: dict[str, dict[str, Any]],
        root_tag_id: str | None,
        root_name: str | None,
    ) -> list[list[str]]:
        """Compute all root-to-parent paths for a tag (excluding the tag itself).

        Stash embeds the parent chain recursively in each tag. We walk each
        parent upward to the root, collecting name segments. The resulting
        paths list ancestor names from root down to the tag's immediate
        parent.

        For DAG tags with multiple parents, each parent produces one or more
        paths, and all are returned.

        The root tag name IS included in paths (matching ``taxonomy.json``
        convention). For a tag whose parent is the root, the path is
        ``["RootName"]``.

        Args:
            tag: Raw tag dict with nested ``parents``.
            tags_by_id: Lookup of all tags by ID.
            root_tag_id: Root tag ID. When the walk reaches this tag, its
                name is included as the path prefix.
            root_name: Root tag name (used for matching when ID is absent in
                the nested parent chain).

        Returns:
            List of paths, where each path is a list of ancestor tag names
            ordered root-first to immediate-parent. Returns an empty list if
            the tag has no parents at all.
        """
        direct_parents = tag.get("parents", [])
        if not direct_parents:
            return []

        all_paths: list[list[str]] = []

        for parent in direct_parents:
            chains = TaxonomyBuilder._walk_parent_chain(parent, root_tag_id, root_name)
            all_paths.extend(chains)

        # Deduplicate (same path reached via different nesting)
        seen: set[tuple[str, ...]] = set()
        unique: list[list[str]] = []
        for path in all_paths:
            key = tuple(path)
            if key not in seen:
                seen.add(key)
                unique.append(path)

        return unique if unique else [[]]

    @staticmethod
    def _walk_parent_chain(
        parent_node: dict[str, Any],
        root_tag_id: str | None,
        root_name: str | None,
    ) -> list[list[str]]:
        """Recursively walk a nested parent node to build path segments.

        Each nested ``parents`` entry can itself have multiple parents (DAG),
        so this returns a list of paths.

        The root tag name is included as the first element of every path.

        Args:
            parent_node: A parent dict with ``id``, ``name``, and optional
                nested ``parents``.
            root_tag_id: When this tag is reached, include its name and stop.
            root_name: Root tag name for matching when ID is not in chain.

        Returns:
            List of name-lists, each ordered root-first to this parent.
        """
        parent_id = str(parent_node.get("id", ""))
        parent_name = parent_node.get("name", "")

        # If this parent IS the root, include root name and stop
        if root_tag_id and parent_id == root_tag_id:
            return [[parent_name]]

        grandparents = parent_node.get("parents", [])

        if not grandparents:
            # Reached the top of the chain. Include this tag's name.
            return [[parent_name]]

        result: list[list[str]] = []
        for gp in grandparents:
            ancestor_paths = TaxonomyBuilder._walk_parent_chain(gp, root_tag_id, root_name)
            for ancestor_path in ancestor_paths:
                result.append(ancestor_path + [parent_name])

        return result

    # ------------------------------------------------------------------
    # Parent ID extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_direct_parent_ids(tag: dict[str, Any]) -> list[str]:
        """Extract direct parent IDs from a tag's nested ``parents`` field.

        Only the top-level parents list represents direct parents; deeper
        nesting is grandparents/etc.

        Returns:
            List of parent ID strings (may be empty).
        """
        return [str(p["id"]) for p in tag.get("parents", []) if "id" in p]
