"""
Tag family grouping via recursive depth-limited traversal of the taxonomy DAG.

A "family" is any non-leaf tag together with all its descendants within
`max_depth` edges.  Because the taxonomy is a DAG (tags can have multiple
parents), a single tag can belong to several families — this is intentional.

Families are used as soft conditioning signals for the classifier, not hard
partitions.  Each family carries a learned bias that modulates how evidence
channels (captions, summary, promo) weight tag predictions.
"""
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


def build_children_map(tags: list) -> Dict[str, List[str]]:
    children: Dict[str, List[str]] = defaultdict(list)
    for tag in tags:
        for pid in tag.get("all_parent_ids", []) or []:
            children[str(pid)].append(str(tag["id"]))
    return children


def collect_descendants(tag_id: str, children_map: Dict[str, List[str]], max_depth: int, _depth: int = 0) -> Set[str]:
    if _depth >= max_depth:
        return set()
    result: Set[str] = set()
    for child_id in children_map.get(tag_id, []):
        result.add(child_id)
        result |= collect_descendants(child_id, children_map, max_depth, _depth + 1)
    return result


def build_families(taxonomy: dict, max_depth: int = 2) -> Tuple[List[Dict], Dict[str, List[int]]]:
    tags = taxonomy.get("tags", [])
    children_map = build_children_map(tags)

    families = []
    tag_to_families: Dict[str, List[int]] = defaultdict(list)

    for tag in tags:
        tag_id = str(tag["id"])
        if tag.get("is_leaf", True):
            continue
        members = collect_descendants(tag_id, children_map, max_depth)
        if not members:
            continue
        family_idx = len(families)
        families.append({"id": family_idx, "root_tag_id": tag_id, "root_tag_name": tag.get("name", ""), "member_tag_ids": sorted(members)})
        for mid in members:
            tag_to_families[mid].append(family_idx)

    logger.info(f"Built {len(families)} tag families from {len(tags)} tags (max_depth={max_depth})")
    return families, dict(tag_to_families)
