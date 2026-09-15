"""Knowledge-graph loading and exact or approximate entity matching."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from agp_research.models import Edge, Node


def normalize(text: str) -> str:
    return " ".join(text.casefold().strip().split())


def _trigrams(text: str) -> set[str]:
    padded = f"  {text}  "
    return {padded[index : index + 3] for index in range(len(padded) - 2)}


def edit_similarity(left: str, right: str) -> float:
    """Return one minus normalized Levenshtein distance for normalized strings."""
    left = normalize(left)
    right = normalize(right)
    if left == right:
        return 1.0
    if not left or not right:
        return 0.0
    previous = list(range(len(right) + 1))
    for row, left_character in enumerate(left, start=1):
        current = [row]
        for column, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_character != right_character),
                )
            )
        previous = current
    return 1.0 - previous[-1] / max(len(left), len(right))


@dataclass(frozen=True)
class KeywordMatch:
    keyword: str
    node_id: str | None
    matched_title: str | None
    score: float
    second_score: float | None
    method: str
    accepted: bool


@dataclass
class KnowledgeGraph:
    nodes: dict[str, Node]
    edges: list[Edge]
    _node_id_by_title: dict[str, str] = field(init=False, repr=False)
    _display_title_by_title: dict[str, str] = field(init=False, repr=False)
    _titles_by_trigram: dict[str, set[str]] = field(init=False, repr=False)
    _trigrams_by_title: dict[str, set[str]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Build reusable lookup indexes once per loaded graph."""
        self._node_id_by_title = {}
        self._display_title_by_title = {}
        self._titles_by_trigram = {}
        self._trigrams_by_title = {}
        for node in self.nodes.values():
            normalized_title = normalize(node.title)
            # Preserve exact_match's previous last-duplicate-wins behavior.
            self._node_id_by_title[normalized_title] = node.id
            # Preserve extraction's previous first-duplicate-wins behavior.
            self._display_title_by_title.setdefault(normalized_title, node.title)
            title_trigrams = _trigrams(normalized_title)
            self._trigrams_by_title[normalized_title] = title_trigrams
            for trigram in title_trigrams:
                self._titles_by_trigram.setdefault(trigram, set()).add(normalized_title)

    def _approximate_candidate(
        self, keyword: str, candidate_limit: int
    ) -> tuple[str | None, float, float | None]:
        """Return the best indexed title, its score, and the runner-up score."""
        normalized_keyword = normalize(keyword)
        keyword_trigrams = _trigrams(normalized_keyword)
        candidates = set().union(
            *(self._titles_by_trigram.get(item, set()) for item in keyword_trigrams)
        )
        if not candidates and len(normalized_keyword) < 3:
            candidates = set(self._node_id_by_title)
        ranked_candidates = sorted(
            candidates,
            key=lambda title: (
                -len(keyword_trigrams & self._trigrams_by_title[title])
                / len(keyword_trigrams | self._trigrams_by_title[title]),
                title,
            ),
        )[:candidate_limit]
        scored = sorted(
            ((edit_similarity(normalized_keyword, title), title) for title in ranked_candidates),
            key=lambda item: (-item[0], item[1]),
        )
        best_score, best_title = scored[0] if scored else (0.0, None)
        second_score = scored[1][0] if len(scored) > 1 else None
        return best_title, best_score, second_score

    @classmethod
    def from_csv(cls, nodes_path: str | Path, edges_path: str | Path) -> "KnowledgeGraph":
        with Path(nodes_path).open(encoding="utf-8", newline="") as stream:
            nodes = {
                row["id"]: Node(row["id"], row["title"], row.get("description", ""))
                for row in csv.DictReader(stream)
            }
        with Path(edges_path).open(encoding="utf-8", newline="") as stream:
            edges = [
                Edge(
                    source=row["source"],
                    target=row["target"],
                    description=row.get("description", ""),
                    weight=float(row.get("weight", 1.0)),
                )
                for row in csv.DictReader(stream)
            ]
        unknown = {
            endpoint
            for edge in edges
            for endpoint in (edge.source, edge.target)
            if endpoint not in nodes
        }
        if unknown:
            raise ValueError(f"Edges refer to unknown node IDs: {sorted(unknown)}")
        return cls(nodes=nodes, edges=edges)

    def exact_match(self, keywords: list[str]) -> tuple[list[str], list[str]]:
        matched: list[str] = []
        unmatched: list[str] = []
        for keyword in keywords:
            node_id = self._node_id_by_title.get(normalize(keyword))
            if node_id is None:
                unmatched.append(keyword)
            elif node_id not in matched:
                matched.append(node_id)
        return matched, unmatched

    def approximate_match(
        self,
        keywords: list[str],
        *,
        threshold: float = 0.85,
        margin: float = 0.10,
        candidate_limit: int = 50,
    ) -> tuple[list[str], list[str], list[KeywordMatch]]:
        """Resolve exact titles first, then accept confident similar-title matches."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("match threshold must be between 0 and 1")
        if not 0.0 <= margin <= 1.0:
            raise ValueError("match margin must be between 0 and 1")
        if candidate_limit <= 0:
            raise ValueError("candidate_limit must be positive")

        matched: list[str] = []
        unmatched: list[str] = []
        details: list[KeywordMatch] = []
        unresolved_indexes: list[int] = []
        for index, keyword in enumerate(keywords):
            normalized_keyword = normalize(keyword)
            exact_id = self._node_id_by_title.get(normalized_keyword)
            if exact_id is not None:
                if exact_id not in matched:
                    matched.append(exact_id)
                details.append(
                    KeywordMatch(
                        keyword, exact_id, self.nodes[exact_id].title,
                        1.0, None, "exact", True
                    )
                )
                continue

            unresolved_indexes.append(index)
            details.append(None)  # type: ignore[arg-type]

        # LLMs sometimes split one long entity title into several adjacent keywords.
        # Try longest unresolved spans first, using common natural-language separators.
        consumed: set[int] = set()
        for span_size in range(len(keywords), 1, -1):
            for start in range(len(keywords) - span_size + 1):
                indexes = range(start, start + span_size)
                if any(index not in unresolved_indexes or index in consumed for index in indexes):
                    continue
                variants = (
                    " ".join(keywords[start : start + span_size]),
                    ", ".join(keywords[start : start + span_size]),
                )
                candidates = [
                    (self._approximate_candidate(value, candidate_limit), value)
                    for value in variants
                ]
                ((best_title, best_score, second_score), combined) = max(
                    candidates, key=lambda item: item[0][1]
                )
                confident = (
                    best_title is not None
                    and best_score >= threshold
                    and (second_score is None or best_score - second_score >= margin)
                )
                if not confident or best_title is None:
                    continue
                node_id = self._node_id_by_title[best_title]
                if node_id not in matched:
                    matched.append(node_id)
                details[start] = KeywordMatch(
                    keyword=combined,
                    node_id=node_id,
                    matched_title=self._display_title_by_title[best_title],
                    score=best_score,
                    second_score=second_score,
                    method="approximate-compound",
                    accepted=True,
                )
                consumed.update(indexes)

        for index in unresolved_indexes:
            if index in consumed:
                continue
            keyword = keywords[index]
            best_title, best_score, second_score = self._approximate_candidate(
                keyword, candidate_limit
            )
            confident = (
                best_title is not None
                and best_score >= threshold
                and (second_score is None or best_score - second_score >= margin)
            )
            node_id = self._node_id_by_title[best_title] if best_title else None
            if confident and node_id is not None:
                if node_id not in matched:
                    matched.append(node_id)
            else:
                unmatched.append(keyword)
            details[index] = (
                KeywordMatch(
                    keyword=keyword,
                    node_id=node_id,
                    matched_title=(
                        self._display_title_by_title[best_title] if best_title else None
                    ),
                    score=best_score,
                    second_score=second_score,
                    method="approximate" if confident else "unresolved",
                    accepted=confident,
                )
            )
        return matched, unmatched, [detail for detail in details if detail is not None]

    def title_mentions(self, text: str) -> list[tuple[str, int, int]]:
        """Find indexed titles at the same Unicode word boundaries as ``\\w``."""
        normalized_text = normalize(text)
        length = len(normalized_text)
        is_word = [
            character == "_" or character.isalnum()
            for character in normalized_text
        ]
        starts = [
            position
            for position in range(length)
            if (position == 0 or not is_word[position - 1])
            and not normalized_text[position].isspace()
        ]
        ends = [
            position
            for position in range(1, length + 1)
            if (position == length or not is_word[position])
            and not normalized_text[position - 1].isspace()
        ]

        mentions: list[tuple[str, int, int]] = []
        for start in starts:
            for end in ends:
                if end <= start:
                    continue
                normalized_title = normalized_text[start:end]
                title = self._display_title_by_title.get(normalized_title)
                if title is not None:
                    mentions.append((title, start, end))
        return mentions
