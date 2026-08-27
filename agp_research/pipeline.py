"""End-to-end AGP retrieval and question-answering pipeline."""

from __future__ import annotations

import time
from typing import Protocol

from agp_research.graph import KnowledgeGraph
from agp_research.llm import OpenAICompatibleClient
from agp_research.models import AGPParameters, PipelineResult
from agp_research.planner import heuristic_keywords, llm_plan, local_keyword_extraction, rule_parameters
from agp_research.propagation import propagate


class PropagationBackend(Protocol):
    def propagate(self, graph, seed_ids, parameters): ...


class AGPPipeline:
    def __init__(
        self,
        graph: KnowledgeGraph,
        client: OpenAICompatibleClient | None = None,
        propagation_backend: PropagationBackend | None = None,
    ):
        self.graph = graph
        self.client = client
        self.propagation_backend = propagation_backend

    def run(
        self,
        question: str,
        *,
        strategy: str = "rules",
        fixed_parameters: AGPParameters | None = None,
        generate_answer: bool = True,
    ) -> PipelineResult:
        started = time.perf_counter()

        # Determine keywords and propagation parameters based on the chosen strategy
        if strategy == "llm":
            if self.client is None:
                raise RuntimeError("LLM strategy requires OPENAI_API_KEY")
            keywords, parameters = llm_plan(question, self.client)
        elif strategy == "fixed":
            keywords = local_keyword_extraction(question, self.graph)
            parameters = (fixed_parameters or AGPParameters()).validate()
        elif strategy == "rules":
            keywords = local_keyword_extraction(question, self.graph)
            parameters = rule_parameters(question)
        elif strategy == "seed-only":
            keywords = local_keyword_extraction(question, self.graph)
            parameters = AGPParameters(depth=0, decay=0.0, top_k=10)
        else:
            raise ValueError("strategy must be seed-only, fixed, rules, or llm")

        # If no keywords were found, fall back to heuristic extraction
        if not keywords:
            keywords = heuristic_keywords(question)

        # Match the extracted keywords against the graph to find seed nodes for propagation
        seed_ids, unmatched = self.graph.exact_match(keywords)

        # execute graph propagation to retrieve relevant nodes based on the seed nodes and parameters
        ranked_nodes = (
            self.propagation_backend.propagate(self.graph, seed_ids, parameters)
            if self.propagation_backend is not None
            else propagate(self.graph, seed_ids, parameters)
        )

        # Build a context string summarizing the retrieved nodes and relationships
        context = self._build_context(ranked_nodes)

        #  Generate an answer using the LLM if requested
        answer = self._answer(question, context) if generate_answer else ""
        return PipelineResult(
            question=question,
            strategy=strategy,
            keywords=keywords,
            matched_seed_ids=seed_ids,
            unmatched_keywords=unmatched,
            parameters=parameters,
            ranked_nodes=ranked_nodes,
            context=context,
            answer=answer,
            elapsed_seconds=time.perf_counter() - started,
            metadata={
                "llm_used": strategy == "llm" or (generate_answer and self.client is not None),
                "propagation_backend": (
                    type(self.propagation_backend).__name__
                    if self.propagation_backend is not None
                    else "python-exact"
                ),
            },
        )

    def _build_context(self, ranked_nodes) -> str:
        if not ranked_nodes:
            return "No graph evidence was retrieved."
        selected = {item.node.id for item in ranked_nodes}
        lines = ["Retrieved entities:"]
        for item in ranked_nodes:
            lines.append(f"- {item.node.title} [score={item.score:.4f}]: {item.node.description}")
        relationships = [
            edge for edge in self.graph.edges if edge.source in selected and edge.target in selected
        ]
        if relationships:
            lines.append("Retrieved relationships:")
            for edge in relationships:
                source = self.graph.nodes[edge.source].title
                target = self.graph.nodes[edge.target].title
                lines.append(f"- {source} -> {target}: {edge.description}")
        return "\n".join(lines)

    def _answer(self, question: str, context: str) -> str:
        if self.client is None:
            return "LLM answer generation skipped. Set OPENAI_API_KEY to generate an answer."
        system = "Answer using only the supplied graph context. If it is insufficient, say so explicitly."
        return self.client.complete_text(system, f"Question: {question}\n\nGraph context:\n{context}")
