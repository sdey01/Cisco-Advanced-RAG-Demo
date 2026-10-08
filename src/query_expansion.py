import logging

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class ExpansionOutput(BaseModel):
    expanded_queries: list[str] = Field(min_length=3, max_length=3)


class QueryExpander:
    def __init__(self, model: ChatOpenAI):
        self.model = model

    def expand(self, query: str) -> list[str]:
        try:
            structured = self.model.with_structured_output(ExpansionOutput)
            result = structured.invoke(
                [
                    (
                        "system",
                        "Create exactly three distinct retrieval queries for the user's "
                        "Cisco Nexus question. Preserve technical identifiers and intent. "
                        "Do not invent terms or answer the question.",
                    ),
                    ("human", query),
                ]
            )
            values = ExpansionOutput.model_validate(result).expanded_queries
            return [value.strip() for value in values]
        except Exception:
            logger.exception("Query expansion failed; using deterministic variations")
            return [
                query,
                f"Cisco Nexus 9000 troubleshooting information for: {query}",
                f"Documented causes, requirements, or resolution related to: {query}",
            ]
