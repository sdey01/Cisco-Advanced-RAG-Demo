import logging
from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class ClassificationOutput(BaseModel):
    category: Literal["FACTUAL_LOOKUP", "BROAD_STRATEGIC"]
    confidence: float = Field(ge=0, le=1)
    reason: str


class QueryClassifier:
    def __init__(self, model: ChatOpenAI):
        self.model = model

    def classify(self, query: str) -> ClassificationOutput:
        try:
            structured = self.model.with_structured_output(ClassificationOutput)
            result = structured.invoke(
                [
                    (
                        "system",
                        "Classify Cisco Nexus troubleshooting questions. Use FACTUAL_LOOKUP "
                        "for one precise definition, code, setting, or requirement. Use "
                        "BROAD_STRATEGIC when synthesis, causes, or multiple steps are needed.",
                    ),
                    ("human", query),
                ]
            )
            return ClassificationOutput.model_validate(result)
        except Exception:
            logger.exception("Query classification failed; using broad fallback")
            return ClassificationOutput(
                category="BROAD_STRATEGIC",
                confidence=0.0,
                reason="Classification failed; using the safe broad-retrieval fallback.",
            )
