from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = CrossEncoder(self.model_name, device="cpu")
        return self._model

    def rerank(self, query: str, chunks: list[dict], limit: int) -> list[dict]:
        if not chunks:
            return []
        pairs = [(query, chunk["content"]) for chunk in chunks]
        scores = self.model.predict(pairs, show_progress_bar=False)
        ranked = []
        for chunk, score in zip(chunks, scores, strict=True):
            ranked.append({**chunk, "rerank_score": float(score)})
        ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return [
            {**chunk, "final_rank": rank}
            for rank, chunk in enumerate(ranked[:limit], start=1)
        ]
