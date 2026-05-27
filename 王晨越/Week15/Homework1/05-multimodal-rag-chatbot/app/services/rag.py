from app.schemas import ChatResponse, SourceItem
from app.services.chat import ChatModel
from app.services.retrieval import EmbeddingModel, VectorStore


def build_context(hits: list[dict]) -> str:
    return "\n".join(hit["text"] for hit in hits if hit.get("text"))


def preview_text(text: str, max_len: int = 200) -> str:
    t = text.replace("\n", " ")
    return t if len(t) <= max_len else t[: max_len - 3] + "..."


class RagService:
    def __init__(
        self,
        embedder: EmbeddingModel,
        vector_store: VectorStore,
        chat_model: ChatModel,
    ):
        self._embedder = embedder
        self._vector_store = vector_store
        self._chat_model = chat_model

    def chat(self, question: str, top_k: int = 5) -> ChatResponse:
        embedding = self._embedder.encode_query(question)
        hits = self._vector_store.search_text(embedding, top_k)
        context = build_context(hits)
        answer = self._chat_model.generate(question, context)
        sources = [
            SourceItem(
                db_id=int(hit["db_id"]),
                file_name=str(hit["file_name"]),
                text_preview=preview_text(str(hit["text"])),
            )
            for hit in hits
        ]
        return ChatResponse(answer=answer, sources=sources)
