from typing import List, Dict
import textwrap
import config
from yaclient import YandexChatClient
from retriever_qdrant import QdrantRetriever, Hit

SYSTEM = {"role": "system", "text": config.SYSTEM_PROMPT}

def _truncate_history(history: List[Dict]) -> List[Dict]:
    """Берём последние HISTORY_TURNS пар (user+assistant). System оставляем."""
    sys = [m for m in history if m["role"] == "system"][:1] or [SYSTEM]
    rest = [m for m in history if m["role"] != "system"]
    tail = rest[-2 * config.HISTORY_TURNS :]
    return sys + tail

def _build_context(hits: List[Hit]) -> str:
    ctx_parts = []
    total = 0
    for i, h in enumerate(hits, 1):
        chunk = f"[{i}] (score={h.score:.3f})\n{h.text}\n"
        ctx_parts.append(chunk)
        total += len(chunk)
        if total >= config.MAX_CTX_CHARS:
            break
    return "\n".join(ctx_parts)

class RAGYaGPT:
    """
    Держит историю, ходит в Qdrant за контекстом и спрашивает YaGPT.
    """
    def __init__(self, top_k: int = None):
        self.client = YandexChatClient()
        self.retriever = QdrantRetriever()
        self.top_k = top_k or config.TOP_K
        self.history: List[Dict] = [SYSTEM]

    def ask(self, user_text: str, temperature: float = 0.2) -> str:
        hits = self.retriever.search(user_text, top_k=self.top_k)

        context = _build_context(hits)
        ctx_instr = (
            "Используй ТОЛЬКО информацию из <context>...</context>. "
            "Если ответа нет в контексте — так и скажи. "
            "Цитируй важные фрагменты кратко."
        )
        sys_ctx = {
            "role": "system",
            "text": f"{ctx_instr}\n<context>\n{context}\n</context>"
        }

        messages = _truncate_history(self.history) + [sys_ctx, {"role": "user", "text": user_text}]
        answer = self.client.chat(messages, temperature=temperature, max_tokens=900)

        self.history.append({"role": "user", "text": user_text})
        self.history.append({"role": "assistant", "text": answer})

        return answer, hits

if __name__ == "__main__":
    chat = RAGYaGPT()
    print("RAG + YaGPT. Введите вопрос (exit для выхода):")
    while True:
        q = input("> ").strip()
        if q.lower() in {"exit", "quit", "выход"}:
            break
        answer, hits = chat.ask(q)
        print("\n=== Ответ ===")
        print(textwrap.fill(answer, 110))
        print("\n=== Источники ===")
        for i, h in enumerate(hits, 1):
            src = h.payload.get("source") or h.payload.get("file_path") or h.payload.get("doc_id") or "unknown"
            print(f"[{i}] score={h.score:.3f} — {src}")
        print()
