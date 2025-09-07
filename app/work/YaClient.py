from typing import List, Dict
import config

class YandexChatClient:
    """
    Лёгкая обёртка над SDK yandex-cloud-ai.
    Сообщения формата: {"role": "system|user|assistant", "text": "..."}
    Возвращает текст ответа.
    """
    def __init__(self):
        from yandex_cloud_ml_sdk import YCloudML
        self.sdk = YCloudML(
            folder_id=config.YANDEX_FOLDER_ID,
            auth=config.YANDEX_SERV_API_KEY,
        )
        self.model = self.sdk.models.completions(config.YANDEX_MODEL)

    def chat(self, messages: List[Dict], temperature: float = 0.2, max_tokens: int = 800) -> str:t
        gen = self.model.configure(temperature=temperature, max_tokens=max_tokens)
        result = gen.run(messages)
        if isinstance(result, list) and result:
            return str(result[0]).strip()
        return str(result).strip()
