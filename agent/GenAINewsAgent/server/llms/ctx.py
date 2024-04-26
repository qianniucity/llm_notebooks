from typing import List, Dict, Literal, Union
from transformers import AutoTokenizer


class ContextManagement:

    def __init__(self):
        # assert "mistral" in model_name, "MistralCtx only available for Mistral models"
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B")

    def __count_tokens__(self, content: str):
        tokens = self.tokenizer.tokenize(content)
        return len(tokens) + 2

    def __pad_content__(self, content: str, num_tokens: int):
        return self.tokenizer.decode(
            self.tokenizer.encode(content, max_length=num_tokens))

    def __call__(self, messages: List[Dict], max_length: int = 28_000):
        managed_messages = []
        current_length = 0
        current_message_role = None
        for ix, message in enumerate(messages[::-1]):
            content = message.get("content")
            message_tokens = self.__count_tokens__(message.get("content"))
            if ix > 0:
                if current_length + message_tokens >= max_length:
                    tokens_to_keep = max_length - current_length
                    if tokens_to_keep > 0:
                        content = self.__pad_content__(content, tokens_to_keep)
                        current_length += tokens_to_keep
                    else:
                        break
                if message.get("role") == current_message_role:
                    managed_messages[-1]["content"] += f"\n\n{content}"
                else:
                    managed_messages.append({
                        "role": message.get("role"),
                        "content": content
                    })
                    current_message_role = message.get("role")
                    current_length += message_tokens
            else:
                if current_length + message_tokens >= max_length:
                    tokens_to_keep = max_length - current_length
                    if tokens_to_keep > 0:
                        content = self.__pad_content__(content, tokens_to_keep)
                        current_length += tokens_to_keep
                        managed_messages.append({
                            "role": message.get("role"),
                            "content": content
                        })
                    else:
                        break
                else:
                    managed_messages.append({
                        "role": message.get("role"),
                        "content": content
                    })
                    current_length += message_tokens
                current_message_role = message.get("role")
            # print(managed_messages)
        print(f"TOTAL TOKENS: ", current_length)
        return managed_messages[::-1]


if __name__ == "__main__":
    import json
    messages = [{
        "role": "user",
        "content": "What is your favourite condiment?"
    }, {
        "role":
        "assistant",
        "content":
        "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"
    }, {
        "role": "user",
        "content": "Do you have mayonnaise recipes?"
    }, {
        "role": "user",
        "content": "Do you have mayonnaise recipes? - 2"
    }]
    ctxmgmt = ContextManagement()
    print(json.dumps(ctxmgmt(messages, 45), indent=4))
