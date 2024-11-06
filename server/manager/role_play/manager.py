from __future__ import annotations

from typing import Optional

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from ..utils import TemplateManager


class RolePlayHistory:
    def __init__(self) -> None:
        self.history: list[ChatCompletionMessageParam] = []

    def add_system(self, response: str) -> None:
        self.history.append({
            "role": "system",
            "content": response
        })

    def add_child(self, response: str) -> None:
        self.history.append({
            "role": "user",
            "content": response
        })

    def add_result(self, response: str) -> None:
        self.history.append({
            "role": "assistant",
            "content": response
        })


class RolePlayManager:
    def __init__(
            self,
            *,
            model: str = "gpt-4o-mini",
            client: Optional[OpenAI] = None
    ) -> None:
        self.templates = TemplateManager(__name__)

        self.model = model
        self.client = client or OpenAI()

    def start(
            self,
            context: str,
            question: str,
            answer: str,
            *,
            temperature: float = 0.8,
            seed: int = 621
    ) -> tuple[str, RolePlayHistory]:
        history = RolePlayHistory()

        history.add_system(self.templates.sys().render())
        history.add_child(self.templates.get("start").render(
            context=context,
            question=question,
            stance=answer
        ))

        response = self.client.chat.completions.create(
            model=self.model,
            messages=history.history,
            temperature=temperature,
            seed=seed
        )

        assert (content := response.choices[0].message.content) is not None

        history.add_result(content)

        return content, history

    def step(
            self,
            context: str,
            question: str,
            answer: str,
            history: RolePlayHistory,
            reply: str,
            *,
            temperature: float = 0.8,
            seed: int = 621
    ) -> tuple[str, RolePlayHistory]:
        history.add_child(reply)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=history.history,
            temperature=temperature,
            seed=seed,
        )

        assert (content := response.choices[0].message.content) is not None

        history.add_result(content)

        return content, history
