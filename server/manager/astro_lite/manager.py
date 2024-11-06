from __future__ import annotations

import re
from typing import Literal, Optional

from langfuse.decorators import observe
from langfuse.openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from ..utils import TemplateManager

PATTERN = r"""```markdown
\[Agenda\] [^[]+
\[Obligation\] [^[]+
\[Thought\] [^[]+
\[Critique\] [^[]+
\[Revised\] [^[]+
\[Dialogue\] I will (start a new|continue the existing) (information-seeking|persuasion|inquiry) dialogue about [^[]+
\[Comment\] [^[]+
\[Response\] [^[]+
\[End\]
\[Terminate\] [^[]+
```"""


class AstroLiteHistory:
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


class AstroLiteManager:
    def __init__(
            self,
            mode: Literal["mixed", "inquiry", "persuasion", "information"],
            *,
            model: str = "gpt-4o-mini",
            client: Optional[OpenAI] = None
    ) -> None:
        self.templates = TemplateManager(__name__)

        self.mode = mode
        self.model = model
        self.client = client or OpenAI()

    @observe(name="astro_lite_init")
    def start(
            self,
            context: str,
            question: str,
            answer: str,
            *,
            temperature: float = 0.8,
            seed: int = 621
    ) -> tuple[str, AstroLiteHistory]:
        history = AstroLiteHistory()

        history.add_system(
            self.templates.sys(suffix=f"sys_{self.mode}").render())
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
        assert (match := re.search(r"\[Response\](.+)\[End\]", content, re.S))

        history.add_result(content)

        return match.group(1).strip(), history

    @observe(name="astro_lite_step")
    def step(
            self,
            context: str,
            question: str,
            answer: str,
            history: AstroLiteHistory,
            reply: str,
            *,
            temperature: float = 0.8,
            seed: int = 621
    ) -> tuple[str, AstroLiteHistory]:
        history.add_child(reply)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=history.history,
            temperature=temperature,
            seed=seed,
            extra_body=dict(guided_regex=PATTERN)
        )

        assert (content := response.choices[0].message.content) is not None
        assert (match := re.search(r"\[Response\](.+)\[End\]", content, re.S))

        history.add_result(content)

        return match.group(1).strip(), history
