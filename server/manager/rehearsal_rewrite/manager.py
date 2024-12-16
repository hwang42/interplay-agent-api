from __future__ import annotations

import os
import uuid
from typing import Literal, Optional

from langfuse.decorators import langfuse_context, observe
from langfuse.openai import OpenAI
from openai.types.chat import ParsedChatCompletion

from ..rehearsal import ActionData, ResponseData
from ..utils import TemplateManager


class RehearsalRewriteHistory:
    def __init__(self) -> None:
        self.history: list[str | dict[str, str]] = []

    def add_agent(self, action: ActionData, response: ResponseData, rewrite: str) -> None:
        assert len(self.history) == 0 or isinstance(self.history[-1], str)

        self.history.append({
            "action_thought": action.thought,
            "action_decision": action.action,
            "response_thought": response.thought,
            "response_response": response.response,
            "response_rewrite": rewrite
        })

    def add_child(self, response: str) -> None:
        assert len(self.history) > 0 and isinstance(self.history[-1], dict)

        self.history.append(response)

    def __str__(self) -> str:
        responses = [
            f"Child: {response}" if isinstance(response, str)
            else f"Agent: ({response['action_decision']}) {response['response_rewrite']}"
            for response in self.history
        ]

        if len(responses) == 0:
            return "EMPTY"

        return "\n".join(responses)


class RehearsalRewriteManager:
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

        self.rewrite_model = os.getenv("rewrite_model", model)
        self.rewrite_client = OpenAI(
            api_key=os.getenv("rewrite_api_key", None),
            base_url=os.getenv("rewrite_base_url", None)
        )

    def select_action(
            self,
            context: str,
            question: str,
            answer: str,
            history: RehearsalRewriteHistory,
            *,
            temperature: float = 0.8,
            seed: int = 621
    ) -> ParsedChatCompletion:
        return self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self.templates.sys(
                        "select", suffix=f"sys_{self.mode}").render()
                },
                {
                    "role": "user",
                    "content": self.templates.usr("select").render(
                        context=context,
                        question=question,
                        stance=answer,
                        conversation=str(history)
                    )
                }
            ],
            response_format=ActionData,
            temperature=temperature,
            seed=seed
        )

    def generate_response(
            self,
            context: str,
            question: str,
            answer: str,
            history: RehearsalRewriteHistory,
            action: Literal[
                "new information-seeking",
                "continue information-seeking",
                "new persuasion",
                "continue persuasion",
                "new co-construction",
                "continue co-construction"
            ],
            *,
            temperature: float = 0.8,
            seed: int = 621
    ) -> ParsedChatCompletion:
        template = {
            "information-seeking": "s",
            "persuasion": "p",
            "co-construction": "i"
        }[action.split()[1]]

        return self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self.templates.sys(
                        "shared", suffix=f"sys_{self.mode}").render()
                },
                {
                    "role": "user",
                    "content": self.templates.usr(template).render(
                        context=context,
                        question=question,
                        stance=answer,
                        conversation=str(history)
                    )
                }
            ],
            response_format=ResponseData,
            temperature=temperature,
            seed=seed
        )

    def generate_rewrite(
            self,
            history: RehearsalRewriteHistory,
            content: str,
            temperature: float = 0.8,
            seed: int = 621
    ) -> str:
        response = self.rewrite_client.chat.completions.create(
            model=self.rewrite_model,
            messages=[
                {
                    "role": "user",
                    "content": f""""What should a conversational agent, acting as a parent, should say in the following conversational context:

{str(history)}

Here is a hint of what an agent not acting as a parent might say: {content}"""
                }
            ],
            temperature=temperature,
            max_tokens=128,
            stop="\n\n",
            seed=seed
        )

        assert response.choices[0].message.content is not None

        return response.choices[0].message.content.strip()

    @observe(name="rehearsal_init")
    def start(
            self,
            context: str,
            question: str,
            answer: str
    ) -> tuple[str, RehearsalRewriteHistory, uuid.UUID]:
        history = RehearsalRewriteHistory()

        action_res = self.select_action(context, question, answer, history)
        assert (action := action_res.choices[0].message.parsed) is not None
        assert isinstance(action, ActionData)

        response_res = self.generate_response(
            context, question, answer, history, action.action)
        assert (response := response_res.choices[0].message.parsed) is not None
        assert isinstance(response, ResponseData)

        response_rewrite = self.generate_rewrite(history, response.response)

        history.add_agent(action, response, response_rewrite)

        trace_uuid = langfuse_context.get_current_trace_id()
        assert trace_uuid is not None

        return response.response, history, uuid.UUID(trace_uuid)

    @observe(name="rehearsal_step")
    def step(
            self,
            context: str,
            question: str,
            answer: str,
            history: RehearsalRewriteHistory,
            reply: str
    ) -> tuple[str, RehearsalRewriteHistory, uuid.UUID]:
        history.add_child(reply)

        action_res = self.select_action(context, question, answer, history)
        assert (action := action_res.choices[0].message.parsed) is not None
        assert isinstance(action, ActionData)

        response_res = self.generate_response(
            context, question, answer, history, action.action)
        assert (response := response_res.choices[0].message.parsed) is not None
        assert isinstance(response, ResponseData)

        response_rewrite = self.generate_rewrite(history, response.response)

        history.add_agent(action, response, response_rewrite)

        trace_uuid = langfuse_context.get_current_trace_id()
        assert trace_uuid is not None

        return response.response, history, uuid.UUID(trace_uuid)
