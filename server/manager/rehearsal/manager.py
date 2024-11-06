from __future__ import annotations

from typing import Literal, Optional

from openai import OpenAI
from openai.types.chat import ParsedChatCompletion
from pydantic import BaseModel

from ..utils import TemplateManager


class ActionData(BaseModel):
    thought: str
    action: Literal[
        "new information-seeking",
        "continue information-seeking",
        "new persuasion",
        "continue persuasion",
        "new inquiry",
        "continue inquiry"
    ]


class ResponseData(BaseModel):
    thought: str
    response: str


class RehearsalHistory:
    def __init__(self) -> None:
        self.history: list[str | dict[str, str]] = []

    def add_agent(self, action: ActionData, response: ResponseData) -> None:
        assert len(self.history) == 0 or isinstance(self.history[-1], str)

        self.history.append({
            "action_thought": action.thought,
            "action_decision": action.action,
            "response_thought": response.thought,
            "response_response": response.response
        })

    def add_child(self, response: str) -> None:
        assert len(self.history) > 0 and isinstance(self.history[-1], dict)

        self.history.append(response)

    def __str__(self) -> str:
        responses = [
            f"Child: {response}" if isinstance(response, str)
            else f"Agent: ({response['action_decision']}) {response['response_response']}"
            for response in self.history
        ]

        if len(responses) == 0:
            return "EMPTY"

        return "\n".join(responses)


class RehearsalManager:
    def __init__(
            self,
            *,
            model: str = "gpt-4o-mini",
            client: Optional[OpenAI] = None
    ) -> None:
        self.templates = TemplateManager(__name__)

        self.model = model
        self.client = client or OpenAI()

    def select_action(
            self,
            context: str,
            question: str,
            answer: str,
            history: RehearsalHistory,
            *,
            temperature: float = 0.8,
            seed: int = 621
    ) -> ParsedChatCompletion:
        return self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self.templates.sys("select").render()
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
            history: RehearsalHistory,
            action: Literal[
                "new information-seeking",
                "continue information-seeking",
                "new persuasion",
                "continue persuasion",
                "new inquiry",
                "continue inquiry"
            ],
            *,
            temperature: float = 0.8,
            seed: int = 621
    ) -> ParsedChatCompletion:
        template = {
            "information-seeking": "s",
            "persuasion": "p",
            "inquiry": "i"
        }[action.split()[1]]

        return self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self.templates.sys("shared").render()
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

    def start(
            self,
            context: str,
            question: str,
            answer: str
    ) -> tuple[str, RehearsalHistory]:
        history = RehearsalHistory()

        action_res = self.select_action(context, question, answer, history)
        assert (action := action_res.choices[0].message.parsed) is not None
        assert isinstance(action, ActionData)

        response_res = self.generate_response(
            context, question, answer, history, action.action)
        assert (response := response_res.choices[0].message.parsed) is not None
        assert isinstance(response, ResponseData)

        history.add_agent(action, response)

        return response.response, history

    def step(
            self,
            context: str,
            question: str,
            answer: str,
            history: RehearsalHistory,
            reply: str
    ) -> tuple[str, RehearsalHistory]:
        history.add_child(reply)

        action_res = self.select_action(context, question, answer, history)
        assert (action := action_res.choices[0].message.parsed) is not None
        assert isinstance(action, ActionData)

        response_res = self.generate_response(
            context, question, answer, history, action.action)
        assert (response := response_res.choices[0].message.parsed) is not None
        assert isinstance(response, ResponseData)

        history.add_agent(action, response)

        return response.response, history
