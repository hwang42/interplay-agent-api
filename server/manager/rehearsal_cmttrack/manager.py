from __future__ import annotations

import os
import uuid
from typing import Literal, Optional

from langfuse.decorators import langfuse_context, observe
from langfuse.openai import OpenAI
from openai.types.chat import ParsedChatCompletion

from ..rehearsal import ActionData, ResponseData
from ..utils import TemplateManager


# implement the dialogue analysis module
# based on commitment tracking, which provides assistance for tracking ideas
# and thus helps in generating more coherent responses for new ideas


class RehearsalCommitTrackHistory:

    def __init__(self, init_history: list[str | dict[str, str]] = None) -> None:
        self.history: list[str | dict[str, str]]
        self.history = init_history or []

    def add_agent(
        self, action: ActionData, response: ResponseData, cmttrack: str
    ) -> None:
        assert len(self.history) == 0 or isinstance(self.history[-1], str)

        self.history.append(
            {
                "action_thought": action.thought,
                "action_decision": action.action,
                "response_thought": response.thought,
                "response_response": response.response,
                "response_cmttrack": cmttrack,
            }
        )

    def add_child(self, response: str) -> None:
        assert len(self.history) > 0 and isinstance(self.history[-1], dict)

        self.history.append(response)

    def __str__(self) -> str:
        responses = [
            (
                f"Child: {response}"
                if isinstance(response, str)
                else f"Agent: ({response['action_decision']}) {response['response_cmttrack']}"
            )
            for response in self.history
        ]

        if len(responses) == 0:
            return "EMPTY"

        return "\n".join(responses)

    def __len__(self) -> int:
        return len(self.history)


class RehearsalCommitTrackManager:
    def __init__(
        self,
        mode: Literal["mixed", "inquiry", "persuasion", "information"],
        *,
        model: str = "gpt-4o-mini",
        client: Optional[OpenAI] = None,
    ) -> None:
        self.templates = TemplateManager(__name__)

        self.mode = mode
        self.model = model
        self.client = client or OpenAI()

        self.cmttrack_model = os.getenv("CMTTRACK_MODEL", model)
        self.cmttrack_client = OpenAI(
            api_key=os.getenv("CMTTRACK_API_KEY", None),
            base_url=os.getenv("CMTTRACK_BASE_URL", None),
        )

    def select_action(
        self,
        context: str,
        question: str,
        answer: str,
        history: RehearsalCommitTrackHistory,
        *,
        temperature: float = 0.8,
        seed: int = 621,
    ) -> ParsedChatCompletion:
        return self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self.templates.sys(
                        "select", suffix=f"sys_{self.mode}"
                    ).render(),
                },
                {
                    "role": "user",
                    "content": self.templates.usr("select").render(
                        context=context,
                        question=question,
                        stance=answer,
                        conversation=str(history),
                    ),
                },
            ],
            response_format=ActionData,
            temperature=temperature,
            seed=seed,
        )

    def generate_response(
        self,
        context: str,
        question: str,
        answer: str,
        history: RehearsalCommitTrackHistory,
        action: Literal[
            "new information-seeking",
            "continue information-seeking",
            "new persuasion",
            "continue persuasion",
            "new co-construction",
            "continue co-construction",
        ],
        cmttrack: str,
        *,
        temperature: float = 0.8,
        seed: int = 621,
    ) -> ParsedChatCompletion:
        template = {
            "information-seeking": "s",
            "persuasion": "p",
            "co-construction": "i",
        }[action.split()[1]]

        rendered_prompt = self.templates.usr(template).render(
            context=context,
            question=question,
            stance=answer,
            conversation=str(history),
        )
        if cmttrack:
            rendered_prompt += f"## Conversation Commitments Summary:\n\n{cmttrack}"

        return self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self.templates.sys(
                        "shared", suffix=f"sys_{self.mode}"
                    ).render(),
                },
                {
                    "role": "user",
                    "content": rendered_prompt,
                },
            ],
            response_format=ResponseData,
            temperature=temperature,
            seed=seed,
        )

    def generate_commitment_tracking(
        self,
        history: RehearsalCommitTrackHistory,
        *,
        turn_based: bool = False,
        temperature: float = 0.8,
        seed: int = 621,
    ) -> str:
        if turn_based:
            # strong commitment tracking, at least one commitment for one turn
            l = [
                self.generate_commitment_tracking(
                    RehearsalCommitTrackHistory(history.history[i : i + 2]),
                    turn_based=False,
                    temperature=temperature,
                    seed=seed,
                )
                for i in range(0, len(history.history), 2)
            ]
            return "\n".join(l)

        response = self.cmttrack_client.chat.completions.create(
            model=self.cmttrack_model,
            messages=[
                {
                    "role": "user",
                    "content": f"""Identify the words or phrases in the following conversation that express commitment or dedication to a particular cause, goal, or obligation.

{str(history)}

Now list all the identified commitments.

- """,
                }
            ],
            temperature=temperature,
            max_tokens=128 * 4,
            seed=seed,
        )

        assert response.choices[0].message.content is not None

        return "- " + response.choices[0].message.content.strip()

    @observe(name="rehearsal_cmttrack_init")
    def start(
        self, context: str, question: str, answer: str
    ) -> tuple[str, RehearsalCommitTrackHistory, uuid.UUID]:
        history = RehearsalCommitTrackHistory()

        action_res = self.select_action(context, question, answer, history)
        assert (action := action_res.choices[0].message.parsed) is not None
        assert isinstance(action, ActionData)

        response_cmttrack = ""
        response_res = self.generate_response(
            context, question, answer, history, action.action, response_cmttrack
        )
        assert (response := response_res.choices[0].message.parsed) is not None
        assert isinstance(response, ResponseData)

        history.add_agent(action, response, response_cmttrack)

        trace_uuid = langfuse_context.get_current_trace_id()
        assert trace_uuid is not None

        return response.response, history, uuid.UUID(trace_uuid)

    @observe(name="rehearsal_cmttrack_step")
    def step(
        self,
        context: str,
        question: str,
        answer: str,
        history: RehearsalCommitTrackHistory,
        reply: str,
    ) -> tuple[str, RehearsalCommitTrackHistory, uuid.UUID]:
        history.add_child(reply)

        action_res = self.select_action(context, question, answer, history)
        assert (action := action_res.choices[0].message.parsed) is not None
        assert isinstance(action, ActionData)

        response_cmttrack = self.generate_commitment_tracking(history)
        response_res = self.generate_response(
            context, question, answer, history, action.action, response_cmttrack
        )
        assert (response := response_res.choices[0].message.parsed) is not None
        assert isinstance(response, ResponseData)

        history.add_agent(action, response, response_cmttrack)

        trace_uuid = langfuse_context.get_current_trace_id()
        assert trace_uuid is not None

        return response.response, history, uuid.UUID(trace_uuid)
