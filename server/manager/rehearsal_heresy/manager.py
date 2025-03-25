from __future__ import annotations

import uuid
from typing import Literal, Optional
import random

from langfuse.decorators import langfuse_context, observe
from langfuse.openai import OpenAI
from openai.types.chat import ParsedChatCompletion

from pydantic import BaseModel

from ..utils import TemplateManager


class ResponseData(BaseModel):
    information_seeking: str
    persuasion: str
    inquiry: str


class EvaluationData(BaseModel):
    information_seeking: Literal[0, 1, 2, 3, 4]
    persuasion: Literal[0, 1, 2, 3, 4]
    inquiry: Literal[0, 1, 2, 4]


class ActionData(BaseModel):
    action: Literal["information-seeking", "persuasion", "inquiry"]
    information_seeking: int
    persuasion: int
    inquiry: int


class RehearsalHeresyHistory:
    def __init__(self) -> None:
        self.history: list[str | dict[str, str]] = []

    def add_agent(self, action: ActionData, response: ResponseData) -> None:
        assert len(self.history) == 0 or isinstance(self.history[-1], str)

        self.history.append({
            "action": action.action,
            "evaluation_s": str(action.information_seeking),
            "evaluation_p": str(action.persuasion),
            "evaluation_i": str(action.inquiry),
            "response_s": response.information_seeking,
            "response_p": response.persuasion,
            "response_i": response.inquiry
        })

    def add_child(self, response: str) -> None:
        assert len(self.history) > 0 and isinstance(self.history[-1], dict)

        self.history.append(response)

    def __str__(self) -> str:
        if len(self.history) == 0:
            return "EMPTY"

        responses = []
        for response in self.history:
            if isinstance(response, str):
                responses.append(f"Child: {response}")
            else:
                match response["action"]:
                    case "information_seeking":
                        response = response["response_s"]
                    case "persuasion":
                        response = response["response_p"]
                    case "inquiry":
                        response = response["response_i"]

                responses.append(f"Agent: {response}")

        return "\n".join(responses)


class RehearsalHeresyManager:
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

    def generate_response(
            self,
            context: str,
            question: str,
            answer: str,
            history: RehearsalHeresyHistory,
            *,
            temperature: float = 0.8,
            seed: int = 621,
    ) -> ParsedChatCompletion:
        return self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self.templates.sys().render(
                        context=context,
                        question=question,
                        answer=answer
                    )
                },
                {
                    "role": "user",
                    "content": self.templates.usr("generate").render(
                        history=str(history)
                    )
                }
            ],
            response_format=ResponseData,
            temperature=temperature,
            seed=seed
        )

    def select_action(
        self,
        context: str,
        question: str,
        answer: str,
        history: RehearsalHeresyHistory,
        candidates: ResponseData,
        *,
        temperature: float = 0.8,
        seed: int = 621
    ) -> ActionData:
        random.seed(seed)

        evaluations_res = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self.templates.sys().render(
                        context=context,
                        question=question,
                        answer=answer
                    )
                },
                {
                    "role": "user",
                    "content": self.templates.usr("select").render(
                        history=str(history),
                        candidates=candidates
                    )
                }
            ],
            response_format=EvaluationData,
            temperature=temperature,
            seed=seed
        )

        assert (evaluations :=
                evaluations_res.choices[0].message.parsed) is not None
        assert isinstance(evaluations, EvaluationData)

        # TODO: The following is not good code, but I'm in a rush
        final_action: Optional[Literal[
            "information-seeking",
            "persuasion",
            "inquiry"
        ]] = None

        if self.mode == "mixed":
            # if mix, use the best response (random tie-break)
            best_score = max(
                evaluations.information_seeking,
                evaluations.persuasion,
                evaluations.inquiry
            )

            moves: list[Literal[
                "information-seeking",
                "persuasion",
                "inquiry"
            ]] = []
            if evaluations.information_seeking == best_score:
                moves.append("information-seeking")
            if evaluations.persuasion == best_score:
                moves.append("persuasion")
            if evaluations.inquiry == best_score:
                moves.append("inquiry")

            final_action = random.choice(moves)
        else:
            # if X, use X if its score >= average score, otherwise use best
            average = (
                evaluations.information_seeking
                + evaluations.persuasion
                + evaluations.inquiry
            ) / 3

            score = 0

            match self.mode:
                case "information-seeking":
                    score = evaluations.information_seeking
                case "persuasion":
                    score = evaluations.persuasion
                case "inquiry":
                    score = evaluations.inquiry

            if score >= average:
                final_action = self.mode  # type: ignore
            else:
                best_score = max(
                    evaluations.information_seeking,
                    evaluations.persuasion,
                    evaluations.inquiry
                )

                moves = []
                if evaluations.information_seeking == best_score:
                    moves.append("information-seeking")
                if evaluations.persuasion == best_score:
                    moves.append("persuasion")
                if evaluations.inquiry == best_score:
                    moves.append("inquiry")

                final_action = random.choice(moves)

        assert final_action is not None

        return ActionData(
            action=final_action,
            information_seeking=evaluations.information_seeking,
            persuasion=evaluations.persuasion,
            inquiry=evaluations.inquiry
        )

    @observe(name="rehearsal_heresy_init")
    def start(
        self,
        context: str,
        question: str,
        answer: str
    ) -> tuple[str, RehearsalHeresyHistory, uuid.UUID]:
        history = RehearsalHeresyHistory()

        candidates_res = self.generate_response(
            context, question, answer, history)
        assert (candidates :=
                candidates_res.choices[0].message.parsed) is not None
        assert isinstance(candidates, ResponseData)

        action = self.select_action(
            context, question, answer, history, candidates)

        history.add_agent(action, candidates)

        trace_uuid = langfuse_context.get_current_trace_id()
        assert trace_uuid is not None

        match action.action:
            case "information-seeking":
                response = candidates.information_seeking
            case "persuasion":
                response = candidates.persuasion
            case "inquiry":
                response = candidates.inquiry

        return response, history, uuid.UUID(trace_uuid)

    @observe(name="rehearsal_heresy_step")
    def step(
        self,
        context: str,
        question: str,
        answer: str,
        history: RehearsalHeresyHistory,
        reply: str
    ) -> tuple[str, RehearsalHeresyHistory, uuid.UUID]:
        history.add_child(reply)

        candidates_res = self.generate_response(
            context, question, answer, history)
        assert (candidates :=
                candidates_res.choices[0].message.parsed) is not None
        assert isinstance(candidates, ResponseData)

        action = self.select_action(
            context, question, answer, history, candidates)

        history.add_agent(action, candidates)

        trace_uuid = langfuse_context.get_current_trace_id()
        assert trace_uuid is not None

        match action.action:
            case "information-seeking":
                response = candidates.information_seeking
            case "persuasion":
                response = candidates.persuasion
            case "inquiry":
                response = candidates.inquiry

        return response, history, uuid.UUID(trace_uuid)
