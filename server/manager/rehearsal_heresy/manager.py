from __future__ import annotations

import random
import uuid
from typing import Literal, Optional, TypedDict

from langfuse.decorators import langfuse_context, observe
from langfuse.openai import OpenAI
from openai.types.chat import ParsedChatCompletion

from pydantic import BaseModel

from ..utils import TemplateManager


class ResponseData(BaseModel):
    information_seeking: str
    persuasion: str
    inquiry: str
    free: str


Score = Literal[0, 1, 2, 3, 4]


class EvaluationData(BaseModel):
    information_seeking: Score
    persuasion: Score
    inquiry: Score
    free: Score


Action = Literal["information-seeking", "persuasion", "inquiry", "free"]


class ActionData(BaseModel):
    action: Action
    information_seeking: int
    persuasion: int
    inquiry: int
    free: int


class RehearsalHeresyHistory:
    class AgentEntry(TypedDict):
        action: Action

        evaluation_s: str
        evaluation_p: str
        evaluation_i: str
        evaluation_f: str

        response_s: str
        response_p: str
        response_i: str
        response_f: str

    def __init__(self) -> None:
        self.history: list[str | RehearsalHeresyHistory.AgentEntry] = []

    def add_agent(self, action: ActionData, response: ResponseData) -> None:
        assert len(self.history) == 0 or isinstance(self.history[-1], str)

        self.history.append({
            "action": action.action,

            "evaluation_s": str(action.information_seeking),
            "evaluation_p": str(action.persuasion),
            "evaluation_i": str(action.inquiry),
            "evaluation_f": str(action.free),

            "response_s": response.information_seeking,
            "response_p": response.persuasion,
            "response_i": response.inquiry,
            "response_f": response.free
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
                    case "information-seeking":
                        response = response["response_s"]
                    case "persuasion":
                        response = response["response_p"]
                    case "inquiry":
                        response = response["response_i"]
                    case "free":
                        response = response["response_f"]

                responses.append(f"Agent: {response}")

        return "\n".join(responses)


class RehearsalHeresyManager:
    def __init__(
            self,
            mode: Literal["mixed", "inquiry", "persuasion", "information"],
            *,
            model: str = "gpt-4o-mini",
            client: Optional[OpenAI] = None,
            free_threshold=0.25
    ) -> None:
        self.templates = TemplateManager(__name__)

        self.mode = mode
        self.model = model
        self.client = client or OpenAI()

        self.free_threshold = free_threshold

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

        scores: dict[Action, int] = {
            "information-seeking": evaluations.information_seeking,
            "persuasion": evaluations.persuasion,
            "inquiry": evaluations.inquiry
        }

        # we only allow certain percentage usage of the "free" response
        total, count = 0, 0
        for entry in history.history:
            if isinstance(entry, str):
                continue

            total += 1
            
            if entry["action"] == "free":
                count += 1
        
        if total == 0 or (count / total) <= self.free_threshold:
            scores["free"] = evaluations.free

        # select the actual move based on the evaluation scores
        final_action: Action

        if self.mode == "mixed":
            # if mixed, use the best response (random tie-break)
            best = max(scores.values())
            moves: list[Action] = [k for k, v in scores.items() if v == best]

            final_action = random.choice(moves)
        else:
            # if X, use X if its score >= average score, otherwise use best
            average = sum(scores.values()) / len(scores)

            match self.mode:
                case "information":
                    score = scores["information-seeking"]
                case "persuasion":
                    score = scores["persuasion"]
                case "inquiry":
                    score = scores["inquiry"]
                case mode:
                    raise RuntimeError(f"Unknown mode: {mode}")

            if score >= average:
                match self.mode:
                    case "information":
                        final_action = "information-seeking"
                    case "persuasion":
                        final_action = "persuasion"
                    case "inquiry":
                        final_action = "inquiry"
            else:
                best = max(scores.values())
                moves = [k for k, v in scores.items() if v == best]

                final_action = random.choice(moves)

        return ActionData(
            action=final_action,
            information_seeking=evaluations.information_seeking,
            persuasion=evaluations.persuasion,
            inquiry=evaluations.inquiry,
            free=evaluations.free
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
            case "free":
                response = candidates.free

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
            case "free":
                response = candidates.free

        return response, history, uuid.UUID(trace_uuid)
