from __future__ import annotations

import os
import re
import uuid as uuid_lib
from typing import Annotated, Type

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from pydantic import BaseModel, Field

from sqlalchemy import select
from sqlalchemy.orm import Session

from .database import History, Setup, get_database
from .manager.astro_lite import AstroLiteHistory, AstroLiteManager
from .manager.astro_next import AstroNextHistory, AstroNextManager
from .manager.rehearsal import RehearsalHistory, RehearsalManager
from .manager.rehearsal_heresy import (RehearsalHeresyHistory,
                                       RehearsalHeresyManager)
from .manager.rehearsal_rewrite import (RehearsalRewriteHistory,
                                        RehearsalRewriteManager)
from .manager.role_play import RolePlayHistory, RolePlayManager

KEY = os.getenv("MASTER_KEY")

MODELS: dict[str,
             tuple[RolePlayManager, Type[RolePlayHistory]]
             | tuple[RehearsalManager, Type[RehearsalHistory]]
             | tuple[RehearsalHeresyManager, Type[RehearsalHeresyHistory]]
             | tuple[RehearsalRewriteManager, Type[RehearsalRewriteHistory]]
             | tuple[AstroLiteManager, Type[AstroLiteHistory]]
             | tuple[AstroNextManager, Type[AstroNextHistory]]] = {
    # role play agent
    "role_play": (RolePlayManager(), RolePlayHistory),
    # rehearsal agent
    "rehearsal_mixed": (RehearsalManager("mixed"), RehearsalHistory),
    "rehearsal_inquiry": (RehearsalManager("inquiry"), RehearsalHistory),
    "rehearsal_persuasion": (RehearsalManager("persuasion"), RehearsalHistory),
    "rehearsal_information": (RehearsalManager("information"), RehearsalHistory),
    # rehearsal heresy agent
    "rehearsal_heresy_mixed": (RehearsalHeresyManager("mixed"), RehearsalHeresyHistory),
    "rehearsal_heresy_inquiry": (RehearsalHeresyManager("inquiry"), RehearsalHeresyHistory),
    "rehearsal_heresy_persuasion": (RehearsalHeresyManager("persuasion"), RehearsalHeresyHistory),
    "rehearsal_heresy_information": (RehearsalHeresyManager("information"), RehearsalHeresyHistory),
    # rehearsal rewrite agent
    # "rehearsal_rewrite_mixed": (RehearsalRewriteManager("mixed"), RehearsalRewriteHistory),
    # "rehearsal_rewrite_inquiry": (RehearsalRewriteManager("inquiry"), RehearsalRewriteHistory),
    # "rehearsal_rewrite_persuasion": (RehearsalRewriteManager("persuasion"), RehearsalRewriteHistory),
    # "rehearsal_rewrite_information": (RehearsalRewriteManager("information"), RehearsalRewriteHistory),
    # astro lite agent
    "astro_lite_mixed": (AstroLiteManager("mixed"), AstroLiteHistory),
    "astro_lite_inquiry": (AstroLiteManager("inquiry"), AstroLiteHistory),
    "astro_lite_persuasion": (AstroLiteManager("persuasion"), AstroLiteHistory),
    "astro_lite_information": (AstroLiteManager("information"), AstroLiteHistory),
    # astro next agent
    "astro_next": (AstroNextManager(), AstroNextHistory)
}

api = FastAPI(version="0.1.7")

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"]
)

security = HTTPBearer()


@api.get("/models")
async def get_models():
    """Get a list of the currently available models."""
    return {"models": list(MODELS.keys())}


class Reply(BaseModel):
    response: str = Field(
        description="Agent's initial response.")
    reference: uuid_lib.UUID = Field(
        description="A UUID reference to be used to get back to this point of the conversation.")


class InitPostData(BaseModel):
    model: str = Field(
        description="A model identifier returned from `/models`.")
    story: str = Field(
        description="A story text to be used throughout this conversation.")
    question: str = Field(
        description="A question to be discussed throughout this conversation.")
    answer: str = Field(
        description="A reference answer to be used as the agent's stance throughout this conversation.")


@api.post("/init")
async def post_init(key: Annotated[HTTPAuthorizationCredentials, Depends(security)], data: InitPostData, database: Session = Depends(get_database)) -> Reply:
    """Initialize a new conversation with a specific model."""

    if key.scheme != "Bearer" or key.credentials != KEY:
        raise HTTPException(401, "Invalid API key")

    if data.model not in MODELS:
        raise HTTPException(404, f"Model: \"{data.model}\" is unknown")

    # generate the initial response
    manager = MODELS[data.model][0]
    response, history, trace_uuid = manager.start(
        context=data.story, question=data.question, answer=data.answer)

    # store the history's content
    with database.begin():
        parent = None

        for index, item in enumerate(history.history):
            if isinstance(history, (RehearsalHistory, RehearsalHeresyHistory, RehearsalRewriteHistory)):
                # item is either a str (child) or dict (agent)
                if isinstance(item, str):
                    entry_data = {"message": item}
                    has_trace = False
                else:
                    entry_data = item
                    has_trace = True
            else:
                # item is all dict (ChatCompletionMessageParam)
                assert not isinstance(item, str)
                entry_data = dict(item)
                has_trace = entry_data["role"] == "assistant"

            database.add(History(
                parent=parent,
                uuid=(parent := uuid_lib.uuid4()),
                trace=trace_uuid if has_trace else None,
                model=data.model,
                data=entry_data
            ))

            if index == 0:
                database.add(Setup(
                    history=parent,
                    context=data.story,
                    question=data.question,
                    answer=data.answer
                ))

    assert parent is not None
    return Reply(response=response, reference=parent)


class StepPostData(BaseModel):
    model: str = Field(
        description="A model identifier returned from `/models`.")
    message: str = Field(
        description="The message to the model.")
    reference: uuid_lib.UUID = Field(
        description="A UUID reference specifying the point of conversation to reply to.")


@api.post("/step")
async def post_step(key: Annotated[HTTPAuthorizationCredentials, Depends(security)], data: StepPostData, database: Session = Depends(get_database)) -> Reply:
    """Continue one additional step with an existing conversation."""

    if key.scheme != "Bearer" or key.credentials != KEY:
        raise HTTPException(401, "Invalid API key")

    if data.model not in MODELS:
        raise HTTPException(404, f"Model: \"{data.model}\" is unknown")

    with database.begin():
        reference = (database.query(History)
                     .filter(History.uuid == data.reference)
                     .one_or_none())

        if reference is None or data.model != reference.model:
            raise HTTPException(
                404, f"Reference: \"{data.reference}\" for model \"{data.model}\" is unknown")

    # retrieve the entire history chain
    with database.begin():
        top = (select(History).where(History.uuid == data.reference)
               .cte("history_entries", recursive=True))
        bottom = select(History).join(top, History.uuid == top.c.parent)
        statement = top.union_all(bottom)

        history_list = database.query(statement).all()[::-1]
        history_setup = (database.query(Setup)
                         .filter(Setup.history == history_list[0][1])
                         .one())

        # reconstruct the history object
        history = MODELS[data.model][1]()
        if not data.model.startswith("rehearsal"):
            history.history = [i[-1] for i in history_list]
        else:
            history.history = [i[-1] if "message" not in i[-1] else i[-1]["message"]
                               for i in history_list]

        # generate the agent's response
        manager = MODELS[data.model][0]

        response, history, trace_uuid = manager.step(
            history_setup.context,
            history_setup.question,
            history_setup.answer,
            history,  # type: ignore
            data.message
        )

        # store the history's content
        parent = history_list[-1][1]

        for item in history.history[len(history_list):]:
            if isinstance(history, (RehearsalHistory, RehearsalHeresyHistory, RehearsalRewriteHistory)):
                # item is either a str (child) or dict (agent)
                if isinstance(item, str):
                    entry_data = {"message": item}
                    has_trace = False
                else:
                    entry_data = item
                    has_trace = True
            else:
                # item is all dict (ChatCompletionMessageParam)
                assert not isinstance(item, str)
                entry_data = dict(item)
                has_trace = entry_data["role"] == "assistant"

            database.add(History(
                parent=parent,
                uuid=(parent := uuid_lib.uuid4()),
                trace=trace_uuid if has_trace else None,
                model=data.model,
                data=entry_data
            ))

    return Reply(response=response, reference=parent)


class MessagesPostData(BaseModel):
    model: str = Field(
        description="A model identifier returned from `/models`.")
    reference: uuid_lib.UUID = Field(
        description="A UUID reference specifying the point of conversation to reply to.")


class Messages(BaseModel):
    model: str = Field(
        description="The model used to create the conversation.")
    messages: list[tuple[uuid_lib.UUID, str, str]] = Field(
        description="A list of messages from the conversation (uuid, role, message).")


@api.post("/messages")
async def post_messages(key: Annotated[HTTPAuthorizationCredentials, Depends(security)], data: MessagesPostData, database: Session = Depends(get_database)) -> Messages:
    """Get the messages in the conversation ending in the referred message."""

    if key.scheme != "Bearer" or key.credentials != KEY:
        raise HTTPException(401, "Invalid API key")

    if data.model not in MODELS:
        raise HTTPException(404, f"Model: \"{data.model}\" is unknown")

    with database.begin():
        reference = (database.query(History)
                     .filter(History.uuid == data.reference)
                     .one_or_none())

        if reference is None or data.model != reference.model:
            raise HTTPException(
                404, f"Reference: \"{data.reference}\" for model \"{data.model}\" is unknown")

    # retrieve the entire history chain
    with database.begin():
        top = (select(History).where(History.uuid == data.reference)
               .cte("history_entries", recursive=True))
        bottom = select(History).join(top, History.uuid == top.c.parent)
        statement = top.union_all(bottom)

        history_list = database.query(statement).all()[::-1]

    messages = []

    for _, uuid, _, _, model, message in history_list:
        print(message)
        if model.startswith("rehearsal"):
            if not "message" in message:
                if not ("heresy" in model or "rewrite" in model):
                    messages.append(
                        (uuid, "agent", message["response_response"]))
                else:
                    messages.append(
                        (uuid, "agent", message["response_rewrite"]))
            else:
                messages.append((uuid, "child", message["message"]))
        elif model.startswith("astro"):
            if message["role"] == "assistant":
                messages.append((uuid, "agent", re.search(
                    r"\[Response\](.+)\[End\]", message["content"], re.S)).group(1))
            elif message["role"] == "user":
                messages.append((uuid, "child", message["content"]))
        else:
            if message["role"] == "assistant":
                messages.append((uuid, "agent", message["content"]))
            elif message["role"] == "user":
                messages.append((uuid, "child", message["content"]))

    return Messages(model=data.model, messages=messages)
