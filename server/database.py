from __future__ import annotations

import os
import uuid as uuid_lib
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    ...


class Setup(Base):
    __tablename__ = "setup"

    id: Mapped[int] = mapped_column(primary_key=True)

    history: Mapped[uuid_lib.UUID] = mapped_column(
        UUID, nullable=False, unique=True, index=True)

    context: Mapped[str]
    question: Mapped[str]
    answer: Mapped[str]


class History(Base):
    __tablename__ = "history"

    id: Mapped[int] = mapped_column(primary_key=True)

    uuid: Mapped[uuid_lib.UUID] = mapped_column(
        UUID, nullable=False, unique=True, index=True)
    trace: Mapped[uuid_lib.UUID] = mapped_column(
        UUID, nullable=True, unique=True)
    parent: Mapped[Optional[uuid_lib.UUID]] = mapped_column(UUID)

    model: Mapped[str] = mapped_column(nullable=False)
    data: Mapped[dict] = mapped_column(JSONB, nullable=False)


DATABASE_URL = os.getenv("SQLALCHEMY_DATABASE_URL")

assert DATABASE_URL is not None, "Must specify SQLALCHEMY_DATABASE_URL"

engine = create_engine(DATABASE_URL)

Session = sessionmaker(engine)

Base.metadata.create_all(engine)


def get_database():
    session = Session()

    return session
