"""Domain exceptions, mapped to HTTP status codes by the API layer."""

from __future__ import annotations


class EvalForgeError(Exception):
    """Base class for expected, user-facing errors."""

    code = "evalforge_error"


class NotFoundError(EvalForgeError):
    """A referenced entity does not exist."""

    code = "not_found"

    def __init__(self, entity: str, entity_id: str) -> None:
        super().__init__(f"{entity} '{entity_id}' was not found")
        self.entity = entity
        self.entity_id = entity_id


class ConflictError(EvalForgeError):
    """The request conflicts with the current state."""

    code = "conflict"


class InvalidRequestError(EvalForgeError):
    """The request is well-formed but cannot be executed."""

    code = "invalid_request"
