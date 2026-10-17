"""Explicit geometry failures, suitable for recording against a detection key."""


class GeometryError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
