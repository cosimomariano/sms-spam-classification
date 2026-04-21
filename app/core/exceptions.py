from __future__ import annotations


class ApiError(Exception):
    def __init__(self, detail: str, error_code: str, status_code: int = 400) -> None:
        super().__init__(detail)
        self.detail = detail
        self.error_code = error_code
        self.status_code = status_code


class ModelNotAvailableError(ApiError):
    def __init__(self, detail: str = 'No trained model is available.') -> None:
        super().__init__(detail=detail, error_code='MODEL_NOT_AVAILABLE', status_code=404)
