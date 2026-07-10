"""Custom exceptions for the langchain-llamacpp-router package."""


class RouterError(Exception):
    """Base exception for router errors."""


class ModelNotAvailableError(RouterError):
    """The requested model is not registered on the router server."""


class ModelColdStartError(RouterError):
    """The model is asleep and must be woken up (cold-start)."""


class NoSuitableModelError(RouterError):
    """No registered model matched the requested tags/constraints."""


class RouterConnectionError(RouterError):
    """Could not connect to the llama.cpp router server."""


class PresetParseError(RouterError):
    """Failed to parse a models-preset INI file."""
