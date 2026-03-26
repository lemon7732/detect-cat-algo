"""Project-specific exception hierarchy."""


class CatRescueAIError(Exception):
    """Base exception for the project."""


class DependencyNotAvailableError(CatRescueAIError):
    """Raised when an optional runtime dependency is missing."""


class InvalidImageError(CatRescueAIError):
    """Raised when an image cannot be decoded or processed."""


class NonCatImageError(CatRescueAIError):
    """Raised when an image is classified as not-cat."""


class CatFaceNotFoundError(CatRescueAIError):
    """Raised when no cat face can be detected."""


class LandmarkPredictionError(CatRescueAIError):
    """Raised when landmark prediction fails."""


class GalleryNotBuiltError(CatRescueAIError):
    """Raised when gallery files are missing or empty."""


class UnknownCatError(CatRescueAIError):
    """Raised when no gallery entry passes the matching threshold."""


class ConfigError(CatRescueAIError):
    """Raised when configuration is invalid."""
