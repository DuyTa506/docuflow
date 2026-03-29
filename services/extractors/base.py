"""
Base extractor ABC.

All concrete extractors inherit from BaseExtractor and must implement extract().
"""
from abc import ABC, abstractmethod
from typing import List

from core.models import UnifiedElement


class BaseExtractor(ABC):
    """Abstract base for all document extractors."""

    @abstractmethod
    def extract(self, file_path: str) -> List[UnifiedElement]:
        """
        Extract content from a file and return a flat list of UnifiedElements.

        Args:
            file_path: Absolute path to the source file.

        Returns:
            Ordered list of UnifiedElement instances.
        """
