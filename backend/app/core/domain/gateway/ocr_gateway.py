from abc import ABC, abstractmethod
from typing import Dict, Any, List


class OcrGateway(ABC):
    """Abstract interface for Optical Character Recognition (OCR) engines."""

    @abstractmethod
    async def extract_text(self, document_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Extracts plain text and token-level layout bounding boxes from raw file bytes.

        Args:
            document_bytes: Binary contents of PDF or image.
            filename: Name of the file for extension inference.

        Returns:
            Dict containing 'text' (str), 'confidence' (float), and 'blocks' (List[Dict]).
        """
        pass
