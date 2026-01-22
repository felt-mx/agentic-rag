from abc import ABC, abstractmethod
from pathlib import Path
from core.models.document import Document


class BaseParser(ABC):
    @abstractmethod
    def parse(self, file_path: Path) -> Document:
        pass

    @abstractmethod
    def supports(self, file_path: Path) -> bool:
        pass
