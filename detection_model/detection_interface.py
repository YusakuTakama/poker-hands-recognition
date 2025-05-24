from typing import List
from app.card import Card

from .model_factory import build_model


class CardDetectorInterface:
    def __init__(self, model_name, model_path: str):
        self.model = build_model(model_name, model_path)

    def detect(self, image) -> List[Card]:
        """Detect cards in the given image and return a list of Card objects."""
        raise NotImplementedError("This method should be implemented by subclasses.")
