from dataclasses import dataclass
from enum import Enum


class Suit(Enum):
    """An enumeration representing the suits of a standard deck of cards."""
    HEARTS = "h"
    DIAMONDS = "d"
    CLUBS = "c"
    SPADES = "s"


class Rank(Enum):
    """An enumeration representing the ranks of a standard deck of cards."""
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


@dataclass
class Card:
    """A class representing a card in a deck of cards."""

    suit: Suit
    rank: Rank

    def __str__(self) -> str:
        return f"{self.rank} of {self.suit}"
