from dataclasses import dataclass
from enum import Enum


class Suit(Enum):
    """An enumeration representing the suits of a standard deck of cards."""
    HEARTS = "Hearts"
    DIAMONDS = "Diamonds"
    CLUBS = "Clubs"
    SPADES = "Spades"


class Rank(Enum):
    """An enumeration representing the ranks of a standard deck of cards."""
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "t"
    JACK = "j"
    QUEEN = "q"
    KING = "k"
    ACE = "a"


@dataclass
class Card:
    """A class representing a card in a deck of cards."""

    suit: Suit
    rank: Rank

    def __str__(self) -> str:
        return f"{self.rank} of {self.suit}"
