from dataclasses import dataclass
from enum import Enum, IntEnum


class Suit(Enum):
    """An enumeration representing the suits of a standard deck of cards."""
    HEARTS = "Heart"
    DIAMONDS = "Diamond"
    CLUBS = "Club"
    SPADES = "Spade"


class Rank(IntEnum):
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
        rank = self.rank.value
        if rank == 11:
            rank = "Jack"
        elif rank == 12:
            rank = "Queen"
        elif rank == 13:
            rank = "King"
        elif rank == 14:
            rank = "Ace"
        else:
            rank = str(rank)

        return f"{rank} of {self.suit}"

    def omittion(self) -> str:
        """Return a string representation of the card for model input."""
        if self.rank <= Rank.TEN:
            return f"{self.rank.value}{self.suit.name[0].lower()}"
        else:
            rank_str = {Rank.JACK: 'J', Rank.QUEEN: 'Q', Rank.KING: 'K', Rank.ACE: 'A'}
            return f"{rank_str[self.rank]}{self.suit.name[0].lower()}"
