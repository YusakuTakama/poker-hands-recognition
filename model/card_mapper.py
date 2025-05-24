from app.card import Card, Suit, Rank

def create_card_from_string(card_str: str) -> Card:
    """Create a Card object from a string representation."""
    suit_map = {
        "h": Suit.HEARTS,
        "d": Suit.DIAMONDS,
        "c": Suit.CLUBS,
        "s": Suit.SPADES
    }

    rank_map = {
        "1": Rank.ACE,
        "2": Rank.TWO,
        "3": Rank.THREE,
        "4": Rank.FOUR,
        "5": Rank.FIVE,
        "6": Rank.SIX,
        "7": Rank.SEVEN,
        "8": Rank.EIGHT,
        "9": Rank.NINE,
        "t": Rank.TEN,
        "j": Rank.JACK,
        "q": Rank.QUEEN,
        "k": Rank.KING
    }

    suit = suit_map[card_str[1]]
    rank = rank_map[card_str[0].lower()]

    return Card(suit, rank)
