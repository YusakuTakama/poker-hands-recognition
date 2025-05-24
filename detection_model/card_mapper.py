from app.card import Card, Suit, Rank


id_to_category = {
    0: "1h",
    1: "1d",
    2: "1c",
    3: "1s",
    4: "2h",
    5: "2d",
    6: "2c",
    7: "2s",
    8: "3h",
    9: "3d",
    10: "3c",
    11: "3s",
    12: "4h",
    13: "4d",
    14: "4c",
    15: "4s",
    16: "5h",
    17: "5d",
    18: "5c",
    19: "5s",
    20: "6h",
    21: "6d",
    22: "6c",
    23: "6s",
    24: "7h",
    25: "7d",
    26: "7c",
    27: "7s",
    28: "8h",
    29: "8d",
    30: "8c",
    31: "8s",
    32: "9h",
    33: "9d",
    34: "9c",
    35: "9s",
    36: "th",
    37: "td",
    38: "tc",
    39: "ts",
    40: "jh",
    41: "jd",
    42: "jc",
    43: "js",
    44: "qh",
    45: "qd",
    46: "qc",
    47: "qs",
    48: "kh",
    49: "kd",
    50: "kc",
    51: "ks"
}


def create_card_from_string(card_id: int) -> Card:
    """Create a Card object from a string representation."""
    card_str = id_to_category[card_id]

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
