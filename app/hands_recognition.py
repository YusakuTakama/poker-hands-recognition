from enum import Enum
from typing import List, Optional, Tuple

from .card import Card, Rank, Suit


class HandRank(Enum):
    HIGH_CARD = 1
    ONE_PAIR = 2
    TWO_PAIR = 3
    THREE_OF_A_KIND = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    FOUR_OF_A_KIND = 8
    STRAIGHT_FLUSH = 9
    ROYAL_FLUSH = 10


class HandsRecognizer:
    """Evaluate a best 5-card poker hand from 7 cards."""

    def __init__(self, cards: List[Card]):
        assert len(cards) == 7, "Exactly 7 cards are required"
        self.cards = cards
        self.ranks = [card.rank.value for card in cards]
        self.suits = [card.suit for card in cards]
        # count occurrences
        self.suits_count = {s: self.suits.count(s) for s in Suit}
        self.ranks_count = {r: self.ranks.count(r.value) for r in Rank}

    def evaluate(self) -> Tuple[HandRank, List[Card]]:
        """Return the best hand rank and the corresponding 5 cards."""
        # Check in descending order
        if rf := self._straight_flush(royal=True):
            return HandRank.ROYAL_FLUSH, rf
        if sf := self._straight_flush():
            return HandRank.STRAIGHT_FLUSH, sf
        if fk := self._of_a_kind(4):
            return HandRank.FOUR_OF_A_KIND, fk
        if fh := self._full_house():
            return HandRank.FULL_HOUSE, fh
        if fl := self._flush():
            return HandRank.FLUSH, fl
        if st := self._straight():
            return HandRank.STRAIGHT, st
        if tk := self._of_a_kind(3):
            return HandRank.THREE_OF_A_KIND, tk
        if tp := self._two_pair():
            return HandRank.TWO_PAIR, tp
        if op := self._of_a_kind(2):
            return HandRank.ONE_PAIR, op
        return HandRank.HIGH_CARD, self._high_card()

    def _straight_flush(self, royal: bool = False) -> Optional[List[Card]]:
        # Find any suit with >=5 cards
        for suit, count in self.suits_count.items():
            if count >= 5:
                cards = [c for c in self.cards if c.suit == suit]
                seq = self._sequence([c.rank.value for c in cards])
                if seq:
                    # royal flush: highest sequence must be 10-14
                    if royal and seq[-1] == Rank.ACE.value and seq[0] == Rank.TEN.value:
                        return [c for c in cards if c.rank.value in seq]
                    if not royal and not (royal is True and seq[0] == Rank.TEN.value):
                        return [c for c in cards if c.rank.value in seq]
        return None

    def _of_a_kind(self, n: int) -> Optional[List[Card]]:
        # returns n-of-a-kind plus highest kickers
        kinds = [r for r, cnt in self.ranks_count.items() if cnt == n]
        if not kinds:
            return None
        best_rank = max(kinds)
        group = [c for c in self.cards if c.rank.value == best_rank]
        # find kickers
        kickers = sorted((c for c in self.cards if c.rank.value != best_rank), key=lambda c: c.rank.value, reverse=True)
        return group + kickers[:5-n]

    def _full_house(self) -> Optional[List[Card]]:
        triples = sorted([r for r, cnt in self.ranks_count.items() if cnt >= 3], reverse=True)
        pairs = sorted([r for r, cnt in self.ranks_count.items() if cnt >= 2 and r not in triples], reverse=True)
        if len(triples) >= 1 and (len(pairs) >= 1 or len(triples) > 1):
            three_rank = triples[0]
            pair_rank = pairs[0] if pairs else triples[1]
            three_cards = [c for c in self.cards if c.rank.value == three_rank][:3]
            pair_cards = [c for c in self.cards if c.rank.value == pair_rank][:2]
            return three_cards + pair_cards
        return None

    def _flush(self) -> Optional[List[Card]]:
        for suit, cnt in self.suits_count.items():
            if cnt >= 5:
                cards = sorted([c for c in self.cards if c.suit == suit], key=lambda c: c.rank.value, reverse=True)
                return cards[:5]
        return None

    def _straight(self) -> Optional[List[Card]]:
        seq = self._sequence(self.ranks)
        if not seq:
            return None
        # select cards matching the sequence
        return [next(c for c in sorted(self.cards, key=lambda c: c.rank.value) if c.rank.value == r) for r in seq]

    def _two_pair(self) -> Optional[List[Card]]:
        pairs = sorted([r for r, cnt in self.ranks_count.items() if cnt >= 2], reverse=True)
        if len(pairs) >= 2:
            first, second = pairs[0], pairs[1]
            cards = [c for c in self.cards if c.rank.value in (first, second)]
            kicker = max((c for c in self.cards if c.rank.value not in (first, second)), key=lambda c: c.rank.value)
            return cards[:4] + [kicker]
        return None

    def _high_card(self) -> List[Card]:
        return sorted(self.cards, key=lambda c: c.rank.value, reverse=True)[:5]

    @staticmethod
    def _sequence(values: List[int]) -> Optional[List[int]]:
        """Find highest 5-card straight in the list of rank values."""
        u = set(values)
        # Ace can be low
        if Rank.ACE.value in u:
            u.add(1)
        runs = []
        for v in sorted(u):
            if not runs or v != runs[-1] + 1:
                runs = [v]
            else:
                runs.append(v)
            if len(runs) >= 5:
                best = runs[-5:]
        return best if 'best' in locals() else None
