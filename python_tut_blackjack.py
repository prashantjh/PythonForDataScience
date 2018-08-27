# -*- coding: utf-8 -*-
"""
Created on Mon May 14 02:34:22 2018

@author: jhapr
"""

## Import modules ##
import os
os.chdir("E:\Learning\Projects\PythonForDS\code")
import games
import cards

## BlackJack_Card class ##
class BlackjackCard(cards.Card):
    """A Blackjack Card."""
    ACE_VALUE = 1
    
    def get_value(self):
        if self.is_face_up:
            value = BlackjackCard.RANKS.index(self.rank) + 1
            if value > 10:
                value = 10
        else:
            value = None
        return value
    
    value = property(get_value)
    

## Blackjack deck class ##
class BlackjackDeck(cards.Deck):
    """A Blackjack Deck."""
    def populate(self):
        for suit in BlackjackCard.SUITS:
            for rank in BlackjackCard.RANKS:
                self.cards.append(BlackjackCard(rank, suit))
    

## Blackjack hand class ##
class BlackjackHand(cards.Hand):
    """A Blackjack hand"""
    def __init__(self, name):
        super(BlackjackHand, self).__init__()
        self.name = name
    
    def __str__(self):
        rep = self.name + ":\t" + super(BlackjackHand, self).__str__()
        if self.total:
            rep += "(" + str(self.total) + ")"
        return rep
