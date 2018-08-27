# -*- coding: utf-8 -*-
"""
Created on Sun May 13 12:39:51 2018

@author: jhapr
"""
#######################################################################
#               CLASSES                                     
#######################################################################

# Simple Critter
# Demonstrates a basic class and object

class Critter(object):
    """A virtual pet"""
    def talk(self):
        print("Hi. I'm an instance of class Critter.")
        

# main
crit = Critter()
crit.talk()


## Constructor Critter ##
# Demonstrates Constructors

class Critter(object):
    """A virtual pet"""
    def __init__(self):
        print("A new critter has been born")
        
    def talk(self):
        print("Hi. I'm an instance of class Critter")
        

crit1 = Critter()
crit2 = Critter()

crit1.talk()
crit2.talk()


## Class attributes ##
# Demonstrates creating and accessing object attributes

class Critter:
    """A virtual pet"""
    def __init__(self, name):   # attribute initialization using constructor
        print("A new critter has been born!")
        self.name = name
        
    def __str__(self):
        rep = "Critter object\n"
        rep += "name: " + self.name + "\n"
        return rep

    def talk(self):
        print("Hi. I'm", self.name, "\n")
        


crit1 = Critter("Pookie")
crit1.talk()
print("Printing crit1")
print(crit1)

crit2 = Critter("Chewie")
crit.talk()
crit2.name
print("Printing crit2")
print(crit2)

crit1.name = "Pooh"
crit1.name
crit1.talk()


## Class attributes and Static Methods ##
class Critter:
    """A virtual pet"""
    total = 0
    
    def status():
        print("The total no. of critters is: ", Critter.total)
        
    staticmethod(status)
    def __init__(self, name):
        print("A critter has been born.")
        self.name = name
        Critter.total +=1
        

print("Accessing the class attribute - Critter.total:", Critter.total)

print("Creating Critters:")
crit1 = Critter("Critter 1")
crit2 = Critter("Critter 2")
crit3 = Critter("Critter 3")

Critter.status()

print("Accessing the class attribute through an object:", crit1.total)


## Private Attributes - Encapsulation ##
class Critter:
    """A virtual pet"""
    # Constructor
    def __init__(self, name, mood):
        print("A new critter has been born!")
        self.name = name        # public attribute
        self.__mood = mood      # private attribute
        
    # Accessing private attribute
    def talk(self):
        print("I'm", self.name)
        print("Right now I feel", self.__mood)
        
    def __private_method(self):
        print("This is a private method")
    
    def public_method(self):
        print("This is a public method. It can access private method. See below:")
        self.__private_method()

        
crit = Critter(name = "Pooh", mood = "Neutral")
print(crit.mood)    # Not gonna happen
print(crit.__mood)  # LOL
print(crit._Critter__mood)  # A hack

crit.talk() # better way
crit.public_method()


## Get/Set Methods ##
class Critter:
    """A virtual pet"""
    # Constructor
    def __init__(self, name):
        print("A new critter has been born!")
        self.__name = name        # public attribute
        
    def get_name(self):
        return self.__name
    
    def set_name(self, new_name):
        if new_name == "":
            print("New name cannot be empty string")
        else:
            self.__name = new_name
            print("Name change successful")
            
    name = property(get_name, set_name)
    def talk(self):
        print("Hi, I'm", self.name)
            
    


crit = Critter("Pooh")
print(crit.get_name())
crit.set_name("")
crit.set_name("Chiku")

print(crit.name)        # calls get method via property automatically
crit.name = "Sammy"     # Calls set method via property automatically

crit.talk()




#######################################################################
#               OOP                                     
#######################################################################

## Alien Blaster
# Demonstrates object interaction
class Player:
    """A player in a shooter game"""
    def blast(self, enemy):
        print("The player blasts an enemy.")
        enemy.die()
        
class Alien:
    """An alien in a shoother game"""
    def die(self):
        print("The alien gasps and says, \n'Oh, this is it. \n\
              This is the big one.\n\
              Yes, it's getting dark now. \n\
              Tell my 1.6 MM larvae that I loved them\n\
              Good-bye, cruel universe.'")
        
print("Death of an Alien")
hero = Player()
invader = Alien()
hero.blast(invader)


## Combining Object
# Playing cards
class Card:
    """A playing card."""
    RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
    SUITS = ["c", "d", "h", "s"]
    
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit
        
    def __str__(self):
        rep = self.rank + self.suit
        return rep
    
# Hand class
class Hand:
    """A hand of playing cards."""
    def __init__(self):
        self.cards = []
        
    def __str__(self):
        if self.cards:
            rep = ""
            for card in self.cards:
                rep +=str(card) + " "
        else:
            rep = "<empty>"
        return rep
    
    def clear(self):
        self.cards = []
        
    def add(self, card):
        self.cards.append(card)
        
    def give(self, card, other_hand):
        self.cards.remove(card)
        other_hand.add(card)

card1 = Card(rank = "A", suit = "c")
print("Printing a card object", card1)

card2 = Card("2", "c")
card3 = Card("3", "c")
card4 = Card("4", "c")
card5 = Card("5", "c")

print("Card2:", card2, " Card3:", card3, " Card4:", card4)


# Combining card using a Hand object
my_hand = Hand()
print(my_hand)
my_hand.add(card1)
my_hand.add(card2)
my_hand.add(card3)
my_hand.add(card4)
my_hand.add(card5)
print("My hand after adding 5 cards: ", my_hand)

your_hand = Hand()
# Transfer first two hand from my hand to your hand
my_hand.give(card1, your_hand)
my_hand.give(card2, your_hand)
print("My hand after giving first two hands:", my_hand)
print("Your hand after receiving two cards:", your_hand)

# Clearing hand
my_hand.clear()
print("My hand after card clear:", my_hand)


## Inheritance ##
# Playing cards 2.0
# Card and Hand class remain the same as above

class Deck(Hand):       # Inheritance
    """A deck of playing cards."""
    # Extending a derived class
    def populate(self):
        for suit in Card.SUITS:
            for rank in Card.RANKS:
                self.add(Card(rank, suit))
            
    def shuffle(self):
        import random
        random.shuffle(self.cards)
        
    def deal(self, hands, per_hand = 1):
        for rounds in range(per_hand):        
            for hand in hands:
                if self.cards:
                    top_card = self.cards[0]
                    self.give(top_card, hand)
                else:
                    print("Can't continue deal. Out of Cards!!")


# Using the Derived class
deck1 = Deck()
print("New deck:", deck1)

deck1.populate()    # Populating the deck w/ 52 cards    
print("Populated Deck:", deck1)

deck1.shuffle()     # Shuffling the deck
print("Shuffled Deck:", deck1)


# Dealing
my_hand = Hand()
your_hand = Hand()
hands = [my_hand, your_hand]
deck1.deal(hands, per_hand= 5)

# After dealing 5 cards per hand:
print("My hand: ", my_hand)
print("Your hand: ", your_hand)
print("Deck: ", deck1)



## Overriding ##
# Same Card class

class UnprintableCard(Card):   # Overriding base class methods
    """A card that won't reveal its rank or suit when printed."""
    def __str__(self):
        return "<unprintable>"

# Class for face up with constructor
class PositionableCard(Card):
    """A Card that can be face up or down."""
    def __init__(self, rank, suit, face_up = True):
        super(PositionableCard, self).__init__(rank, suit)
        self.is_face_up = face_up
    
    def __str__(self):
        if self.is_face_up:
            rep = super(PositionableCard, self).__str__()
        else:
            rep = "XX"
        return rep
    
    def flip(self):
        self.is_face_up = not self.is_face_up

card1 = Card("A", "c")
card2 = UnprintableCard("A", "d")
card3 = PositionableCard("A", "h")

print("Printing a card object: ", card1)
print("Printing an unprintable card: ", card2)
print("Printing a Positionable card: ", card3)

print("Flipping a positionable card: ")
card3.flip()
print("Printing after flipping: ", card3)


## Polymorphism ##




