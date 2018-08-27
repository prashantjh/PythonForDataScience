# =============================================================================
# Class Example
#     A Critter Program
# 
# =============================================================================



## A class with all the methods for Name, Mood, Eat & Play
class Critter(object):
    """A virtual pet"""
    
    # A constructor method
    def __init__(self, name, hunger = 0, boredom = 0):
        self.name = name
        self.hunger = hunger
        self.boredom = boredom
    
    def __pass_time(self):
        self.hunger += 1
        self.boredom += 1
    
    def __get_mood(self):
        unhappiness = self.hunger + self.boredom
        if unhappiness < 5:
            mood = 'happy'
        elif unhappiness >=5 and unhappiness < 10:
            mood = 'okay'
        elif unhappiness >=10 and unhappiness < 15:
            mood = 'frustrated'
        else:
            mood = 'mad'
        return mood
    
    mood = property(__get_mood)
    
    # Talk to the critter
    def talk(self):
        print("I'm", self.name, "and I feel", self.mood)
        self.__pass_time()
    
    def eat(self, food=4):
        print("Burpppp!!")
        self.hunger -= food
        if self.hunger < 0:
            self.hunger = 0
        self.__pass_time()
    
    def play(self, fun = 3):
        print("Yayyyy!!")
        self.boredom -= fun
        if self.boredom < 0:
            self.boredom = 0
        self.__pass_time()
        


def main():
    crit_name = input("What do you want to name your Critter?")
    crit = Critter(crit_name)
    
    choice = None
    while choice != '0':
        print("""Critter Caretaker:
            0: Quit
            1: Listen to you Critter
            2: Feed your Critter
            3: Play with your critter
            """)
        
        choice = input("Enter your choice: ")
        if choice == '0':
            print("Good Bye")
        elif choice == '1':
            crit.talk()
        elif choice == '2':
            crit.eat()
        elif choice == '3':
            crit.play()
        else:
            print("Sorry, the choice is invalid")


main()