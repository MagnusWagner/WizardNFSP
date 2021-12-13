from termcolor import colored

class WizardCard:

    info = {'type':  ['number', 'fool', 'wizard'],
            'color': ['r', 'g', 'b', 'y'],
            'trait': ['fool', '1', '2', '3', '4','wizard']
            }

    def __init__(self, card_type, color, trait):
        ''' Initialize the class of WizardCard

        Args:
            card_type (str): The type of card
            color (str): The color of card
            trait (str): The trait of card
        '''
        self.type = card_type
        self.color = color
        self.trait = trait
        self.str = self.get_str()

    def get_str(self):
        ''' Get the string representation of card

        Return:
            (str): The string of card's color and trait
        '''
        return self.color + '-' + self.trait


    @staticmethod
    def print_cards(cards):
        ''' Print out card in a nice form

        Args:
            card (str or list): The string form or a list of a UNO card
            wild_color (boolean): True if assign collor to wild cards
        '''
        if isinstance(cards, str):
            cards = [cards]
        for i, card in enumerate(cards):
            color, trait = card.split('-')
            if trait == 'wizard':
                trait = 'Wizard'
            elif trait == 'fool':
                trait = 'Fool'
            if trait == 'Wizard' or trait == 'Fool':
                print(trait, end='')
            elif color == 'r':
                print(colored(trait, 'red'), end='')
            elif color == 'g':
                print(colored(trait, 'green'), end='')
            elif color == 'b':
                print(colored(trait, 'blue'), end='')
            elif color == 'y':
                print(colored(trait, 'yellow'), end='')

            if i < len(cards) - 1:
                print(', ', end='')
