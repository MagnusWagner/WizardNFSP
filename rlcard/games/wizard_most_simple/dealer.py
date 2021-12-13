
from rlcard.games.wizard_most_simple.utils import init_deck


class WizardDealer:
    ''' Initialize a Wizard dealer class
    '''
    def __init__(self, np_random):
        self.np_random = np_random
        self.deck = init_deck()
        self.shuffle()

    def shuffle(self):
        ''' Shuffle the deck
        '''
        self.np_random.shuffle(self.deck)

    def deal_cards(self, player, num):
        ''' Deal some cards from deck to one player

        Args:
            player (object): The object of WizardPlayer
            num (int): The number of cards to be dealed
        '''
        for _ in range(num):
            player.hand.append(self.deck.pop())
