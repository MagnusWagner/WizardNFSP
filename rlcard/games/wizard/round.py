from rlcard.games.wizard.card import WizardCard
from rlcard.games.wizard.utils import cards2list, WIZARD, FOOL, TRAIT_MAP, getPlayerOrders
import numpy as np

class WizardRound:

    def __init__(self, dealer, players, np_random, trump_color, played_cards, starting_player):
        ''' Initialize the round class

        One round = One trick 

        Args:
            dealer (object): the object of WizardDealer
            num_players (int): the number of players in game
        '''
        self.np_random = np_random
        self.dealer = dealer
        self.first_card = None
        self.color_to_play = None
        self.current_player = starting_player
        self.starting_player = starting_player
        self.players = players
        self.direction = 1
        self.played_cards = played_cards
        self.played_cards_in_trick = []
        self.trick_winner = None
        self.trump_color = trump_color



    def proceed_round(self, players, action):
        ''' Call other Classes's functions to keep one round running

        Args:
            player (object): object of WizardPlayer
            action (str): string of legal action
        '''
        card_info = action.split('-')
        color = card_info[0]
        trait = card_info[1]
        # remove correspongding card
        remove_index = None
        for index, card in enumerate(player.hand):
            if color == card.color and trait == card.trait:
                remove_index = index
                break
        card = player.hand.pop(remove_index)
        '''
        TODO: Check if round is over when all players have no cards!
        '''

        self.played_cards.append(card)
        self.played_cards_in_trick.append(card)
        self.current_player = (self.current_player + self.direction) % self.num_players
        # self.target = card
    
    def play_trick(self, players):
        self.current_player = players[self.starting_player]
        play_order = self.player_orders[self.starting_player]
        self.played_cards_in_trick = []
        self.color_to_play = None
        '''
        Insert all players playing one card. Await action from player.
        '''
        self.first_card = play_first_card(self.current_player,action)
        if self.first_card.trait != "wizard" or self.first_card.trait != "fool":
            self.color_to_play = self.first_card.color
        for i in range(len(self.num_players)-1):
            play_other_card(self.current_player,action)

        if not self.current_player.hand:
            self.is_over = True

        self.trick_scores = 

    def get_legal_actions(self, players, player_id):
        legal_actions = []
        hand = players[player_id].hand
        if not self.color_to_play:
            legal_actions+=[card.str for card in hand]
        else:
            ### Check if player can serve
            has_suited_cards = any([card.type == "number" and card.color == self.color_to_play for card in hand])
            ### Player can serve
            if has_suited_cards:
                for card in hand:
                    if card.type == 'wizard' or card.type == "fool" or card.color == self.color_to_play:
                        legal_actions.append(card.str)
            ### Player cannot serve
            else:
                legal_actions+=[card.str for card in hand]
        return legal_actions

    def get_state(self, players, player_id):
        ''' Get player's state

        Args:
            players (list): The list of UnoPlayer
            player_id (int): The id of the player
        '''
        state = {}
        player = players[player_id]
        #### Direct information
        state['hand'] = cards2list(player.hand)
        state['played_cards_in_trick'] = cards2list(self.played_cards_in_trick)
        state['played_cards'] = cards2list(self.played_cards)
        #### Meta information
        state['legal_actions'] = self.get_legal_actions(players, player_id)
        state['color_to_play'] = self.color_to_play

        return state

    def replace_deck(self):
        ''' Add cards have been played to deck
        '''
        self.dealer.deck.extend(self.played_cards)
        self.dealer.shuffle()
        self.played_cards = []




