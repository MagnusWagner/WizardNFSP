from copy import deepcopy
import numpy as np

from rlcard.games.wizard import Dealer
from rlcard.games.wizard import Player
from rlcard.games.wizard import Round
from rlcard.games.wizard.utils import COLOR_MAP, getPlayerOrders, cards2list


class WizardGame:
    '''
    TODO: Convert everything into single games (ignore round)
    - The algorithm starts by setting up an environment and agents. Then, the environment is run (with the agents.)
    - The environment calls the action from each agent with state as input.
    - The steps with actions are transferred in the "game"-class which calculates what the actions mean for the state.

    '''
    def __init__(self, allow_step_back=False, num_players=2, num_cards = 2):
        self.actions_in_trick_left = num_players
        self.allow_step_back = allow_step_back
        self.direction = 1
        self.np_random = np.random.RandomState()
        self.num_cards = num_cards # number of cards for that round
        self.num_cards_left = self.num_cards
        self.num_players = num_players
        self.played_cards = []
        self.player_orders = getPlayerOrders(self.num_players)
        self.player_without_color = np.zeros((self.num_players,4))
        self.state = dict()
        self.trick_scores = []
        self.trick_over = False
        self.trump_card = ""
        self.trump_color = ""
        self.wizard_played = False
        


    def configure(self, game_config):
        ''' Specifiy some game specific parameters, such as number of players
        '''
        self.num_players = game_config['game_num_players']




    def init_game(self):
        ''' Initialize players and state

        Returns:
            (tuple): Tuple containing:

                (dict): The first state in one game
                (int): Current player's id
        '''
        # Initalize values
        self.trick_scores = [0 for _ in range(self.num_players)]
        self.player_without_color = np.zeros((self.num_players,4))
        self.played_cards = []
        self.num_cards_left = self.num_cards
        self.is_over = False


        # Initialize a dealer that can deal cards
        self.dealer = Dealer(self.np_random)

        # Initialize n players to play the game
        self.players = [Player(i, self.np_random) for i in range(self.num_players)]

        # Deal n cards to each player to prepare for the game
        for player in self.players:
            self.dealer.deal_cards(player, self.num_cards)


        # flip top card
        self.trump_card = self.flip_top_card()
        self.trump_color = self.trump_card.color

        # Save the hisory for stepping back to the last state.
        self.history = []

        # first trick
        self.initialize_trick()

        # state initialization
        self.state['num_players'] = self.get_num_players()
        self.state['num_cards'] = self.num_cards


        player_id = self.current_player.player_id
        state = self.get_state(player_id)

        return state, player_id

    def initialize_trick(self):
        '''
        Initialize playing a trick in the game. 
        Replaced the round class with a function, as it was too tedious to transfer all parameters to the subclass.
        '''
        self.trick_over = False
        self.color_to_play = None
        self.wizard_played = False
        self.current_player = self.players[self.starting_player.player_id]
        self.current_play_order = self.player_orders[self.starting_player.player_id]
        self.played_cards_in_trick = []
        self.actions_in_trick_left = self.num_players

    def step(self, action):
        ''' Get the next state

        Args:
            action (str): A specific action

        Returns:
            (tuple): Tuple containing:

                (dict): next player's state
                (int): next plater's id
        '''

        if self.allow_step_back:
            # First snapshot the current state
            his_dealer = deepcopy(self.dealer)
            his_state = deepcopy(self.state)
            his_players = deepcopy(self.players)
            self.history.append((his_dealer, his_players, his_state))

        # self.round.proceed_round(self.players, action)
        if self.actions_in_trick_left == self.num_players:
            self.play_first_card(self.current_player,action)
        else:
            self.play_other_card(self.current_player,action)
        
        if self.actions_in_trick_left == 0:
            self.starting_player = self.calculate_new_trick_scores(self.play_order)
            if not self.current_player.hand:
                self.is_over = True
            else: 
                self.initialize_trick()
        player_id = self.current_player.player_id
        state = self.get_state(player_id)
        return state, player_id

    def step_back(self):
        ''' Return to the previous state of the game

        Returns:
            (bool): True if the game steps back successfully
        '''
        if not self.history:
            return False
        self.dealer, self.players, self.state = self.history.pop()
        return True

    def flip_top_card(self):
        ''' Flip the top card of the card pile

        Returns:
            (object of WizardCard): the top card in game

        '''
        top = self.dealer.flip_top_card()
        # self.target = top
        self.played_cards.append(top)
        self.trump_color = top.color
        return top

    def get_state(self, player_id):
        ''' Return player's state

        Args:
            player_id (int): player id

        Returns:
            (dict): The state of the player
        '''
        player = self.players[player_id]
        # Direct information
        self.state['hand'] = cards2list(player.hand)
        self.state['color_to_play'] = self.color_to_play
        self.state['played_cards'] = cards2list(self.played_cards)
        self.state['played_cards_in_trick'] = cards2list(self.played_cards_in_trick)
        self.state['trump_color'] = self.trump_color
        # Meta information
        self.state['legal_actions'] = self.get_legal_actions(self.players, player_id)
        self.state['num_cards_left'] = self.num_cards_left
        self.state['actions_in_trick_left'] = self.actions_in_trick_left
        self.state['player_without_color'] = self.player_without_color
        self.state['wizard_played'] = self.wizard_played
        # needed for later with actual predictions for number of tricks
        # self.state[''] = self.trick_scores = [0 for _ in range(self.num_players)]
        return self.state

    def get_payoffs(self):
        ''' Return the payoffs of the game

        Returns:
            (list): Each entry corresponds to the payoff of one player
        '''
        return self.trick_scores


    def get_num_players(self):
        ''' Return the number of players in Limit Texas Hold'em

        Returns:
            (int): The number of players in the game
        '''
        return self.num_players

    @staticmethod
    def get_num_actions(self):
        ''' Return the number of applicable actions

        Returns:
            (int): The number of actions. There are 60 actions
        '''
        return self.num_cards

    def get_player_id(self):
        ''' Return the current player's id

        Returns:
            (int): current player's id
        '''
        return self.current_player.player_id

    # Extra functions for Wizard 

    def get_legal_actions(self, players, player_id):
        ''' Return the legal actions for current player

        Returns:
            (list): A list of legal actions
        '''
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


    def replace_deck(self):
        ''' Add cards have been played to deck
        '''
        self.dealer.deck.extend(self.played_cards)
        self.dealer.shuffle()
        self.played_cards = []

    def play_first_card(self,player,action):
        '''
        Play the first card in a trick.
        _______________
        Args:
            player (object): object of WizardPlayer
            action (str): string of legal action
        '''
        card_info = action.split('-')
        color = card_info[0]
        trait = card_info[1]
        # Remove index
        remove_index = None
        for index, card in enumerate(player.hand):
            if color == card.color and trait == card.trait:
                remove_index = index
                break
        card = player.hand.pop(remove_index)
        if card.type == "number":
            self.color_to_play = card.color
        # Wizard flag
        if card.type =="wizard":
            self.wizard_played = True

        self.played_cards.append(card)
        self.played_cards_in_trick.append(card)
        self.current_player = self.players[(self.current_player.player_id + self.direction) % self.num_players]
        self.actions_in_trick_left -= 1
        self.num_cards_left -= 1

    def play_other_card(self,player,action):
        '''
        Play a card which is not the first card in a trick.
        _______________
        Args:
            player (object): object of WizardPlayer
            action (str): string of legal action
        '''
        card_info = action.split('-')
        color = card_info[0]
        trait = card_info[1]
        # Remove index
        remove_index = None
        for index, card in enumerate(player.hand):
            if color == card.color and trait == card.trait:
                remove_index = index
                break
        # Check if we have a first played color
        if not self.color_to_play and not self.wizard_played and card.type == "number":
            self.color_to_play = card.color
        # Update info-state about player having color:
        if self.color_to_play and card.type == "number" and self.color_to_play!=card.color:
            self.player_without_color[player.get_player_id(),COLOR_MAP[card.color]]=1
        # Remove card from hand and add it to the played cards
        card = player.hand.pop(remove_index)        
        self.played_cards.append(card)
        self.played_cards_in_trick.append(card)
        # Wizard flag
        if card.type =="wizard":
            self.wizard_played = True
        # Next player
        self.current_player = self.players[(self.current_player.player_id + self.direction) % self.num_players]
        self.actions_in_trick_left -= 1
        self.num_cards_left -= 1

    def calculate_new_trick_scores(self,play_order):
        card_scores = [0 for _ in self.played_cards_in_trick]
        wizard_scorer = 4
        fool_scorer = 4
        for idx, card in enumerate(self.played_cards_in_trick):
            if card.type=="wizard":
                card_scores[idx] = 100 + wizard_scorer
                wizard_scorer -=1
            elif card.type == "fool":
                card_scores[idx] = -100 + fool_scorer
                fool_scorer -=1
            else:
                if self.color_to_play:
                    if card.color == self.trump_color:
                        card_scores[idx] = 15 + int(card.trait)
                    elif card.color == self.color_to_play:
                        card_scores[idx] = int(card.trait)
        trick_winner_idx = play_order[np.argmax(card_scores)]
        self.trick_scores[trick_winner_idx] += 1
        return self.players[trick_winner_idx]