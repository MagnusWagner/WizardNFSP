from copy import deepcopy
import numpy as np
import xgboost as xgb
import pandas as pd
from rlcard.games.wizard_s_trickpreds import Dealer, Player, Card
from rlcard.games.wizard_s_trickpreds.utils import COLOR_MAP,COLOR_MAP_WORDS, getPlayerOrders, cards2list, encode_cards, convert_hitscores_to_points
from termcolor import colored

class WizardGame:
    '''
    Sets up a whole game of wizard with only 24 cards (all 4 colors with numbers 1 to 4, red is always trump, 4 wizards, 4 fools.)
    Trick Predictions are used to calculate payoffs and Human Agents can make their own Trick Predictions.
    '''
    def __init__(self, allow_step_back=False, num_players=2, num_cards = 2, seed=None, trickpred_model=None, human_ids = None):
        self.actions_in_trick_left = num_players
        self.allow_step_back = allow_step_back
        self.direction = 1
        
        if seed:
            self.seed = seed
        else:
            self.seed = 42
        self.np_random = np.random.RandomState(self.seed)
        self.num_cards = num_cards # number of cards for that round
        self.num_cards_left = self.num_cards
        self.num_players = num_players
        self.played_cards = []
        self.player_orders = getPlayerOrders(self.num_players)
        self.player_without_color = np.zeros((self.num_players,4))
        self.tricks_to_predict = []
        self.state = dict()
        self.trick_scores = []
        self.trick_over = False
        self.trump_color = "r"
        self.wizard_played = False
        self.trickpred_model = trickpred_model
        self.human_ids = human_ids
        


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
        self.tricks_to_predict = []


        # Initialize a dealer that can deal cards
        self.dealer = Dealer(self.np_random)

        # Initialize n players to play the game
        self.players = [Player(i, self.np_random) for i in range(self.num_players)]
        self.starting_player = self.players[np.random.randint(low=0,high=self.num_players)]
        self.current_player = self.starting_player

        # Deal n cards to each player to prepare for the game
        for player in self.players:
            self.dealer.deal_cards(player, self.num_cards)
        self.starting_hands = self.save_starting_hands()
        # Initialize tricks_predicted
        for idx, player in enumerate(self.players):
            if self.human_ids and idx in self.human_ids:
                print("############ Your Starting-Hand #############")
                Card.print_cards(self.starting_hands[idx])
                print("")
                player.tricks_predicted = int(input(f'\n>> How many tricks do you want to make? (Max. of {self.num_cards})": '))
            else:
                shand = pd.DataFrame(np.expand_dims(encode_cards(self.starting_hands[idx]).flatten(),axis=0))
                player.tricks_predicted = int(self.trickpred_model.predict(xgb.DMatrix(shand)))
            self.tricks_to_predict.append(player.tricks_predicted)

        # Save the hisory for stepping back to the last state.
        self.history = []

        # first trick
        self.initialize_trick()

        # state initialization
        self.state['num_players'] = self.get_num_players()
        self.state['num_cards'] = self.num_cards


        player_id = self.current_player.player_id
        state = self.get_state(player_id)
        self.seed +=1
        self.np_random = np.random.RandomState(self.seed)

        return state, player_id

    def initialize_trick(self):
        '''
        Initialize a new trick for the game. All information about the previous trick is reset.
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

        if self.actions_in_trick_left == self.num_players:
            self.play_first_card(self.current_player,action)
        else:
            self.play_other_card(self.current_player,action)
        
        if self.actions_in_trick_left == 0:
            self.num_cards_left -= 1
            self.starting_player = self.calculate_new_trick_scores(self.current_play_order)
            if not self.current_player.hand:
                self.is_over = True
            else: 
                self.initialize_trick()
        player_id = self.current_player.player_id
        state = self.get_state(player_id)
        return state, player_id

    def step_back(self):
        ''' 
        Return to the previous state of the game, not really used by Wizard (and only necessary for CFR-Agents.)

        Returns:
            (bool): True if the game steps back successfully
        '''
        if not self.history:
            return False
        self.dealer, self.players, self.state = self.history.pop()
        return True

    def is_over(self):
        ''' Is the game over?

        Returns:
            (bool): True if game is over
        '''
        return self.is_over

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
        self.state['current_player'] = self.current_player.player_id
        self.state['color_to_play'] = self.color_to_play
        self.state['played_cards'] = cards2list(self.played_cards)
        self.state['played_cards_in_trick'] = cards2list(self.played_cards_in_trick)
        self.state['predicted_tricks'] = self.tricks_to_predict

        # Meta information
        self.state['legal_actions'] = self.get_legal_actions()
        self.state['num_cards_left'] = self.num_cards_left
        self.state['actions_in_trick_left'] = self.actions_in_trick_left
        self.state['player_without_color'] = self.player_without_color
        self.state['trick_scores'] = self.trick_scores
        self.state['wizard_played'] = self.wizard_played
        self.state['tricks_left_to_get'] = player.tricks_predicted-self.trick_scores[player_id]

        return self.state

    def get_payoffs(self):
        ''' Return the payoffs of the game

        As this version is about trick predictions, the following algorithm is used:
            - No. of final tricks as predicted: 20 Points + 10 Points per correct trick.
            - No. of final tricks WRONG: -10 Points per trick that was wrong.

        Returns:
            (list): Each entry corresponds to the payoff of one player
        '''
        
        hit_scores = (np.array(self.tricks_to_predict)-np.array(self.trick_scores)).astype(np.int32)
        payoffs = convert_hitscores_to_points(hit_scores,self.trick_scores)

        return payoffs


    def get_num_players(self):
        ''' Return the number of players in Wizard

        Returns:
            (int): The number of players in the game
        '''
        return self.num_players

    def get_num_actions(self):
        ''' Return the number of applicable actions

        Returns:
            (int): The number of actions. There are 60 actions
        '''
        return 24

    def get_player_id(self):
        ''' Return the current player's id

        Returns:
            (int): current player's id
        '''
        return self.current_player.player_id

    # Extra functions for Wizard 

    def get_legal_actions(self):
        ''' Return the legal actions for current player

        Returns:
            (list): A list of legal actions
        '''
        player_id = self.current_player.player_id
        legal_actions = []
        hand = self.players[player_id].hand
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

    def calculate_new_trick_scores(self,play_order):
        '''
        This calculation is based on the Wizard rules. The trick_scores are updated regarding the person who won the trick.
        Some small sample rules:
            - First Wizard in a trick always wins the trick.
            - Trump cards (red) are always higher than other coloured number cards.
            - Higher numbers of the color_to_play get the trick if no trump or wizard was played.
        
        return: Player who won the trick.
        '''
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
    
    def save_starting_hands(self):
        '''
        Returns the starting_hands in list format.
        '''
        return_list = []
        for player in self.players:
            return_list.append(cards2list(player.hand))
        return return_list

    def get_starting_hands(self):
        return self.starting_hands