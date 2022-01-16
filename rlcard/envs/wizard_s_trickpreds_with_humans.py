import numpy as np
from collections import OrderedDict
import xgboost as xgb
from rlcard.envs import Env
from rlcard.games.wizard_s_trickpreds.game import WizardGame
from rlcard.games.wizard_s_trickpreds.utils import encode_cards, encode_color, ACTION_SPACE, ACTION_LIST, cards2list

DEFAULT_GAME_CONFIG = {
        'game_num_players': 2,
        'game_num_cards': 5,
        'seed':42,
        }

class WizardEnv(Env):

    def __init__(self, config):
        self.name = 'wizard'
        if config:
            self.config = DEFAULT_GAME_CONFIG
            for key in config:
                self.config[key]=config[key]
        else:
            self.config = DEFAULT_GAME_CONFIG
        self.seed = self.config["seed"]
        self.model_xgb = xgb.Booster()
        self.model_xgb.load_model(f'./rlcard/games/wizard_s_trickpreds/xgb_models/{self.config["game_num_players"]}P{self.config["game_num_cards"]}C.json')
        self.human_ids = list(np.arange(config['no_human_players']))
        self.game = WizardGame(num_players = self.config["game_num_players"],num_cards = self.config["game_num_cards"], seed=self.seed, trickpred_model=self.model_xgb, human_ids=self.human_ids)
        super().__init__(config)
        self.state_shape = [[96+5+1+1+4*self.num_players+1+1] for _ in range(self.num_players)]  
        self.action_shape = [None for _ in range(self.num_players)]

    def get_encoded_starting_hands(self):
        self.starting_hands=self.game.get_starting_hands()
        self.encoded_starting_hands = np.array([encode_cards(hand).flatten() for hand in self.starting_hands])
        return self.encoded_starting_hands

    def _extract_state(self, state):
        '''
        Extract state and add:
            - "hand": 4x6 matrix of cards in hand
            - "played_cards": 4x6 matrix of cards already played
            - "played_cards_in_trick": 4x6 matrix of cards already played this trick
            - "legal_actions": 4x6 matrix of cards that can be played by agent
            - "color_to_play": 5x1 one-hot encoded list for color to play or None
            - "num_cards_left": int: number of cards left to play
            - "actions_in_trick_left": int: number of cards left to play in trick
            - 'player_without_color': num_players X 4 one-hot encoded matrix to see which player is missing a color already
            - "wizard_played": 1 or 0

        '''
        obs = np.zeros((4, 4, 6), dtype=int)
        obs[0] = encode_cards(state['hand'])
        obs[1] = encode_cards(state['played_cards'])
        obs[2] = encode_cards(state['played_cards_in_trick'])
        obs[3] = encode_cards(state['legal_actions'])
        obs=obs.flatten()
        # Trump color encoded in array of 4.
        color_to_play = encode_color(state["color_to_play"],no_color_possible=True)
        # How many cards are left? Encoded in Array of game_num_cards.
        num_cards_left_array = np.array([state['num_cards_left']])
        # num_cards_left_array = np.zeros(self.config.get("game_num_cards"))
        # num_cards_left_array[(state['num_cards_left']-1)]=1
        # How many actions are left in trick? Encoded in Array of game_num_players.
        num_actions_left_in_trick_array = np.array([state['actions_in_trick_left']])
        # num_actions_left_in_trick_array = np.zeros(self.config.get("game_num_players"))
        # num_actions_left_in_trick_array[(state['actions_in_trick_left']-1)]=1
        # Which players have which color.
        players_without_color_array = state['player_without_color'].flatten()
        # Was a Wizard already played?
        wizard_played = np.array([int(state['wizard_played'])])
        # Add trick predictions here.
        tricks_left_to_get = np.array([state['tricks_left_to_get']])    
        #
        #
        #

        obs = np.concatenate((obs,color_to_play,num_cards_left_array,num_actions_left_in_trick_array,players_without_color_array,wizard_played,tricks_left_to_get))
        legal_action_ids = self._get_legal_actions()
        extracted_state = {'obs': obs, 'legal_actions': legal_action_ids}
        extracted_state['raw_obs'] = state
        extracted_state['raw_legal_actions'] = [a for a in state['legal_actions']]
        extracted_state['action_record'] = self.action_recorder
        return extracted_state

    def get_payoffs(self):

        return np.array(self.game.get_payoffs())

    def _decode_action(self, action_id):
        '''
        Only run actions that are legal. If an action is illegal, do a random move.
        '''
        legal_ids = self._get_legal_actions()
        if action_id in legal_ids:
            return ACTION_LIST[action_id]
        # if (len(self.game.dealer.deck) + len(self.game.round.played_cards)) > 17:
        #    return ACTION_LIST[60]
        return ACTION_LIST[np.random.choice(legal_ids)]

    def _get_legal_actions(self):
        legal_actions = self.game.get_legal_actions()
        legal_ids = {ACTION_SPACE[action]: None for action in legal_actions}
        return OrderedDict(legal_ids)

    def get_perfect_information(self):
        ''' Get the perfect information of the current state

        Returns:
            (dict): A dictionary of all the perfect information of the current state
        '''
        state = {}
        state['num_players'] = self.num_players
        state['current_player'] = self.game.current_player
        state['color_to_play'] = self.game.color_to_play
        state['hand_cards'] = [cards2list(player.hand) for player in self.game.players]
        state['legal_actions'] = self.game.get_legal_actions()
        state['played_cards'] = cards2list(self.game.played_cards)
        state['played_cards_in_trick'] = cards2list(self.game.played_cards_in_trick)
        state['player_without_color'] = self.game.player_without_color
        state['trick_scores'] = self.game.trick_scores
        state['predicted_tricks'] = self.game.tricks_to_predict
        state['wizard_played'] = self.game.wizard_played
        return state

