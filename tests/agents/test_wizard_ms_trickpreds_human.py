import unittest
import numpy as np
from rlcard.agents.human_agents.wizard_ms_trickpred_human_agent import _print_state, _print_action



class TestLeducHuman(unittest.TestCase):

    def test_print_state(self):
        raw_state = {'hand': ['r-2','b-1',"y-3", 'b-3'],
        'current_player': 0,
        'color_to_play':"g",
        'legal_actions': ['y-3'],
        'played_cards': [ 'r-1', 'r-3','y-2'],
        'played_cards_in_trick':['y-2'],
        'num_cards_left': 4,
        'actions_in_trick_left':1,
        'player_without_color':np.zeros((2,4)),
        'wizard_played:':True,
        'trick_scores': [0,1],
        'predicted_tricks': [2,2],
        'tricks_left_to_get':[2,1]
        }
        action_record = []
        _print_state(raw_state, action_record)
        

    def test_print_action(self):
        _print_action('r-2')

if __name__ == '__main__':
    unittest.main(exit=False)