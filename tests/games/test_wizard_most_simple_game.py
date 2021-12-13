import unittest
import numpy as np

from rlcard.games.wizard_most_simple.game import WizardGame as Game
from rlcard.games.wizard_most_simple.player import WizardPlayer as Player
from rlcard.games.wizard_most_simple.utils import ACTION_LIST
from rlcard.games.wizard_most_simple.utils import encode_cards, COLOR_MAP, TRAIT_MAP

class TestWizardMethods(unittest.TestCase):

    def test_get_num_player(self):
        game = Game()
        num_players = game.get_num_players()
        self.assertEqual(num_players, 2)

    def test_get_num_actions(self):
        game = Game()
        num_actions = game.get_num_actions()
        self.assertEqual(num_actions, 12)

    def test_init_game(self):
        game = Game()
        state, _ = game.init_game()

    def test_get_player_id(self):
        game = Game()
        _, player_id = game.init_game()
        current = game.get_player_id()
        self.assertEqual(player_id, current)


    def test_get_legal_actions(self):
        game = Game()
        game.init_game()
        actions = game.get_legal_actions()
        for action in actions:
            self.assertIn(action, ACTION_LIST)

    def test_step(self):
        game = Game()
        game.init_game()
        action = np.random.choice(game.get_legal_actions())
        state, next_player_id = game.step(action)
        current = game.current_player.player_id
        self.assertLessEqual(len(state['played_cards']), 2)
        self.assertEqual(next_player_id, current)

    def test_get_payoffs(self):
        game = Game()
        game.init_game()
        while not game.is_over:
            actions = game.get_legal_actions()
            action = np.random.choice(actions)
            state, _ = game.step(action)
        payoffs = game.get_payoffs()
        total = 0
        for payoff in payoffs:
            total += payoff


    def test_encode_cards(self):
        hand1 = ['r-2',"y-3", 'b-3', 'b-1']
        encoded_hand1=encode_cards(hand1)
        for card in hand1:
            card = card.split("-")
            color = card[0]
            trait = card[1]
            self.assertEqual(encoded_hand1[COLOR_MAP[color],TRAIT_MAP[trait]], 1)

    def test_player_get_player_id(self):
        player = Player(0, np.random.RandomState())
        self.assertEqual(0, player.get_player_id())

if __name__ == '__main__':
    unittest.main()
