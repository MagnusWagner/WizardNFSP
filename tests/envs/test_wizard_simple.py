import unittest
import numpy as np

import rlcard
from rlcard.agents.random_agent import RandomAgent
from rlcard.games.wizard_simple.utils import ACTION_LIST
from determism_util import is_deterministic


class TestWizardEnv(unittest.TestCase):

    def test_reset_and_extract_state(self):
        env = rlcard.make('wizard_simple')
        state, _ = env.reset()
        self.assertEqual(state['obs'].size, 112) ###

    def test_is_deterministic(self):
        self.assertTrue(is_deterministic('wizard_simple'))

    def test_get_legal_actions(self):
        env = rlcard.make('wizard_simple')
        env.set_agents([RandomAgent(env.num_actions) for _ in range(env.num_players)])
        env.reset()
        legal_actions = env._get_legal_actions()
        for legal_action in legal_actions:
            self.assertLessEqual(legal_action, 24) ###

    def test_step(self):
        env = rlcard.make('wizard_simple')
        state, _ = env.reset()
        action = np.random.choice(list(state['legal_actions'].keys()))
        _, player_id = env.step(action)
        self.assertEqual(player_id, env.game.current_player.player_id)


    def test_run(self):
        env = rlcard.make('wizard_simple')
        env.set_agents([RandomAgent(env.num_actions) for _ in range(env.num_players)])
        trajectories, payoffs = env.run(is_training=False)
        self.assertEqual(len(trajectories), 2)
        total = 0
        for payoff in payoffs:
            total += payoff

        trajectories, payoffs = env.run(is_training=True)
        total = 0
        for payoff in payoffs:
            total += payoff


    def test_decode_action(self):
        env = rlcard.make('wizard_simple')
        env.reset()
        legal_actions = env._get_legal_actions()
        for legal_action in legal_actions:
            decoded = env._decode_action(legal_action)
            self.assertLessEqual(decoded, ACTION_LIST[legal_action])

    def test_get_perfect_information(self):
        env = rlcard.make('wizard_simple')
        _, player_id = env.reset()
        self.assertEqual(player_id, env.get_perfect_information()['current_player'].player_id)

        
if __name__ == '__main__':
    unittest.main()
