import unittest

import rlcard
from rlcard.envs.registration import register, make
from determism_util import is_deterministic


class TestRegistration(unittest.TestCase):

    def test_register(self):
        register(env_id='test_reg', entry_point='rlcard.envs.wizard:WizardEnv')
        with self.assertRaises(ValueError):
            register(env_id='test_reg', entry_point='rlcard.envs.wizard:WizardEnv')

    def test_make(self):
        register(env_id='test_make', entry_point='rlcard.envs.wizard:WizardEnv')
        env = rlcard.make('test_make')
        _, player = env.reset()
        with self.assertRaises(ValueError):
            make('test_random_make')

    def test_make_modes(self):
        register(env_id='test_env', entry_point='rlcard.envs.wizard:WizardEnv')

if __name__ == '__main__':
    unittest.main()
