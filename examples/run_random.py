''' An example of playing randomly in RLCard
'''
import argparse
import pprint

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import set_seed


DEFAULT_GAME_CONFIG = {
        'game_num_players': 2,
        'game_num_cards': 5,
        'seed':None,
        }

def run(args):
    # Make environment
    env = rlcard.make(args.env, DEFAULT_GAME_CONFIG)

    # Seed numpy, torch, random
    set_seed(42)

    # Set agents
    agent = RandomAgent(num_actions=env.num_actions)
    env.set_agents([agent for _ in range(env.num_players)])

    # Generate data from the environment
    trajectories, player_wins = env.run(is_training=False)
    # Print out the trajectories
    print('\nTrajectories:')
    print(trajectories)
    print('\nSample raw observation:')
    pprint.pprint(trajectories[0][0]['raw_obs'])
    print('\nSample raw legal_actions:')
    pprint.pprint(trajectories[0][0]['raw_legal_actions'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Random example in RLCard")
    parser.add_argument('--env', type=str, default='wizard_ms_trickpreds',
            choices=['wizard_trickpreds',"wizard_s_trickpreds","wizard_ms_trickpreds",'wizard','wizard_simple','wizard_most_simple'])

    args = parser.parse_args()
    run(args)

