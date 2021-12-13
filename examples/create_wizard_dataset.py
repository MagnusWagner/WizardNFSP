''' An example of evluating the trained models in RLCard
'''
import os
import argparse
import pandas as pd

import rlcard
from rlcard.agents import DQNAgent, RandomAgent
from rlcard.agents.nfsp_agent_wz import NFSPAgentWZ
from rlcard.utils import get_device, set_seed, get_payoff_state_combinations

def load_model(model_path, env=None, position=None, device=None):
    if os.path.isfile(model_path):  # Torch model
        import torch
        agent = torch.load(model_path, map_location=device)
        agent.set_device(device)
    elif model_path == 'random':  # Random model
        from rlcard.agents import RandomAgent
        agent = RandomAgent(num_actions=env.num_actions)
    else:  # A model in the model zoo
        from rlcard import models
        agent = models.load(model_path).agents[position]
    
    return agent

def create_dataset(args):
    '''
    Create a dataset out of two bots playing against each other trying to maximize their tricks.
    The trajectories and payoffs are used to build & train a model that can predict the approximate 
    number of tricks an agent should make in a game. 
    The model is used to train models that work with trick predictions.

    Important inputs:
        - SEED: Important to change when retraining models, otherwise, the same game rounds are played again.
        - GAME_NUM_PLAYERS: Number of players playing the game.
        - GAME_NUM_CARDS: Number of cards in the game. Default is 5 here.
        - args.models : model_paths for the models that play against each other. WARNING: Should not be random!
        - args.num_games: Number of games to be played and to create the dataset
        - args.save_path: Where the results are saved.
        
    Results are saved in the path 
    '''
    SEED=args.seed
    GAME_NUM_PLAYERS=len(args.models)
    GAME_NUM_CARDS=5
    # Check whether gpu is available
    device = get_device()
        
    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(args.env, config={
        'game_num_players': GAME_NUM_PLAYERS,
        'game_num_cards': GAME_NUM_CARDS,
        'seed':SEED,
        })

    # Load models
    agents = []
    for position, model_path in enumerate(args.models):
        agents.append(load_model(model_path, env, position, device))
    env.set_agents(agents)

    # Evaluate
    payoff_hand_pairs = get_payoff_state_combinations(env, args.num_games)
    payoff_hand_pairs_DF = pd.DataFrame(payoff_hand_pairs)
    payoff_hand_pairs_DF.to_pickle(args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Evaluation of Wizard Bot")
    parser.add_argument('--env', type=str, default='wizard',
            choices=['wizard','wizard_simple','wizard_most_simple'])
    parser.add_argument('--models', nargs='*', default=['experiments/wizard_nfsp_result/model.pth', 'experiments/wizard_nfsp_result/model.pth'])
    parser.add_argument('--cuda', type=str, default='')
    parser.add_argument('--seed', type=int, default=1321234)
    parser.add_argument('--num_games', type=int, default=500000)
    parser.add_argument('--save-path', type=str, default='experiments/wizard_nfsp_result/trick_prediction_results/tp01.pickle')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    create_dataset(args)

