''' An example of evluating the trained models in RLCard
'''
import os
import argparse

import rlcard
from rlcard.agents import DQNAgent, RandomAgent
from rlcard.agents.nfsp_agent_wz import NFSPAgentWZ
from rlcard.utils import get_device, set_seed, tournament

def load_model(model_path, env=None, position=None, device=None):
    if os.path.isfile(model_path):  # Torch model
        import torch
        assert args.env == "wizard" or args.env == "wizard_trickpreds" or model_path.split("_")[1]==args.env.split("_")[1] 
        agent = torch.load(model_path, map_location=device)
        agent.set_device(device)
    elif model_path == 'random':  # Random model
        from rlcard.agents import RandomAgent
        agent = RandomAgent(num_actions=env.num_actions)
    else: 
        raise("No Agent found!")
    return agent

def evaluate(args):
    '''
    Evaluate the model for a number of rounds against another chosen model.

    Important inputs:
        - SEED: Important to change when retraining models, otherwise, the same game rounds are played again.
        - GAME_NUM_PLAYERS: Number of players playing the game.
        - GAME_NUM_CARDS: Number of cards in the game. Default is 5 here.
        - args.models : model_paths for the models that play against each other.
        - args.num_games: Number of games to be played.
        
    Results are only printed to the console and need to be saved manually.
    '''
    SEED=args.seed
    GAME_NUM_PLAYERS=2
    assert len(args.models)==2
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
    rewards = tournament(env, args.num_games)
    for position, reward in enumerate(rewards):
        print(position, args.models[position], reward)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Evaluation of Wizard Bot")
    parser.add_argument('--env', type=str, default="wizard",choices=['wizard_trickpreds',"wizard_s_trickpreds","wizard_ms_trickpreds",'wizard','wizard_simple','wizard_most_simple'])
    parser.add_argument('--models', nargs='*', default=['experiments/wizard_nfsp_result/model.pth', 'random'])
    parser.add_argument('--cuda', type=str, default='')
    parser.add_argument('--seed', type=int, default=12172)
    parser.add_argument('--num_games', type=int, default=10000)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    evaluate(args)

