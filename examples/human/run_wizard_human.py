''' A toy example of playing against rule-based bot on Wizard without trick predictions.
'''
import numpy as np
import rlcard
from rlcard import models
from rlcard.agents.human_agents.wizard_human_agent import HumanAgent, _print_action
from rlcard.agents import RandomAgent
import torch
import argparse
import os

def run_example(args):
    # Make environment
    config = {
            'game_num_players': 2,
            'game_num_cards': 5,
            'seed':args.seed,
            'env':args.env,
            'no_human_players': args.n_human_players,
            'opponent': args.opponent,
            'load_path_agent': args.load_path_agent,
    }

    env = rlcard.make(config["env"],config)

    # How many human players are in the game?
    assert config["no_human_players"] <= config["game_num_players"] and config["no_human_players"] >=1
    agents = [HumanAgent(env.num_actions) for _ in range(config["no_human_players"])]
    # Append more agents to the game until the number of players is reached.
    for i in range(config["no_human_players"],config["game_num_players"]):
        if config["opponent"]=="nfsp":
            additional_agent = torch.load(config["load_path_agent"])
        else:
            additional_agent = RandomAgent(num_actions=env.num_actions)
        agents.append(additional_agent)

    env.set_agents(agents)


    print(">> Wizard game")

    while (True):
        print(">> Start a new game")

        trajectories, payoffs = env.run(is_training=False)
        # If the human does not take the final action, we need to
        # print other players action
        final_state = trajectories[0][-1]
        action_record = final_state['action_record']
        state = final_state['raw_obs']
        _action_list = []
        for i in range(1, len(action_record)+1):
            if action_record[-i][0] == state['current_player']:
                break
            _action_list.insert(0, action_record[-i])
        for pair in _action_list:
            print('>> Player', pair[0], 'chooses ', end='')
            _print_action(pair[1])
            print('')


        print('=============== Current score ===============')
        for idx, score in enumerate(state['trick_scores']):
            print("P",str(idx+1),": ",score)

        print('===============     Result     ===============')
        winner_list = []
        max_score = np.max(payoffs)
        for idx,score in enumerate(payoffs):
            if score == max_score:
                winner_list.append(idx+1)
        if len(winner_list)==1:
            print('Player',winner_list[0],'wins!')
        else:
            print('Players',winner_list,'win!')
        print('')
        input("Press any key to continue...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Script to play wizard with humans/agents and trickpredictions")
    parser.add_argument('--env', type=str, default='wizard', choices=['wizard', 'wizard_simple', 'wizard_most_simple'])
    parser.add_argument('--cuda', type=str, default='')
    parser.add_argument('--seed', type=int, default=45143982)
    parser.add_argument('--n_human_players', type=int, default=1, choices=[1,2])
    parser.add_argument('--opponent', type=str, default='nfsp', choices=['nfsp','random'])
    parser.add_argument('--load_path_agent', type=str, default='experiments/wizard_result_nfsp/model.pth')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    run_example(args)
