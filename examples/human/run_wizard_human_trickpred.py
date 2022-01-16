''' A toy example of playing against rule-based bot on Wizard with trick predictions.
'''
import numpy as np
import rlcard
from rlcard import models
from rlcard.agents import RandomAgent
import torch
import os
import argparse
import random 
def run_example(args):
    # Make environment
    config = {
            'env':args.env,
            'game_num_players': 2,
            'game_num_cards': 5,
            'seed':args.seed,
            'no_human_players':args.n_human_players,
            'opponent': args.opponent,
            'load_path_agent': args.load_path_agent,
    }
    if config["seed"]==0:
        config["seed"]=random.randint(1,100000)
    
    environment_specific = config["env"].split("_")[1]
    # assert environment_specific == config["load_path_agent"].split("_")[1]

    if environment_specific=="s":
        from rlcard.agents.human_agents.wizard_s_trickpred_human_agent import HumanAgent, _print_action
        env = rlcard.make(f"wizard_{environment_specific}_trickpreds_with_humans",config)
    elif environment_specific=="ms":
        from rlcard.agents.human_agents.wizard_ms_trickpred_human_agent import HumanAgent, _print_action
        env = rlcard.make(f"wizard_{environment_specific}_trickpreds_with_humans",config)
    else:
        from rlcard.agents.human_agents.wizard_s_trickpred_human_agent import HumanAgent, _print_action
        env = rlcard.make(f"wizard_trickpreds_with_humans",config)


    # How many human players are in the game?
    assert config["no_human_players"] <= config["game_num_players"] and config["no_human_players"] >=1
    agents = [HumanAgent(env.num_actions) for _ in range(config["no_human_players"])]

    # Append more agents to the game until the number of players is reached.
    for i in range(config["no_human_players"],config["game_num_players"]):
        if config["opponent"]=="nfsp":
            # additional_agent = torch.load(config["load_path_agent"])
            additional_agent = torch.load(config["load_path_agent"], map_location=torch.device('cpu'))
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

        print('=============== Evaluation ===============')
        print('=== Trick Scores ===')
        for idx, score in enumerate(state['trick_scores']):
            print("P",str(idx+1),": ",str(score),"/",str(state['predicted_tricks'][idx]))

        print('=== Payoffs ===')
        for idx, score in enumerate(payoffs):
            print("P",str(idx+1),": ",score)

        print('===============     Result     ===============')
        winner_list = []
        max_score = np.max(payoffs)
        for idx,score in enumerate(payoffs):
            if score == max_score:
                winner_list.append(idx)
        if len(winner_list)==1:
            print('Player',winner_list[0]+1,'wins!')
        else:
            print('Players',[winner+1 for winner in winner_list],'win!')
        print('')
        input("Press any key to continue...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Script to play wizard with humans/agents and trickpredictions")
    parser.add_argument('--env', type=str, default='wizard_s_trickpreds', choices=['wizard_trickpreds','wizard_s_trickpreds',"wizard_ms_trickpreds"])
    parser.add_argument('--cuda', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_human_players', type=int, default=1, choices=[1,2])
    parser.add_argument('--opponent', type=str, default='nfsp', choices=['nfsp','random'])
    parser.add_argument('--load_path_agent', type=str, default='experiments/wizard_s_trickpreds_result_nfsp/model.pth')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    run_example(args)