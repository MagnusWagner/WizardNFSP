''' An example of training a reinforcement learning agent on the environments in RLCard
'''
import os
import argparse

import torch

import rlcard
from rlcard.agents import RandomAgent
from rlcard.agents.nfsp_agent_wz import NFSPAgentWZ
from rlcard.utils import get_device, set_seed, tournament, reorganize, Logger, plot_curve

def train(args):
    '''
    Train the model with the settings below for the number of steps specified in the parser arguments.
    If args.load_model==True, the model training is continued from a checkpoint. Additionally, an opponent model is used to train against.
    For the wizard_trickpreds variant, the model tries to make as many tricks as were predicted by the XGBoost Model from previous random results.
    Further improvements would be to retrain the XGBModel with a better NFSP-Model, but for time efficiency reasons, I stopped there.

    Important inputs:
        - HIDDEN_LAYER_SIZE: Layer-Size for both networks, each network currently has two hidden layers.
        - SEED: Important to change when retraining models, otherwise, the same game rounds are played again.
        - GAME_NUM_PLAYERS: Number of players playing the game.
        - GAME_NUM_CARDS: Number of cards in the game. Default is 5 here.
        - RL_LR: LR of the Reinforcement Learner (Here DQN)
        - Q_EPSILON_START = First Value for Epsilon which gets smaller with the number of decay steps.
        - Q_EPSILON_DECAY_STEPS = Number of steps where Epsilon reduces to 0.
        - RANDOM_OPPONENT: Is Opponent a random agent or a trained NFSP agent?
        - args.load_model: Bool: If true, it loads the model from the experiment folder.
    '''

    # General settings
    RANDOM_OPPONENT=args.random_opponent
    LOAD_MODEL = args.load_model
    SEED=args.seed
    GAME_NUM_PLAYERS=2
    GAME_NUM_CARDS=5

    ### Settings to train
    HIDDEN_LAYER_SIZE = 512
    RL_LR = 0.0005 #ms_trickpreds2
    Q_EPSILON_START = 0.5 #ms_trickpreds2
    # RL_LR = 0.00005 #s_trickpreds
    # Q_EPSILON_START = 0.8 #s_trickpreds
    # RL_LR = 0.01 #s_trickpreds2
    # Q_EPSILON_START = 0.5 #s_trickpreds2
    Q_EPSILON_DECAY_STEPS=4*int(1e5)

    # Load paths
    load_path_main = os.path.join(args.log_dir, 'model.pth')
    load_path_opponent = os.path.join(args.log_dir, 'model_opponent.pth')
    save_path = os.path.join(args.log_dir, 'model.pth')
    save_path_opponent = os.path.join(args.log_dir, 'model_opponent.pth')
    # Check whether gpu is available
    device = get_device()
        
    # Seed numpy, torch, random
    set_seed(SEED)

    # Make the environment with seed
    env = rlcard.make(args.env, config={
        'game_num_players': GAME_NUM_PLAYERS,
        'game_num_cards': GAME_NUM_CARDS,
        'seed':SEED,
        })

    ##########################################
    ##### Loading in either DQN-Agent or NFSP-Agent (NFSP is able to actually improve on Wizard.)
    if not LOAD_MODEL:
        if args.algorithm == 'dqn':
            from rlcard.agents import DQNAgent
            agent = DQNAgent(num_actions=env.num_actions,
                            state_shape=env.state_shape[0],
                            mlp_layers=[HIDDEN_LAYER_SIZE,HIDDEN_LAYER_SIZE],
                            learning_rate=RL_LR,
                            epsilon_decay_steps=Q_EPSILON_DECAY_STEPS,
                            epsilon_start=Q_EPSILON_START,
                            device=device)
        elif args.algorithm == 'nfsp':
            agent = NFSPAgentWZ(num_actions=env.num_actions,
                            state_shape=env.state_shape[0],
                            hidden_layers_sizes=[HIDDEN_LAYER_SIZE,HIDDEN_LAYER_SIZE],
                            q_mlp_layers=[HIDDEN_LAYER_SIZE,HIDDEN_LAYER_SIZE],
                            rl_learning_rate=RL_LR,
                            q_epsilon_start=Q_EPSILON_START,
                            q_epsilon_decay_steps=Q_EPSILON_DECAY_STEPS,
                            device=device)

    ##### Saving previous model as next opponent_agent.
    if LOAD_MODEL:
        agent = torch.load(load_path_main)

    agents = [agent]
    for _ in range(env.num_players):
        if RANDOM_OPPONENT:
            agents.append(RandomAgent(num_actions=env.num_actions))
        else:
            agent_opponent = torch.load(load_path_opponent)
            agents.append(agent_opponent)

    if args.load_model == True:
        torch.save(agent, save_path_opponent)
        print('Opponent-Model saved in', save_path_opponent)
    
    env.set_agents(agents)



    # Start training
    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):

            if args.algorithm == 'nfsp':
                agents[0].sample_episode_policy()

            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)

            # Reorganaize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectories, payoffs)

            # Feed transitions into agent memory, and train the agent
            # Here, we assume that DQN always plays the first position
            # and the other players play randomly (or another model)
            for ts in trajectories[0]:
                agent.feed(ts)

            # Evaluate the performance. Play with random agents.
            if episode % args.evaluate_every == 0:
                logger.log_performance(env.timestep, tournament(env, args.num_eval_games)[0])

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Plot the learning curve
    plot_curve(csv_path, fig_path, args.algorithm)

    # Save model
    torch.save(agent, save_path)
    print('Model saved in', save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("NFSP example with Wizard & Trick Predictions")
    parser.add_argument('--env', type=str, default='wizard_ms_trickpreds',choices=['wizard_trickpreds',"wizard_s_trickpreds","wizard_ms_trickpreds"])
    parser.add_argument('--algorithm', type=str, default='nfsp', choices=['nfsp','dqn'])
    parser.add_argument('--load_model', type=bool, default=False, choices=[True,False]) # Change default to False for new model.
    parser.add_argument('--random_opponent', type=bool, default=True, choices=[True,False]) # Change default to False for new model.
    parser.add_argument('--cuda', type=str, default='')
    parser.add_argument('--seed', type=int, default=19593121)
    parser.add_argument('--num_episodes', type=int, default=200)
    parser.add_argument('--num_eval_games', type=int, default=100)
    parser.add_argument('--evaluate_every', type=int, default=100)
    parser.add_argument('--log_dir', type=str, default='experiments/newtest/') # Final testing

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train(args)

