"""
Simple Web Server with Flask

See docs for further information: https://flask.palletsprojects.com/en/2.0.x/quickstart/#a-minimal-application
"""
from flask import Flask
from flask import request, send_from_directory, jsonify
from flask_cors import CORS, cross_origin
import json
import os
from typing import Dict
import random
import numpy as np
import pandas as pd
import rlcard
from rlcard import models
from rlcard.agents.human_agents.webapp_wizard_s_trickpred_human_agent import HumanAgent, _print_action
from rlcard.agents import RandomAgent
import torch

app = Flask(__name__, static_folder='dist')

CORS(app,resources={r"/api": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

class WizardServer:
    """This class keeps the game state or resets if a new game is started."""
    def __init__(self, seed):
        self.app = app
        if not seed:
            seed = random.randint(1,10000)
        
        self.config = {
            'game_num_players': 2,
            'game_num_cards': 5,
            'env': 'wizard_simple_webapp',
            'seed':seed,
            'no_human_players': 1,
            'opponent': 'nfsp',
            'load_path_agent': 'experiments/wizard_simple_nfsp_result/model.pth',
        }

        self.env = rlcard.make(self.config["env"],self.config, self.app)
        agents = [HumanAgent(self.env.num_actions, self.app) for _ in range(self.config["no_human_players"])]
        # Append more agents to the game until the number of players is reached.
        for i in range(self.config["no_human_players"],self.config["game_num_players"]):
            if self.config["opponent"]=="nfsp":
                additional_agent = torch.load(self.config["load_path_agent"])
            else:
                additional_agent = RandomAgent(num_actions=self.env.num_actions)
            agents.append(additional_agent)
        self.env.set_agents(agents)
        self.state={}
        

    def run_game(self):
        print(">> Start a new game")

        trajectories, payoffs = self.env.run(is_training=False)
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

        

@app.route("/api/selected_card_index",  methods = ['POST'])
@cross_origin(origin='*',headers=['content-type'])
def getNewStateAfterPlayedCard():
    print(json.loads(request.data)["selected_card_index"])
    # if len(selected_card)>0:
    #     print(int(selected_card))
    # else:
    #     print("No card was selected!")

    return {"tester":"Test-Response"}


@app.route("/api/current_hand", methods = ['GET'])
@cross_origin(origin='*',headers=['content-type'])
def getCurrentHand() -> Dict:
    starting_hand = ["r-1","r-4","b-wizard","y-1","g-fool"]
    starting_hand = {
        "starting_hand": starting_hand
    }
    print(starting_hand)
    return starting_hand





if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081)
