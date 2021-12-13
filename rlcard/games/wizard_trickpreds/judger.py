
import numpy as np

class WizardJudger:

    @staticmethod
    def judge_winner(self, players, trick_scores):
        ''' Judge the winner of the game

        Args:
            players (list): The list of players who play the game
            trick_scores (list): The scores of all players in the right order.

        Returns:
            (list): The player id of the winner
        '''
        winner_idx = np.argmax(trick_scores)
        return players[winner_idx].player_id
