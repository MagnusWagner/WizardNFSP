import os
import json
import numpy as np
from collections import OrderedDict

import rlcard

from rlcard.games.wizard_simple.card import WizardCard as Card

# Read required docs
ROOT_PATH = rlcard.__path__[0]

# a map of abstract action to its index and a list of abstract action
with open(os.path.join(ROOT_PATH, 'games/wizard_simple/jsondata/action_space.json'), 'r') as file:
    ACTION_SPACE = json.load(file, object_pairs_hook=OrderedDict)
    ACTION_LIST = list(ACTION_SPACE.keys())

# a map of color to its index
COLOR_MAP = {'r': 0, 'g': 1, 'b': 2, 'y': 3}
COLOR_MAP_WORDS = {'r': "red", 'g': "green", 'b': "blue", 'y': "yellow"}

# a map of trait to its index
TRAIT_MAP = {'fool': 0, '1': 1, '2': 2, '3': 3, '4': 4, 'wizard': 5}

WIZARD = ['r-wizard', 'g-wizard', 'b-wizard', 'y-wizard']

FOOL = ['r-fool', 'g-fool', 'b-fool', 'y-fool']


def init_deck():
    ''' Generate wizard deck of 24 cards
    '''
    deck = []
    card_info = Card.info
    for color in card_info['color']:

        # init number cards
        for num in card_info['trait'][1:5]:
            deck.append(Card('number', color, num))

        deck.append(Card('fool', color, card_info['trait'][0]))
        deck.append(Card('wizard', color, card_info['trait'][5]))

    return deck


def cards2list(cards):
    ''' Get the corresponding string representation of cards

    Args:
        cards (list): list of WizardCards objects

    Returns:
        (string): string representation of cards
    '''
    cards_list = []
    for card in cards:
        cards_list.append(card.get_str())
    return cards_list

def hand2dict(hand):
    ''' Get the corresponding dict representation of hand

    Args:
        hand (list): list of string of hand's card

    Returns:
        (dict): dict of hand
    '''
    hand_dict = {}
    for card in hand:
        if card not in hand_dict:
            hand_dict[card] = 1
        else:
            hand_dict[card] += 1
    return hand_dict

def encode_cards(cardlist):
    ''' Encode hand and represerve it into plane
 
    Args:
        cardlist (list): list of string of cards

    Returns:
        (array): 4*15 numpy array
    '''
    plane = np.zeros((4, 6), dtype=int)
    for card in cardlist:
        card_info = card.split('-')
        color = COLOR_MAP[card_info[0]]
        trait = TRAIT_MAP[card_info[1]]
        plane[color][trait] = 1
    return plane

def encode_legal_actions(legal_actions):
    ''' Encode hand and represerve it into plane
    Args:
        cardlist (list): list of string of cards

    Returns:
        (array): 4*15 numpy array
    '''
    plane = np.zeros((4, 6), dtype=int)
    # plane[0] = np.ones((4, 15), dtype=int)
    for card in legal_actions:
        card_info = card.split('-')
        color = COLOR_MAP[card_info[0]]
        trait = TRAIT_MAP[card_info[1]]
        plane[color][trait] = 1
    return plane

def encode_color(color,no_color_possible=False):
    ''' Encode hand and represerve it into plane
    Args:
        color: "r","g","b","y"

    Returns:
        (array): 4*1 numpy array
    '''
    if no_color_possible:
        plane = np.zeros(5, dtype=int)
        if color:
            plane[COLOR_MAP.get(color)]=1
        else:
            plane[4]=1
    else:
        plane = np.zeros(4, dtype=int)
        plane[COLOR_MAP.get(color)]=1
    return plane

def getPlayerOrders(num_players):
    '''
    returns a list of player orders as a dict().
    Key is the starting player. 
    Value is order_list of players following the starting player.
    (More relevant for more than 2 players.)
    '''
    player_orders = dict()
    for starting_player_idx in range(num_players):
        player_order = []
        order_player_index=starting_player_idx
        for i in range(num_players):
            player_order.append(order_player_index)
            order_player_index = (order_player_index + 1) % num_players
        player_orders[starting_player_idx]=player_order
    return player_orders
