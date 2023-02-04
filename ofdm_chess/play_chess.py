#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Nolan Pearce
# Created Date: 2022-12-27
# version ='1.0'
#----------------------------------------------------------------------------
"""play_chess.py sets the game environment and workflow for an acoustic chess game"""
#----------------------------------------------------------------------------
from tx_ofdm import *
from rx_ofdm import *
import chess

# Functions
def draw_board(board):
    print(board)

# Initialize
chess.reset()
board = chess.Board()
## Black/White selection
color = input("Color (b/w): ")
if color == "b":
    chess.Color = False
elif color == "w":
    chess.Color = True

# Game Loop
while True:
    ## Draw Board
    draw_board(board)
    ## White moveset
    if chess.Color == True:
        ### Input Move
        move = input("Make new move: ")
        print("Move: ", move)
        uci_move = chess.Move.from_uci(move)
        board.push(uci_move)  # Make the move
        ### Transmit
        tx_ofdm(move)
        chess.Color = False

    ## Draw Board
    draw_board(board)

    ## Black moveset
    if chess.Color == False:
        ### Receive
        decision = ''
        while decision != 'y':
            move = rx_ofdm()
            ### Receive move
            print("Received move is: ", move)
            decision = input("Receive again? y/n")
        uci_move = chess.Move.from_uci(move)
        board.push(uci_move)
        chess.Color = True