# Reza Heidari
# Student ID : 400109616
# Use depth 3 or less for faster performance

import copy
from Board import BoardUtility
import random
import numpy as np


class Player:
    def __init__(self, player_piece):
        self.piece = player_piece

    def play(self, board):
        return 0


class RandomPlayer(Player):
    def play(self, board):
        return [random.choice(BoardUtility.get_valid_locations(board)), random.choice([1, 2, 3, 4]), random.choice(["skip", "clockwise", "anticlockwise"])]


class HumanPlayer(Player):
    def play(self, board):
        move = input("row, col, region, rotation\n")
        move = move.split()
        print(move)
        return [[int(move[0]), int(move[1])], int(move[2]), move[3]]


class MiniMaxPlayer(Player):
    def __init__(self, player_piece, depth=3):
        super().__init__(player_piece)
        self.depth = depth

    def play(self, board):
        row = -1
        col = -1
        region = -1
        rotation = -1
        [row, col], region, rotation = MiniMaxUtility.max_value(self, board, 0, -np.inf, np.inf, is_prob=False)[0]
        return [[row, col], region, rotation]


class MiniMaxProbPlayer(Player):
    def __init__(self, player_piece, depth=3, prob_stochastic=0.1):
        super().__init__(player_piece)
        self.depth = depth
        self.prob_stochastic = prob_stochastic

    def play(self, board):
        row = -1
        col = -1
        region = -1
        rotation = -1
        [row, col], region, rotation = MiniMaxUtility.random_prob_node(self, board, 0)[0]
        return [[row, col], region, rotation]


class MiniMaxUtility:
    anti_rotation = {'clockwise': 'anticlockwise', 'anticlockwise': 'clockwise', 'skip': 'skip'}

    @staticmethod
    def random_prob_node(player, board, depth):
        move = []
        if BoardUtility.is_terminal_state(board) or depth == player.depth:
            return move, MiniMaxUtility.evaluation(player, board)
        move, value = MiniMaxUtility.max_value(player, board, depth, -np.inf, np.inf, is_prob=False)
        value *= (1 - player.prob_stochastic)
        (i, j), region, rotation = random.choice(MiniMaxUtility.get_states(player, board, depth))
        move_stochastic = [[i, j], region, rotation]
        BoardUtility.make_move(board, i, j, region, rotation, player.piece)
        value += player.prob_stochastic * MiniMaxUtility.min_value(player, board, depth + 1, -np.inf, np.inf, is_prob=True)[1]
        BoardUtility.rotate_region(board, region, MiniMaxUtility.anti_rotation[rotation])
        board[i][j] = 0
        if random.random() <= player.prob_stochastic:
            move = move_stochastic
        return move, value

    @staticmethod
    def max_value(player, board, depth, alpha, beta, is_prob=False):
        value = -np.inf
        move = []
        if BoardUtility.is_terminal_state(board) or depth == player.depth:
            return move, MiniMaxUtility.evaluation(player, board)

        for (i, j), region, rotation in MiniMaxUtility.get_states(player, board, depth):
            BoardUtility.make_move(board, i, j, region, rotation, player.piece)
            new_value = MiniMaxUtility.min_value(player, board, depth + 1, alpha, beta, is_prob)[1]
            if new_value > value:
                value = new_value
                move = [[i, j], region, rotation]
            BoardUtility.rotate_region(board, region, MiniMaxUtility.anti_rotation[rotation])
            board[i][j] = 0
            if value >= beta and not is_prob:
                return move, value
            alpha = max(alpha, value)
        return move, value

    @staticmethod
    def min_value(player, board, depth, alpha,  beta, is_prob=False):
        value = np.inf
        move = []
        if BoardUtility.is_terminal_state(board) or depth == player.depth:
            return move, MiniMaxUtility.evaluation(player, board)

        for (i, j), region, rotation in MiniMaxUtility.get_states(player, board, depth):
            BoardUtility.make_move(board, i, j, region, rotation, 2 if player.piece == 1 else 1)
            if not is_prob:
                new_value = MiniMaxUtility.max_value(player, board, depth + 1, alpha, beta, is_prob)[1]
            else:
                new_value = MiniMaxUtility.random_prob_node(player, board, depth + 1)[1]
            if new_value < value:
                value = new_value
                move = [[i, j], region, rotation]
            BoardUtility.rotate_region(board, region, MiniMaxUtility.anti_rotation[rotation])
            board[i][j] = 0
            if value <= alpha and not is_prob:
                return move, value
            beta = min(beta, value)
        return move, value

    @staticmethod
    def get_states(player, board, depth):
        valid_locations = BoardUtility.get_valid_locations(board)
        moves = []
        for i, j in valid_locations:
            for rotation in ['clockwise', 'anticlockwise', 'skip']:
                for region in range(1, 5):
                    moves.append(((i, j), region, rotation))
                    if rotation == 'skip':
                        break
        random.shuffle(moves)
        return moves[:min((200 if isinstance(player, MiniMaxProbPlayer) else 100) // player.depth, len(moves))]
        # return moves[:min(200 // player.depth, len(moves))]

    @staticmethod
    def evaluation(player, board):
        if BoardUtility.has_player_won(board, player.piece):
            return 1000000000
        if BoardUtility.has_player_won(board, 1 if player.piece == 2 else 2):
            return -1000000000
        return 4 * MiniMaxUtility.check_continues_marbles(player.piece, board) - MiniMaxUtility.check_continues_marbles(1 if player.piece == 2 else 2, board)

    @staticmethod
    def check_continues_marbles(piece, board):
        result = 0
        for i in range(6):
            for j in range(6):
                if board[i][j] != piece:
                    continue
                horizontal = 0
                vertical = 0
                diagonal_right = 0
                diagonal_left = 0
                for k in range(5):
                    if j + k > 5 or board[i][j + k] != piece:
                        break
                    horizontal += 1
                for k in range(5):
                    if i + k > 5 or board[i + k][j] != piece:
                        break
                    vertical += 1
                for k in range(5):
                    if j + k > 5 or i + k > 5 or board[i + k][j + k] != piece:
                        break
                    diagonal_right += 1
                for k in range(5):
                    if j + k > 5 or i - k < 0 or board[i - k][j + k] != piece:
                        break
                    diagonal_left += 1
                result += horizontal**3 + vertical**3 + diagonal_right**3 + diagonal_left**3
        result += 20*(piece == board[1][1] + piece == board[1][4] + piece == board[4][1] + piece == board[4][4])
        return result
