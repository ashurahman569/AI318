import math

class Mancala:
    def __init__(self):
        self.matrix = [[4] * 6, [4] * 6]
        self.mystore = 0
        self.oppstore = 0
        self.extra_turn = False
        self.captured = 0

    def copy(self):
        new = Mancala()
        new.matrix = [row[:] for row in self.matrix]
        new.mystore = self.mystore
        new.oppstore = self.oppstore
        new.extra_turn = self.extra_turn
        new.captured = self.captured
        return new

    def heuristic1(self):
        return self.mystore - self.oppstore

    def heuristic2(self, W1=1, W2=1):
        stonesmyside = sum(self.matrix[1])
        stonesoppside = sum(self.matrix[0])
        return W1 * self.heuristic1() + W2 * (stonesmyside - stonesoppside)

    def heuristic3(self, W1=1, W2=1, W3=1):
        return self.heuristic2(W1, W2) + W3 * (1 if self.extra_turn else 0)

    def heuristic4(self, W1=1, W2=1, W3=1, W4=1):
        return self.heuristic3(W1, W2, W3) + W4 * self.captured

    def evaluate(self, heuristic_num, weights):
        if heuristic_num == 1:
            return self.heuristic1()
        if heuristic_num == 2:
            return self.heuristic2(*weights) # * is dereferencing the array. doesnt work if i only write weights
        if heuristic_num == 3:
            return self.heuristic3(*weights)
        return self.heuristic4(*weights)

    def legal_moves(self):
        return [i for i in range(6) if self.matrix[1][i] > 0]

    def is_terminal(self):
        return sum(self.matrix[0]) == 0 or sum(self.matrix[1]) == 0

    def apply_move(self, move):
        stones = self.matrix[1][move]
        self.matrix[1][move] = 0
        self.extra_turn = False
        self.captured = 0

        bin_index = move  # 0-5
        last_side = None  # 'me'/'opp'/'store'
        last_bin = None  # bindex of the final stone

        while stones > 0:
            bin_index += 1
            while bin_index < 6 and stones > 0:
                self.matrix[1][bin_index] += 1
                stones -= 1
                last_side, last_bin = 'me', bin_index
                if stones == 0:
                    break
                bin_index += 1

            if stones == 0:
                break

            self.mystore += 1
            stones -= 1
            last_side, last_bin = 'store', None
            if stones == 0:
                break

            opp_index = -1
            while opp_index < 5 and stones > 0:
                opp_index += 1
                self.matrix[0][opp_index] += 1
                stones -= 1
                last_side, last_bin = 'opp', opp_index
                if stones == 0:
                    break

            bin_index = -1

        self.extra_turn = (last_side == 'store')

        if last_side == 'me' and self.matrix[1][last_bin] == 1:
            opp_idx = 5 - last_bin
            opp_stones = self.matrix[0][opp_idx]
            if opp_stones > 0:
                self.mystore += opp_stones + 1
                self.matrix[0][opp_idx] = 0
                self.matrix[1][last_bin] = 0
                self.captured = opp_stones + 1

        self._sweep_if_terminal()
        return self.extra_turn

    def _sweep_if_terminal(self):
        if sum(self.matrix[1]) == 0:
            self.oppstore += sum(self.matrix[0])
            self.matrix[0] = [0] * 6
        elif sum(self.matrix[0]) == 0:
            self.mystore += sum(self.matrix[1])
            self.matrix[1] = [0] * 6

    def flip(self):
        self.matrix[0], self.matrix[1] = self.matrix[1], self.matrix[0]
        self.mystore, self.oppstore = self.oppstore, self.mystore

    def print_board(self, mover_label):
        print(f"\n[Opp store = {self.oppstore}]")
        print("  opp bins:", self.matrix[0])
        print("  my bins :", self.matrix[1])
        print(f" [My store = {self.mystore}]   <- to move: {mover_label}\n")


def minimax(state, depth, alpha, beta, heuristic_num, weights):
    if depth == 0 or state.is_terminal():
        return state.evaluate(heuristic_num, weights), None

    moves = state.legal_moves()
    if not moves:
        return state.evaluate(heuristic_num, weights), None

    best_val, best_move = -math.inf, moves[0]
    for move in moves:
        child = state.copy()
        extra = child.apply_move(move)
        if extra:
            val, mv = minimax(child, depth - 1, alpha, beta, heuristic_num, weights)
        else:
            child.flip()
            val, mv = minimax(child, depth - 1, -beta, -alpha, heuristic_num, weights)
            val = -val
        if val > best_val:
            best_val, best_move = val, move
        alpha = max(alpha, best_val)
        if alpha >= beta:
            break  # prune
    return best_val, best_move


def main():
    p1_type = "c"
    p2_type = "c"
    # print("Search depth for computer: ")
    # depth = int(input())
    # print("Heuristic number (1-4): ")
    # heuristic_num = int(input())
    win1 = 0
    win2 = 0
    player_type = {1: p1_type, 2: p2_type}
    heuristics = [1,2,3,4]
    depths = [2,3,4,5,6]
    heuristicwithdepth = [(h,d,h2, d2) for h in heuristics for d in depths for h2 in heuristics for d2 in depths]

    for heuristic_1, depth_1, heuristic_2, depth_2 in heuristicwithdepth:
        state = Mancala()      
        current = 1           
        weights1 = (1, 1, 1, 1)[: heuristic_1 - 1] if heuristic_1 > 1 else ()
        weights2 = (1, 1, 1, 1)[: heuristic_2 - 1] if heuristic_2 > 1 else ()
        while not state.is_terminal():
            #state.print_board(f"Player {current} ({player_type[current]})")
            moves = state.legal_moves()
            moves = [m + 1 for m in moves]
            if player_type[current].startswith('h'):
                move = None
                while move not in moves:
                    try:
                        print(f"Player {current}, pick a bin from {moves}: ")
                        move = int(input())
                    except (ValueError, TypeError):
                        move = None
                move = move - 1
            else:
                if current == 1:
                    x, move = minimax(state, depth_1, -math.inf, math.inf, heuristic_1, weights1)
                else:
                    x, move = minimax(state, depth_2, -math.inf, math.inf, heuristic_2, weights2)
                #print(f"Computer (Player {current}) plays bin {move + 1}")

            extra = state.apply_move(move)
            if not extra:
                state.flip()
                current = 2 if current == 1 else 1
            #else:
                #print(f"Player {current} finished in his store, extra turn")

        other = 2 if current == 1 else 1
        if p1_type == "c":
            print(f"\nPlayer 1 (Heuristic {heuristic_1}, Depth {depth_1})")
        if p2_type == "c":
            print(f"Player 2 (Heuristic {heuristic_2}, Depth {depth_2})")
        print(f"Player {current} store: {state.mystore}")
        print(f"Player {other} store: {state.oppstore}")
        if state.mystore > state.oppstore:
            print(f"Player {current} wins!")
            if current == 1:
                win1 += 1
            else:
                win2 += 1
        elif state.oppstore > state.mystore:
            print(f"Player {other} wins!")
            if other == 1:
                win1 += 1   
            else:
                win2 += 1
        else:
            print("It's a draw!")

    print(f"\nFinal results after 400 games: Player 1 wins: {win1}, Player 2 wins: {win2}, Draws: {400 - win1 - win2}")


if __name__ == "__main__":
    main()