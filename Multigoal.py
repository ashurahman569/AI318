import math
import copy
import heapq
import itertools

# =============================================================================
# VARIANT 7: Multiple Goal States
# Find the minimum cost path to ANY of the given goal states.
# Input format:
#   n
#   n lines: initial board
#   k          <- number of goal boards
#   n lines: goal board 1
#   n lines: goal board 2
#   ...
#
# Heuristic: min Manhattan distance across all goals (admissible).
# Cost: uniform (1 per move) — combine with other variants if needed.
# =============================================================================

class Heuristic:
    def __init__(self, func):
        self.func = func
    def execute(self, Matrix):
        return self.func(Matrix)
    def setHeuristic(self, func):
        self.func = func


class MultiGoalManhattan(Heuristic):
    """
    For each tile, compute Manhattan distance to its position in each goal,
    take the minimum across goals. Sum over all tiles.
    This is admissible because it lower-bounds the distance to the nearest goal.
    """
    def __init__(self, goals):
        super().__init__(self.calculate)
        # Precompute goal position maps: [{tile: (row,col)}, ...]
        self.goal_maps = []
        for goal in goals:
            gmap = {}
            for i in range(len(goal)):
                for j in range(len(goal[0])):
                    gmap[goal[i][j]] = (i, j)
            self.goal_maps.append(gmap)

    def _manhattan_to_goal(self, Matrix, gmap):
        total = 0
        for i in range(Matrix.n):
            for j in range(Matrix.n):
                t = Matrix.matrix[i][j]
                if t != 0 and t in gmap:
                    gr, gc = gmap[t]
                    total += abs(i - gr) + abs(j - gc)
        return total

    def calculate(self, Matrix):
        # Take the minimum heuristic value across all goals
        return min(self._manhattan_to_goal(Matrix, gmap) for gmap in self.goal_maps)


class Matrix:
    def __init__(self, n):
        self.n = n
        self.g = 0
        self.matrix = []
        self.zeropos = (-1, -1)
        self.heuristic = None
        self.parent = None
        self.last_move_cost = 0
        self.goal_maps = []     # list of {tile: (row,col)} for each goal

    def copy(self, other):
        self.n = other.n
        self.g = other.g
        self.matrix = copy.deepcopy(other.matrix)
        self.zeropos = other.zeropos
        self.heuristic = other.heuristic
        self.goal_maps = other.goal_maps    # shared, never mutated
        self.last_move_cost = 0

    def setHeuristic(self, heuristic):
        self.heuristic = heuristic

    def setGoals(self, goals):
        """Precompute and store goal position maps."""
        self.goal_maps = []
        for goal in goals:
            gmap = {}
            for i in range(len(goal)):
                for j in range(len(goal[0])):
                    gmap[goal[i][j]] = (i, j)
            self.goal_maps.append(gmap)

    def isGoal(self):
        """True if current board matches ANY of the goal states."""
        for gmap in self.goal_maps:
            match = True
            for i in range(self.n):
                for j in range(self.n):
                    t = self.matrix[i][j]
                    if gmap.get(t) != (i, j):
                        match = False
                        break
                if not match:
                    break
            if match:
                return True
        return False

    def computeHeuristic(self):
        return self.heuristic.execute(self)

    def addrow(self, row):
        self.matrix.append(row)

    def finalise(self):
        for i in range(self.n):
            for j in range(self.n):
                if self.matrix[i][j] == 0:
                    self.zeropos = (i, j)
                    break

    def swap(self, pos1, pos2):
        """Uniform cost: each move costs 1."""
        self.g += 1
        self.last_move_cost = 1
        self.matrix[pos1[0]][pos1[1]], self.matrix[pos2[0]][pos2[1]] = \
            self.matrix[pos2[0]][pos2[1]], self.matrix[pos1[0]][pos1[1]]
        self.zeropos = pos2

    def countInversions(self, reference_order):
        tile_rank = {tile: rank for rank, tile in enumerate(reference_order)}
        flatlist = [num for row in self.matrix for num in row if num != 0]
        ranked = [tile_rank[t] for t in flatlist if t in tile_rank]
        inversion = 0
        l = len(ranked)
        for i in range(l):
            for j in range(i + 1, l):
                if ranked[i] > ranked[j]:
                    inversion += 1
        return inversion

    def findBlankRow(self):
        return self.n - self.zeropos[0]

    def isSolvableToGoal(self, goal):
        """Check solvability for one specific goal."""
        reference_order = [goal[i][j] for i in range(self.n) for j in range(self.n) if goal[i][j] != 0]
        inv = self.countInversions(reference_order)
        if self.n % 2 == 1:
            return inv % 2 == 0
        blank_row = self.findBlankRow()
        if blank_row % 2 == 0 and inv % 2 == 1:
            return True
        if blank_row % 2 == 1 and inv % 2 == 0:
            return True
        return False

    def isSolvable(self, goals):
        """Solvable if reachable to AT LEAST one goal."""
        return any(self.isSolvableToGoal(goal) for goal in goals)


def solveNPuzzle(matrix):
    counter = itertools.count()
    pq = []
    heapq.heappush(pq, (matrix.g + matrix.computeHeuristic(), next(counter), matrix))
    best_g = {}

    def state_key(m):
        return tuple(tuple(row) for row in m.matrix)

    while pq:
        current = heapq.heappop(pq)[2]
        key = state_key(current)

        if key in best_g and best_g[key] <= current.g:
            continue
        best_g[key] = current.g

        if current.isGoal():
            path = []
            node = current
            while node is not None:
                path.append(node)
                node = node.parent
            path.reverse()
            return current.g, path

        zero_row, zero_col = current.zeropos
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dr, dc in directions:
            new_row, new_col = zero_row + dr, zero_col + dc
            if 0 <= new_row < current.n and 0 <= new_col < current.n:
                new_matrix = Matrix(current.n)
                new_matrix.copy(current)
                new_matrix.swap((zero_row, zero_col), (new_row, new_col))
                new_matrix.parent = current

                new_key = state_key(new_matrix)
                if new_key in best_g and best_g[new_key] <= new_matrix.g:
                    continue

                f = new_matrix.g + new_matrix.computeHeuristic()
                heapq.heappush(pq, (f, next(counter), new_matrix))

    return None, []


def print_board(m):
    for row in m.matrix:
        print(" ".join(str(x) for x in row))


def main():
    with open('input.txt', 'r') as file:
        n = int(file.readline().strip())

        matrix = Matrix(n)
        for _ in range(n):
            row = list(map(int, file.readline().strip().split()))
            matrix.addrow(row)
        matrix.finalise()

        k = int(file.readline().strip())   # number of goals
        goals = []
        for _ in range(k):
            goal = []
            for _ in range(n):
                row = list(map(int, file.readline().strip().split()))
                goal.append(row)
            goals.append(goal)

    heuristic = MultiGoalManhattan(goals)
    matrix.setHeuristic(heuristic)
    matrix.setGoals(goals)

    if not matrix.isSolvable(goals):
        print("Unsolvable puzzle")
        return

    total_cost, path = solveNPuzzle(matrix)

    print(f"Minimum total cost (moves): {total_cost}")
    print()
    for step, node in enumerate(path):
        if step == 0:
            print("Initial board:")
        else:
            print(f"Step {step} — move cost: {node.last_move_cost}, total cost so far: {node.g}")
        print_board(node)
        print()

    # Show which goal was reached
    print("Reached goal:")
    for gmap in matrix.goal_maps:
        # reconstruct goal board from gmap
        goal_board = [[0] * n for _ in range(n)]
        for tile, (r, c) in gmap.items():
            goal_board[r][c] = tile
        # check if final state matches this goal
        final = path[-1]
        if all(final.matrix[i][j] == goal_board[i][j] for i in range(n) for j in range(n)):
            for row in goal_board:
                print(" ".join(str(x) for x in row))
            break


if __name__ == "__main__":
    main()
