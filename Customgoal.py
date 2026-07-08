import math
import copy
import heapq
import itertools

# =============================================================================
# VARIANT 6: Custom Goal State
# The goal state is read from input instead of assuming 1,2,3...0.
# Input format:
#   n
#   n lines: initial board
#   n lines: goal board
#
# Heuristic: Manhattan distance to the custom goal.
# Cost: uniform (1 per move) — combine with other variants if needed.
# =============================================================================

class Heuristic:
    def __init__(self, func):
        self.func = func
    def execute(self, Matrix):
        return self.func(Matrix)
    def setHeuristic(self, func):
        self.func = func


class CustomGoalManhattan(Heuristic):
    """
    Manhattan distance where goal positions come from a custom goal board.
    goal_pos: dict {tile_value: (goal_row, goal_col)}
    Precomputed once from the goal board for O(1) lookup per tile.
    """
    def __init__(self, goal):
        super().__init__(self.calculate)
        self.goal_pos = {}
        for i in range(len(goal)):
            for j in range(len(goal[0])):
                self.goal_pos[goal[i][j]] = (i, j)

    def calculate(self, Matrix):
        total = 0
        for i in range(Matrix.n):
            for j in range(Matrix.n):
                t = Matrix.matrix[i][j]
                if t != 0 and t in self.goal_pos:
                    gr, gc = self.goal_pos[t]
                    total += abs(i - gr) + abs(j - gc)
        return total


class Matrix:
    def __init__(self, n):
        self.n = n
        self.g = 0
        self.matrix = []
        self.zeropos = (-1, -1)
        self.heuristic = None   # set after goal is known
        self.parent = None
        self.last_move_cost = 0
        self.goal_pos = {}      # {tile: (row, col)} for goal check

    def copy(self, other):
        self.n = other.n
        self.g = other.g
        self.matrix = copy.deepcopy(other.matrix)
        self.zeropos = other.zeropos
        self.heuristic = other.heuristic
        self.goal_pos = other.goal_pos  # shared reference, never mutated
        self.last_move_cost = 0

    def setHeuristic(self, heuristic):
        self.heuristic = heuristic

    def setGoal(self, goal):
        """Store goal positions for fast goal checking."""
        self.goal_pos = {}
        for i in range(len(goal)):
            for j in range(len(goal[0])):
                self.goal_pos[goal[i][j]] = (i, j)

    def isGoal(self):
        """Check if current board matches the custom goal."""
        for i in range(self.n):
            for j in range(self.n):
                t = self.matrix[i][j]
                if t != 0 and self.goal_pos.get(t) != (i, j):
                    return False
                if t == 0 and self.goal_pos.get(0) != (i, j):
                    return False
        return True

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
        """
        Count inversions relative to the goal's tile ordering.
        reference_order: list of tiles in goal reading order (excluding 0)
        """
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

    def isSolvable(self, goal):
        """
        Solvability check accounting for custom goal.
        We count inversions in the initial state relative to the goal ordering,
        and also check the blank row parity for even-sized boards.
        """
        reference_order = [goal[i][j] for i in range(self.n) for j in range(self.n) if goal[i][j] != 0]
        inv = self.countInversions(reference_order)

        if self.n % 2 == 1:
            return inv % 2 == 0
        if self.n % 2 == 0:
            blank_row = self.findBlankRow()
            if blank_row % 2 == 0 and inv % 2 == 1:
                return True
            if blank_row % 2 == 1 and inv % 2 == 0:
                return True
        return False


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

        # Use custom goal check instead of h==0
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


def print_board(matrix):
    for row in matrix.matrix:
        print(" ".join(str(x) for x in row))


def main():
    with open('input.txt', 'r') as file:
        n = int(file.readline().strip())

        matrix = Matrix(n)
        for _ in range(n):
            row = list(map(int, file.readline().strip().split()))
            matrix.addrow(row)
        matrix.finalise()

        # Read goal board
        goal = []
        for _ in range(n):
            row = list(map(int, file.readline().strip().split()))
            goal.append(row)

    # Set heuristic and goal positions
    heuristic = CustomGoalManhattan(goal)
    matrix.setHeuristic(heuristic)
    matrix.setGoal(goal)

    if not matrix.isSolvable(goal):
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

    print("Goal board:")
    for row in goal:
        print(" ".join(str(x) for x in row))


if __name__ == "__main__":
    main()
