import math
import copy
import heapq
import itertools

# =============================================================================
# VARIANT 5: Position-Based Cost
# Cost of moving a tile = tile_value * (destination_row + 1)
# Moving into row 0 is cheapest, row n-1 is most expensive.
# Heuristic: weighted Manhattan using minimum multiplier = 1 (row 0),
#            keeping it admissible.
# =============================================================================

class Heuristic:
    def __init__(self, func):
        self.func = func
    def execute(self, Matrix):
        return self.func(Matrix)
    def setHeuristic(self, func):
        self.func = func


class ManhattanDistance(Heuristic):
    def __init__(self):
        super().__init__(self.calculate)
    def calculate(self, Matrix):
        md = 0
        n = Matrix.n
        for i in range(n):
            for j in range(n):
                if Matrix.matrix[i][j] != 0:
                    row = (Matrix.matrix[i][j] - 1) // n
                    col = (Matrix.matrix[i][j] - 1) % n
                    md += abs(i - row) + abs(j - col)
        return md


class PositionWeightedManhattan(Heuristic):
    """
    Admissible heuristic for position-based cost.
    Actual cost of a move = tile * (dest_row + 1), minimum multiplier = 1.
    So lower bound = tile * 1 * manhattan_dist(tile).
    """
    def __init__(self):
        super().__init__(self.calculate)

    def calculate(self, Matrix):
        n = Matrix.n
        total = 0
        for i in range(n):
            for j in range(n):
                t = Matrix.matrix[i][j]
                if t != 0:
                    goal_r = (t - 1) // n
                    goal_c = (t - 1) % n
                    total += t * (abs(i - goal_r) + abs(j - goal_c))
        return total


class Matrix:
    def __init__(self, n):
        self.n = n
        self.g = 0
        self.matrix = []
        self.zeropos = (-1, -1)
        self.heuristic = PositionWeightedManhattan()
        self.parent = None
        self.last_move_cost = 0

    def copy(self, other):
        self.n = other.n
        self.g = other.g
        self.matrix = copy.deepcopy(other.matrix)
        self.zeropos = other.zeropos
        self.heuristic = other.heuristic
        self.last_move_cost = 0

    def setHeuristic(self, heuristic):
        self.heuristic = heuristic

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
        """
        pos1 = blank position (where tile lands)
        cost = tile_value * (dest_row + 1)
        """
        tile = self.matrix[pos2[0]][pos2[1]]
        dest_row = pos1[0]  # blank's row = where tile is going
        cost = tile * (dest_row + 1)
        self.g += cost
        self.last_move_cost = cost

        self.matrix[pos1[0]][pos1[1]], self.matrix[pos2[0]][pos2[1]] = \
            self.matrix[pos2[0]][pos2[1]], self.matrix[pos1[0]][pos1[1]]
        self.zeropos = pos2

    def countInversions(self):
        flatlist = [num for row in self.matrix for num in row if num != 0]
        inversion = 0
        l = len(flatlist)
        for i in range(l):
            for j in range(i + 1, l):
                if flatlist[i] > flatlist[j]:
                    inversion += 1
        return inversion

    def findBlankRow(self):
        return self.n - self.zeropos[0]

    def isSolvable(self):
        if self.n % 2 == 1:
            return self.countInversions() % 2 == 0
        if self.n % 2 == 0:
            if self.findBlankRow() % 2 == 0 and self.countInversions() % 2 == 1:
                return True
            if self.findBlankRow() % 2 == 1 and self.countInversions() % 2 == 0:
                return True
        return False


def solveNPuzzle(matrix):
    heuristic = PositionWeightedManhattan()
    matrix.setHeuristic(heuristic)

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

        if current.computeHeuristic() == 0:
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

    if not matrix.isSolvable():
        print("Unsolvable puzzle")
        return

    total_cost, path = solveNPuzzle(matrix)

    print(f"Minimum total cost: {total_cost}")
    print()
    for step, node in enumerate(path):
        if step == 0:
            print("Initial board:")
        else:
            print(f"Step {step} — move cost: {node.last_move_cost}, total cost so far: {node.g}")
        print_board(node)
        print()


if __name__ == "__main__":
    main()
