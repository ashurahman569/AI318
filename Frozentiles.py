import math
import copy
import heapq
import itertools

# =============================================================================
# VARIANT 9: Frozen Tiles
# Some tiles cannot be moved at all.
# Frozen tile values are read from input.
# Input format:
#   n
#   n lines: initial board
#   f          <- number of frozen tiles
#   f tile values on one line (e.g. "3 7")
#
# Heuristic: Manhattan distance ignoring frozen tiles
#   (frozen tiles are already in place or we just can't move them —
#    they contribute 0 to heuristic since we can't fix them anyway).
# Cost: uniform (1 per move).
#
# Note: if a frozen tile is NOT in its goal position, the puzzle may be
# unsolvable. We check for this before running A*.
# =============================================================================

class Heuristic:
    def __init__(self, func):
        self.func = func
    def execute(self, Matrix):
        return self.func(Matrix)
    def setHeuristic(self, func):
        self.func = func


class FrozenAwareManhattan(Heuristic):
    """
    Manhattan distance that skips frozen tiles.
    Frozen tiles cannot move so they don't contribute to the heuristic.
    Only non-frozen, non-blank tiles are counted.
    """
    def __init__(self, frozen):
        super().__init__(self.calculate)
        self.frozen = frozen    # set of frozen tile values

    def calculate(self, Matrix):
        n = Matrix.n
        total = 0
        for i in range(n):
            for j in range(n):
                t = Matrix.matrix[i][j]
                if t != 0 and t not in self.frozen:
                    goal_r = (t - 1) // n
                    goal_c = (t - 1) % n
                    total += abs(i - goal_r) + abs(j - goal_c)
        return total


class Matrix:
    def __init__(self, n):
        self.n = n
        self.g = 0
        self.matrix = []
        self.zeropos = (-1, -1)
        self.heuristic = None
        self.parent = None
        self.last_move_cost = 0
        self.frozen = set()     # tile values that cannot be moved

    def copy(self, other):
        self.n = other.n
        self.g = other.g
        self.matrix = copy.deepcopy(other.matrix)
        self.zeropos = other.zeropos
        self.heuristic = other.heuristic
        self.frozen = other.frozen      # shared, never mutated
        self.last_move_cost = 0

    def setHeuristic(self, heuristic):
        self.heuristic = heuristic

    def setFrozen(self, frozen_tiles):
        self.frozen = set(frozen_tiles)

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

    def frozenTilesInPlace(self):
        """
        Returns True if all frozen tiles are already in their goal positions.
        A frozen tile at position (i,j) should have value i*n + j + 1.
        If any frozen tile is out of place, the puzzle is unsolvable.
        """
        n = self.n
        for i in range(n):
            for j in range(n):
                t = self.matrix[i][j]
                if t in self.frozen:
                    if t != i * n + j + 1:
                        return False
        return True

    def countInversions(self):
        """
        Count inversions only among non-frozen, non-blank tiles.
        Frozen tiles are fixed, so they're irrelevant to parity.
        """
        flatlist = [num for row in self.matrix for num in row
                    if num != 0 and num not in self.frozen]
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
        # First: all frozen tiles must already be in goal positions
        if not self.frozenTilesInPlace():
            return False
        # Then: standard parity check on non-frozen tiles
        if self.n % 2 == 1:
            return self.countInversions() % 2 == 0
        if self.n % 2 == 0:
            blank_row = self.findBlankRow()
            if blank_row % 2 == 0 and self.countInversions() % 2 == 1:
                return True
            if blank_row % 2 == 1 and self.countInversions() % 2 == 0:
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

                # KEY CHANGE: skip if the tile to move is frozen
                tile_to_move = current.matrix[new_row][new_col]
                if tile_to_move in current.frozen:
                    continue

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


def print_board(m, frozen=set()):
    for row in m.matrix:
        # Mark frozen tiles with * for clarity
        out = []
        for x in row:
            if x in frozen:
                out.append(f"{x}*")
            else:
                out.append(str(x))
        print(" ".join(out))


def main():
    with open('input.txt', 'r') as file:
        n = int(file.readline().strip())

        matrix = Matrix(n)
        for _ in range(n):
            row = list(map(int, file.readline().strip().split()))
            matrix.addrow(row)
        matrix.finalise()

        f = int(file.readline().strip())    # number of frozen tiles
        if f > 0:
            frozen_tiles = list(map(int, file.readline().strip().split()))
        else:
            frozen_tiles = []
            file.readline()                 # consume empty line if present

    matrix.setFrozen(frozen_tiles)
    heuristic = FrozenAwareManhattan(matrix.frozen)
    matrix.setHeuristic(heuristic)

    frozen = matrix.frozen
    print(f"Frozen tiles: {sorted(frozen) if frozen else 'None'}")
    print()

    if not matrix.isSolvable():
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
        print_board(node, frozen)
        print()


if __name__ == "__main__":
    main()
