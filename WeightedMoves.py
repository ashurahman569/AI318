import math
import copy
import heapq
import itertools


class Heuristic:
    def __init__(self, func):
        self.func = func

    def execute(self, Matrix):
        return self.func(Matrix)

    def setHeuristic(self, func):
        self.func = func


class HammingDistance(Heuristic):
    def __init__(self):
        super().__init__(self.calculate)

    def calculate(self, Matrix):
        hd = 0
        n = Matrix.n
        for i in range(n):
            for j in range(n):
                if Matrix.matrix[i][j] != 0 and Matrix.matrix[i][j] != (i * n + j + 1):
                    hd += 1
        return hd


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


class EuclideanDistance(Heuristic):
    def __init__(self):
        super().__init__(self.calculate)

    def calculate(self, Matrix):
        ed = 0
        n = Matrix.n
        for i in range(n):
            for j in range(n):
                if Matrix.matrix[i][j] != 0:
                    row = (Matrix.matrix[i][j] - 1) // n
                    col = (Matrix.matrix[i][j] - 1) % n
                    ed += math.sqrt((i - row) ** 2 + (j - col) ** 2)
        return ed


class LinearConflict(Heuristic):
    def __init__(self):
        super().__init__(self.calculate)

    def calculate(self, Matrix):
        n = Matrix.n
        conflict = 0
        for i in range(n):
            row_conflict = 0
            col_conflict = 0
            for j in range(n):
                if Matrix.matrix[i][j] != 0 and (Matrix.matrix[i][j] - 1) // n == i:
                    for k in range(j + 1, n):
                        if Matrix.matrix[i][k] != 0 and (Matrix.matrix[i][k] - 1) // n == i and Matrix.matrix[i][j] > Matrix.matrix[i][k]:
                            row_conflict += 1
                if Matrix.matrix[j][i] != 0 and (Matrix.matrix[j][i] - 1) % n == i:
                    for k in range(j + 1, n):
                        if Matrix.matrix[k][i] != 0 and (Matrix.matrix[k][i] - 1) % n == i and Matrix.matrix[j][i] > Matrix.matrix[k][i]:
                            col_conflict += 1
            conflict += row_conflict + col_conflict

        return ManhattanDistance().calculate(Matrix) + 2 * conflict


class CustomHeuristic(Heuristic):  # Corner Tiles basically
    def __init__(self):
        super().__init__(self.calculate)

    def calculate(self, Matrix):
        n = Matrix.n
        md = ManhattanDistance().calculate(Matrix)
        penalty = 0

        def goal_val(r, c):
            return r * n + c + 1

        corners = [
            ((0, 0), (0, 1), (1, 0)),
            ((0, n - 1), (0, n - 2), (1, n - 1)),
            ((n - 1, 0), (n - 2, 0), (n - 1, 1))
        ]

        for (cr, cc), (nr1, nc1), (nr2, nc2) in corners:
            corner_tile = Matrix.matrix[cr][cc]
            if corner_tile != goal_val(cr, cc):
                neighbor1_correct = Matrix.matrix[nr1][nc1] == goal_val(nr1, nc1)
                neighbor2_correct = Matrix.matrix[nr2][nc2] == goal_val(nr2, nc2)
                if neighbor1_correct and neighbor2_correct:
                    penalty += 2

        return md + penalty


class WeightedManhattan(Heuristic):
    """
    Heuristic for the non-uniform cost variant.
    h(n) = sum over all tiles t of: t * (|r_t - r*_t| + |c_t - c*_t|)
    Tile fatigue is ignored in the heuristic (only applied in g).
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
        self.g = 0               # total path cost (not step count)
        self.matrix = []
        self.zeropos = (-1, -1)
        self.heuristic = WeightedManhattan()
        self.parent = None
        self.move_counts = {}    # {tile_value: number_of_times_moved}
        self.last_move_cost = 0  # cost of the single move that created this state

    def copy(self, other):
        self.n = other.n
        self.g = other.g         # will be incremented in swap()
        self.matrix = copy.deepcopy(other.matrix)
        self.zeropos = other.zeropos
        self.heuristic = other.heuristic
        self.move_counts = copy.deepcopy(other.move_counts)
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
        pos1 is the blank position, pos2 is the tile sliding in.
        Cost = tile_value * (times_moved_so_far + 1)
        """
        tile = self.matrix[pos2[0]][pos2[1]]   # tile moving into the blank
        times = self.move_counts.get(tile, 0)
        cost = tile * (times + 1)
        self.move_counts[tile] = times + 1
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
            if self.countInversions() % 2 == 0:
                return True
        if self.n % 2 == 0:
            if self.findBlankRow() % 2 == 0 and self.countInversions() % 2 == 1:
                return True
            elif self.findBlankRow() % 2 == 1 and self.countInversions() % 2 == 0:
                return True
        return False


def solveNPuzzle(matrix) -> tuple:
    heuristic = WeightedManhattan()
    matrix.setHeuristic(heuristic)

    counter = itertools.count()
    pq = []
    heapq.heappush(pq, (matrix.computeHeuristic() + matrix.g, next(counter), matrix))
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
            # Reconstruct path
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
