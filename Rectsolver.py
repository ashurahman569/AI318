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
        rows, cols = Matrix.rows, Matrix.cols
        for i in range(rows):
            for j in range(cols):
                if Matrix.matrix[i][j] != 0 and Matrix.matrix[i][j] != (i * cols + j + 1):
                    hd += 1
        return hd

class ManhattanDistance(Heuristic):
    def __init__(self):
        super().__init__(self.calculate)

    def calculate(self, Matrix):
        md = 0
        rows, cols = Matrix.rows, Matrix.cols
        for i in range(rows):
            for j in range(cols):
                v = Matrix.matrix[i][j]
                if v != 0:
                    goal_row = (v - 1) // cols
                    goal_col = (v - 1) % cols
                    md += abs(i - goal_row) + abs(j - goal_col)
        return md

class EuclideanDistance(Heuristic):
    def __init__(self):
        super().__init__(self.calculate)

    def calculate(self, Matrix):
        ed = 0
        rows, cols = Matrix.rows, Matrix.cols
        for i in range(rows):
            for j in range(cols):
                v = Matrix.matrix[i][j]
                if v != 0:
                    goal_row = (v - 1) // cols
                    goal_col = (v - 1) % cols
                    ed += math.sqrt((i - goal_row) ** 2 + (j - goal_col) ** 2)
        return ed

class LinearConflict(Heuristic):
    def __init__(self):
        super().__init__(self.calculate)

    def calculate(self, Matrix):
        rows, cols = Matrix.rows, Matrix.cols
        conflict = 0

        # Row conflicts: for each row i, look along the cols
        for i in range(rows):
            row_conflict = 0
            for j in range(cols):
                v1 = Matrix.matrix[i][j]
                if v1 != 0 and (v1 - 1) // cols == i:  # v1's goal row is this row
                    for k in range(j + 1, cols):
                        v2 = Matrix.matrix[i][k]
                        if v2 != 0 and (v2 - 1) // cols == i and v1 > v2:
                            row_conflict += 1
            conflict += row_conflict

        # Column conflicts: for each col j, look along the rows
        for j in range(cols):
            col_conflict = 0
            for i in range(rows):
                v1 = Matrix.matrix[i][j]
                if v1 != 0 and (v1 - 1) % cols == j:  # v1's goal col is this col
                    for k in range(i + 1, rows):
                        v2 = Matrix.matrix[k][j]
                        if v2 != 0 and (v2 - 1) % cols == j and v1 > v2:
                            col_conflict += 1
            conflict += col_conflict

        return ManhattanDistance().calculate(Matrix) + 2 * conflict

class CustomHeuristic(Heuristic):  # Corner Tiles
    def __init__(self):
        super().__init__(self.calculate)

    def calculate(self, Matrix):
        rows, cols = Matrix.rows, Matrix.cols
        md = ManhattanDistance().calculate(Matrix)
        penalty = 0

        def goal_val(r, c):
            return r * cols + c + 1

        # Only build a corner check if the grid actually has room for
        # the two neighbor cells that corner needs (guards rows==1 / cols==1).
        # NOTE: the bottom-right corner is deliberately excluded -- that cell
        # holds the blank (0) in the goal state, so it can never equal its
        # "goal value" and would make the heuristic never reach 0.
        corners = []
        if rows >= 2 and cols >= 2:
            corners.append(((0, 0), (0, 1), (1, 0)))
            corners.append(((0, cols - 1), (0, cols - 2), (1, cols - 1)))
            corners.append(((rows - 1, 0), (rows - 2, 0), (rows - 1, 1)))

        for (cr, cc), (nr1, nc1), (nr2, nc2) in corners:
            corner_tile = Matrix.matrix[cr][cc]
            if corner_tile != goal_val(cr, cc):
                neighbor1_correct = Matrix.matrix[nr1][nc1] == goal_val(nr1, nc1)
                neighbor2_correct = Matrix.matrix[nr2][nc2] == goal_val(nr2, nc2)
                if neighbor1_correct and neighbor2_correct:
                    penalty += 2

        return md + penalty

class WalkingDistance(Heuristic):
    def __init__(self):
        super().__init__(self.calculate)

    def calculate(self, Matrix):
        rows, cols = Matrix.rows, Matrix.cols
        row_cost = 0
        col_cost = 0
        for i in range(rows):
            for j in range(cols):
                v = Matrix.matrix[i][j]
                if v != 0:
                    goal_row = (v - 1) // cols
                    goal_col = (v - 1) % cols
                    if goal_row != i:
                        row_cost += 1
                    if goal_col != j:
                        col_cost += 1
        return row_cost + col_cost

class Matrix:
    def __init__(self, rows, cols, g=-1):
        self.rows = rows
        self.cols = cols
        self.g = g + 1  # initial moves 0 if this is the first matrix
        self.matrix = []
        self.zeropos = (-1, -1)
        self.heuristic = HammingDistance()
        self.parent = None

    def copy(self, other):
        self.rows = other.rows
        self.cols = other.cols
        self.g = other.g + 1  # one move made
        self.matrix = copy.deepcopy(other.matrix)
        self.zeropos = other.zeropos
        self.heuristic = other.heuristic

    def setHeuristic(self, heuristic):
        self.heuristic = heuristic

    def computeHeuristic(self):
        return self.heuristic.execute(self)

    def addrow(self, row):
        self.matrix.append(row)

    def finalise(self):
        for i in range(self.rows):
            for j in range(self.cols):
                if self.matrix[i][j] == 0:
                    self.zeropos = (i, j)
                    return

    def swap(self, pos1, pos2):
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

    def findBlankRowFromBottom(self):
        # 1-indexed distance of the blank's row from the bottom row
        return self.rows - self.zeropos[0]

    def isSolvable(self):
        # Classic 15-puzzle rule generalized to rectangles:
        # parity branch depends on the GRID WIDTH (number of columns),
        # not on the total size, because a vertical blank-move shifts
        # the flattened array by (cols) positions -> parity flips when cols is even.
        inversions = self.countInversions()
        if self.cols % 2 == 1:
            return inversions % 2 == 0
        else:
            blank_row_from_bottom = self.findBlankRowFromBottom()
            if blank_row_from_bottom % 2 == 0:
                return inversions % 2 == 1
            else:
                return inversions % 2 == 0


def solveNPuzzle(matrix, W=1.0):
    heuristic = CustomHeuristic()  # change from here
    matrix.setHeuristic(heuristic)

    counter = itertools.count()
    pq = []
    heapq.heappush(pq, (W * matrix.computeHeuristic() + matrix.g, next(counter), matrix))
    best_g = {}

    def state_key(m):
        return tuple(tuple(row) for row in m.matrix)

    explored = 0
    while pq:
        current = heapq.heappop(pq)[2]
        key = state_key(current)
        if key in best_g and best_g[key] <= current.g:
            continue
        best_g[key] = current.g
        explored += 1

        if current.computeHeuristic() == 0:
            path = []
            node = current
            while node is not None:
                path.append(node)
                node = node.parent
            path.reverse()
            return current.g, explored, path

        zero_row, zero_col = current.zeropos
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # WASD

        for dr, dc in directions:
            new_row, new_col = zero_row + dr, zero_col + dc
            if 0 <= new_row < current.rows and 0 <= new_col < current.cols:
                new_matrix = Matrix(current.rows, current.cols)
                new_matrix.copy(current)
                new_matrix.swap((zero_row, zero_col), (new_row, new_col))
                new_matrix.parent = current

                new_key = state_key(new_matrix)
                if new_key in best_g and best_g[new_key] <= new_matrix.g:
                    continue

                heapq.heappush(pq, (W * new_matrix.computeHeuristic() + new_matrix.g, next(counter), new_matrix))

    return None  # no solution found (shouldn't happen if isSolvable() passed)


def main():
    weights = [1.0, 1.2, 2.0, 5.0]
    # input.txt format per puzzle block:
    #   line 1: "rows cols"
    #   next `rows` lines: `cols` integers each (0 = blank)
    with open('input.txt', 'r') as file:
        for i in range(10):
            first_line = file.readline().strip()
            if not first_line:
                break
            rows_str, cols_str = first_line.split()
            rows, cols = int(rows_str), int(cols_str)

            matrix = Matrix(rows, cols)
            for x in range(rows):
                row = list(map(int, file.readline().strip().split()))
                matrix.addrow(row)
            matrix.finalise()

            if not matrix.isSolvable():
                print("Unsolvable puzzle")
            else:
                print("Puzzle :" + str(i + 1))
                for W in weights:
                    result = solveNPuzzle(matrix, W)
                    if result is None:
                        print("No solution found")
                        continue
                    moves, nodes, path = result
                    print(f"Weight: {W}")
                    print(f"Minimum number of moves: {moves}")
                    print(f"Number of nodes explored: {nodes}")
                    # print("Moves:")
                    # for state in path:
                    #     for row in state.matrix:
                    #         print(row)
                    #     print()

if __name__ == "__main__":
    main()
