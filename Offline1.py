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
    
class CustomHeuristic(Heuristic):  #Corner Tiles basically
    def __init__(self):
        super().__init__(self.calculate)

    def calculate(self, Matrix):
        n = Matrix.n
        md = ManhattanDistance().calculate(Matrix)
        penalty = 0

        def goal_val(r, c):
            return r * n + c + 1

        corners = [((0, 0), (0, 1), (1, 0)), ((0, n - 1), (0, n - 2), (1, n - 1)), ((n - 1, 0), (n - 2, 0), (n - 1, 1))]

        for (cr, cc), (nr1, nc1), (nr2, nc2) in corners:
            corner_tile = Matrix.matrix[cr][cc]
            if corner_tile != goal_val(cr, cc):
                neighbor1_correct = Matrix.matrix[nr1][nc1] == goal_val(nr1, nc1)
                neighbor2_correct = Matrix.matrix[nr2][nc2] == goal_val(nr2, nc2)
                if neighbor1_correct and neighbor2_correct:
                    penalty += 2

        return md + penalty

class Matrix:
    def __init__(self, n, g=-1):
        self.n = n
        self.g = g + 1 #initial moves 0 if its the first matrix
        self.matrix = []
        self.zeropos = (-1,-1)
        self.heuristic = HammingDistance()
        self.parent = None

    def copy(self, other):
        self.n = other.n
        self.g = other.g + 1 #one move made
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
        for i in range(self.n):
            for j in range(self.n):
                if self.matrix[i][j] == 0:
                    self.zeropos = (i, j)
                    break

    def swap(self, pos1, pos2):
        self.matrix[pos1[0]][pos1[1]], self.matrix[pos2[0]][pos2[1]] = self.matrix[pos2[0]][pos2[1]], self.matrix[pos1[0]][pos1[1]]
        self.zeropos = pos2

    def countInversions(self):
        flatlist = [num for row in self.matrix for num in row if num != 0]  #making a single list
        inversion = 0
        l = len(flatlist)
        for i in range(l):
            for j in range(i + 1, l):
                if flatlist[i] > flatlist[j]:
                    inversion += 1
        return inversion
    
    def findBlankRow(self):
        return self.n - self.zeropos[0]  #row number of the blank   
     
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
    
def solveNPuzzle(matrix, W =1.0) -> int:
    heuristic = CustomHeuristic() #change from here
    matrix.setHeuristic(heuristic)

    counter = itertools.count()
    pq = []
    heapq.heappush(pq, (W * matrix.computeHeuristic() + matrix.g, next(counter), matrix))
    best_g = {}  #best g seen so far

    def state_key(m):
        return tuple(tuple(row) for row in m.matrix)
    explored = 0
    while pq:
        current = heapq.heappop(pq)[2]
        key = state_key(current)
        # Skip stale/duplicate entries
        if key in best_g and best_g[key] <= current.g:
            continue
        best_g[key] = current.g
        explored+=1

        if current.computeHeuristic() == 0:
            path = []
            node = current
            while node is not None:
                path.append(node)
                node = node.parent
            path.reverse()
            return current.g, explored, path
        
        zero_row, zero_col = current.zeropos
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] #WASD

        for dr, dc in directions:
            new_row, new_col = zero_row + dr, zero_col + dc
            if 0 <= new_row < current.n and 0 <= new_col < current.n:
                new_matrix = Matrix(current.n)
                new_matrix.copy(current)
                new_matrix.swap((zero_row, zero_col), (new_row, new_col))
                new_matrix.parent = current  # for path

                new_key = state_key(new_matrix)
                if new_key in best_g and best_g[new_key] <= new_matrix.g:
                    continue  #already reached this state at least as cheaply

                heapq.heappush(pq, (W * new_matrix.computeHeuristic() + new_matrix.g, next(counter), new_matrix))                

def main():
    weights = [1.0, 1.2, 2.0, 5.0] 
    with open('input.txt', 'r') as file:
        for i in range (10):
            
            n = int(file.readline().strip())
            matrix = Matrix(n)
            #next n lines to fill the matrix
            for x in range(n):
                row = list(map(int, file.readline().strip().split()))
                matrix.addrow(row)
            matrix.finalise()    
        
            if not matrix.isSolvable():
                print("Unsolvable puzzle")
            else:
                print("Puzzle :" + str(i+1))
                for W in weights:
                    moves, nodes, path = solveNPuzzle(matrix, W)
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
