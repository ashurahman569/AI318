# Add to Matrix:
self.goal = None  # set after reading input

# New method in Matrix:
def setGoal(self, goal_matrix):
    self.goal = goal_matrix  # 2D list

def computeHeuristic(self):
    return self.heuristic.execute(self)

# New heuristic that uses custom goal:
class CustomGoalManhattan(Heuristic):
    def __init__(self, goal):
        super().__init__(self.calculate)
        # Precompute goal positions for O(1) lookup
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

# In solveNPuzzle(), goal check changes:
def isGoal(current, goal_pos):
    for i in range(current.n):
        for j in range(current.n):
            t = current.matrix[i][j]
            if t != 0:
                if goal_pos.get(t) != (i, j):
                    return False
    return True
# Replace: if current.computeHeuristic() == 0:
# With:    if isGoal(current, goal_pos):
