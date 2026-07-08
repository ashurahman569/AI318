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
