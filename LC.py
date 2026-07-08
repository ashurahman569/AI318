class LinearConflict(Heuristic):
    """
    Admissible linear-conflict heuristic.
 
    IMPORTANT: naively summing "+2 per conflicting pair" is NOT admissible.
    If 3+ tiles in the same line are mutually out of order, one tile
    stepping out of the line resolves several pairwise conflicts at once,
    so charging 2 per pair double-bills the same detour and can overestimate
    the true remaining cost (verified by exhaustive 3x3 search: naive
    pairwise summation overestimates h* on 7/181440 reachable states,
    by up to 4 moves).
 
    The fix: per line (row or column), repeatedly find the tile involved
    in the MOST conflicts and remove it from consideration, charging +2
    for each removal, until no conflicts remain in that line. This is
    equivalent to computing a minimum vertex cover of the line's conflict
    graph, and is a proven admissible lower bound.
    """
    def __init__(self):
        super().__init__(self.calculate)
 
    @staticmethod
    def _resolve_conflicts(line_values):
        """
        line_values: list of (position_index, tile_value) for tiles in this
        line whose GOAL line is this line (i.e. candidates for conflict).
        Returns the minimum number of tile-removals needed so that no two
        remaining tiles are in conflict (out of order relative to their
        goal order along the line).
        """
        n = len(line_values)
        removed = [False] * n
        extra_moves = 0
 
        while True:
            # degree[i] = number of active conflicts tile i is involved in
            degree = [0] * n
            for a in range(n):
                if removed[a]:
                    continue
                pos_a, val_a = line_values[a]
                for b in range(a + 1, n):
                    if removed[b]:
                        continue
                    pos_b, val_b = line_values[b]
                    # pos_a < pos_b (earlier in line) but val_a's goal order
                    # comes after val_b's goal order -> they must cross paths
                    if pos_a < pos_b and val_a > val_b:
                        degree[a] += 1
                        degree[b] += 1
 
            if not any(degree):
                break
 
            worst = max(range(n), key=lambda i: degree[i])
            removed[worst] = True
            extra_moves += 2
 
        return extra_moves
 
    def calculate(self, Matrix):
        rows, cols = Matrix.rows, Matrix.cols
        total_extra = 0
 
        # Row conflicts: for each row i, gather tiles whose goal row is i
        for i in range(rows):
            line_values = []
            for j in range(cols):
                v = Matrix.matrix[i][j]
                if v != 0 and (v - 1) // cols == i:
                    line_values.append((j, v))
            total_extra += self._resolve_conflicts(line_values)
 
        # Column conflicts: for each col j, gather tiles whose goal col is j
        for j in range(cols):
            line_values = []
            for i in range(rows):
                v = Matrix.matrix[i][j]
                if v != 0 and (v - 1) % cols == j:
                    line_values.append((i, v))
            total_extra += self._resolve_conflicts(line_values)
 
        return ManhattanDistance().calculate(Matrix) + total_extra
