Standard GRASP restarts from scratch every iteration, forgetting the structure of previously found good solutions. 
Path Relinking is an advanced intensification strategy. It maintains an "Elite Pool" of the best solutions found so far. 
After generating a new local optimum, it explores the "path" between this new solution and a randomly chosen elite solution 
by gradually changing the new solution's vertices to match the elite solution's partition, keeping track of the best solution found along this path.
Task: Implement Path Relinking. After a standard GRASP iteration finds a local optimum, select an elite solution. 
Identify the vertices that are in different partitions between the two solutions. Iteratively move one differing vertex to match the elite solution, 
evaluating the cut at each step. Update the elite pool if a better solution is found.

def PathRelinking(S_curr, Sprime_curr, S_guide, Sprime_guide, G):
    """
    Explores the path between current solution and guide solution.
    Returns the best solution found along the path.
    """
    S = set(S_curr)
    Sprime = set(Sprime_curr)
    
    best_S, best_Sprime = set(S), set(Sprime)
    best_cut = cutWeight(G, S, Sprime)
    
    # Find vertices that differ between current and guide
    # Vertices in S but should be in Sprime (according to guide)
    diff_to_Sprime = S.intersection(Sprime_guide)
    # Vertices in Sprime but should be in S (according to guide)
    diff_to_S = Sprime.intersection(S_guide)
    
    # Combine all differing vertices
    diff_set = list(diff_to_Sprime.union(diff_to_S))
    random.shuffle(diff_set) # Randomize the order of the path
    
    for v in diff_set:
        # Move v to match the guide's partition
        if v in S and v in Sprime_guide:
            S.remove(v)
            Sprime.add(v)
        elif v in Sprime and v in S_guide:
            Sprime.remove(v)
            S.add(v)
            
        # Evaluate the cut at this intermediate step
        current_cut = cutWeight(G, S, Sprime)
        if current_cut > best_cut:
            best_cut = current_cut
            best_S, best_Sprime = set(S), set(Sprime)
            
    return best_S, best_Sprime, best_cut

def GRASP_PathRelinking(maxIter, G, alpha=0.5, elite_pool_size=5):
    elite_pool = [] # Stores tuples of (cut_value, S, Sprime)
    xStar, wStar = None, -1
    
    for i in range(1, maxIter + 1):
        # 1. Standard GRASP iteration
        X, Y = SemiGreedyMaxCut(G, alpha)
        X, Y = LocalSearch(X, Y, G)
        w = cutWeight(G, X, Y)
        
        if w > wStar:
            xStar, wStar = (X, Y), w
            
        # 2. Update Elite Pool
        elite_pool.append((w, set(X), set(Y)))
        elite_pool.sort(key=lambda item: item[0], reverse=True)
        if len(elite_pool) > elite_pool_size:
            elite_pool.pop() # Remove the worst
            
        # 3. Path Relinking (if we have enough elite solutions)
        if len(elite_pool) >= 2:
            # Pick a random elite solution (excluding the current one if possible)
            guide_w, guide_S, guide_Sprime = random.choice(elite_pool)
            
            pr_S, pr_Sprime, pr_w = PathRelinking(X, Y, guide_S, guide_Sprime, G)
            
            # Apply Local Search to the best solution found on the path
            pr_S, pr_Sprime = LocalSearch(pr_S, pr_Sprime, G)
            pr_w = cutWeight(G, pr_S, pr_Sprime)
            
            if pr_w > wStar:
                xStar, wStar = (pr_S, pr_Sprime), pr_w
                
            # Add the PR result back to the elite pool
            elite_pool.append((pr_w, set(pr_S), set(pr_Sprime)))
            elite_pool.sort(key=lambda item: item[0], reverse=True)
            if len(elite_pool) > elite_pool_size:
                elite_pool.pop()
                
    return xStar, wStar
