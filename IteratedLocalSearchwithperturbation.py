Standard GRASP restarts the construction phase from scratch every iteration. Iterated Local Search (ILS) is more efficient: 
it takes the best solution found so far, applies a random "perturbation" (e.g., randomly flipping 10% of the vertices to the other side), 
and then runs Local Search on this perturbed solution. This allows the algorithm to jump out of deep local optima without starting from zero.
Task: Implement ILS by adding a perturbation step to your GRASP loop and compare its convergence speed against standard GRASP.

def PerturbSolution(X, Y, G, perturbation_rate=0.1):
    """Randomly flips a percentage of vertices to escape local optima."""
    n = G.n
    X = set(X)
    Y = set(Y)
    
    num_to_flip = int(n * perturbation_rate)
    all_vertices = list(range(1, n + 1))
    vertices_to_flip = random.sample(all_vertices, num_to_flip)
    
    for v in vertices_to_flip:
        if v in X:
            X.remove(v)
            Y.add(v)
        else:
            Y.remove(v)
            X.add(v)
            
    return X, Y

def IteratedLocalSearch(maxIter, G, alpha=0.5, perturbation_rate=0.1):
    # 1. Get initial good solution using standard GRASP
    (X, Y), wStar = GRASP(10, G, alpha) # Run a few standard iterations first
    
    for i in range(maxIter):
        # 2. Perturb the current best solution
        X_pert, Y_pert = PerturbSolution(X, Y, G, perturbation_rate)
        
        # 3. Apply Local Search to the perturbed solution
        X_ls, Y_ls = LocalSearch(X_pert, Y_pert, G)
        w = cutWeight(G, X_ls, Y_ls)
        
        # 4. Accept the new solution if it's better (or equal)
        if w >= wStar:
            X, Y = X_ls, Y_ls
            wStar = w
            
    return (X, Y), wStar
