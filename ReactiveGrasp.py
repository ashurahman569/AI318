In standard GRASP, α is fixed (e.g., 0.5). However, different graphs require different levels of greediness. Reactive GRASP solves this 
by maintaining a pool of α values (e.g., 0.1, 0.3, 0.5, 0.7, 0.9). It tracks which α values produced the best solutions in previous iterations. 
If an α yields a new global best, its probability of being chosen increases; otherwise, it decreases.
Task: Implement Reactive GRASP and compare it to standard GRASP with a fixed α=0.5.

def ReactiveGRASP(maxIter, G):
    # Pool of alpha values and their initial uniform probabilities
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    probs = [1.0 / len(alphas)] * len(alphas)
    
    xStar = None
    wStar = -1
    
    for i in range(1, maxIter + 1):
        # 1. Select alpha based on current probabilities
        r = random.random()
        cumulative = 0.0
        chosen_alpha = alphas[-1]
        for idx, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                chosen_alpha = alphas[idx]
                break
                
        # 2. Build solution and apply Local Search
        X, Y = SemiGreedyMaxCut(G, chosen_alpha)
        X, Y = LocalSearch(X, Y, G)
        w = cutWeight(G, X, Y)
        
        # 3. Update best solution
        if i == 1 or w > wStar:
            xStar = (X, Y)
            wStar = w
            
            # 4. Reactive update: Increase prob of chosen_alpha, decrease others
            # Simple multiplicative update rule
            factor = 1.2 
            new_probs = []
            for idx, a in enumerate(alphas):
                if a == chosen_alpha:
                    new_probs.append(probs[idx] * factor)
                else:
                    new_probs.append(probs[idx] / factor)
            
            # Normalize probabilities so they sum to 1
            sum_p = sum(new_probs)
            probs = [p / sum_p for p in new_probs]
            
    return xStar, wStar
