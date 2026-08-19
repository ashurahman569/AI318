Your current Semi-Greedy uses a Value-based RCL (using the α threshold). 
A common alternative is the Cardinality-based RCL. Instead of calculating a threshold, 
you simply sort all unassigned vertices by their greedy value g(v) in descending order, 
and select the top K vertices to form the RCL. You then pick uniformly from this top-K list.
Task: Implement Cardinality-based Semi-Greedy with K=5 and compare its GRASP performance against the Value-based approach.

def SemiGreedyMaxCutCardinality(G, K=5):
    n = G.n
    adj = G.adj
    
    # Seed with heaviest edge
    maxu, maxv, maxw = -1, -1, -1e18
    for u in range(1, n + 1):
        for (v, w) in adj[u]:
            if w > maxw:
                maxw = w
                maxu, maxv = u, v
                
    X = {maxu}
    Y = {maxv}
    Vprime = set(range(1, n + 1)) - X - Y
    
    sigmaX = {v: 0 for v in Vprime}
    sigmaY = {v: 0 for v in Vprime}
    for v in Vprime:
        for (x, w) in adj[v]:
            if x == maxu: sigmaX[v] += w
            elif x == maxv: sigmaY[v] += w
            
    while Vprime:
        # Calculate greedy values
        greedy_val = {v: max(sigmaX[v], sigmaY[v]) for v in Vprime}
        
        # Sort vertices by greedy value descending
        sorted_V = sorted(Vprime, key=lambda v: greedy_val[v], reverse=True)
        
        # Take top K for RCL (or all if less than K remain)
        RCL = sorted_V[:K]
        
        # Pick uniformly from RCL
        chosen = random.choice(RCL)
        
        # Greedy placement rule
        went_to_X = sigmaX[chosen] < sigmaY[chosen]
        if went_to_X: X.add(chosen)
        else: Y.add(chosen)
            
        Vprime.remove(chosen)
        del sigmaX[chosen]
        del sigmaY[chosen]
        
        # Update sigmas for remaining candidates
        for (x, w) in adj[chosen]:
            if x in Vprime:
                if went_to_X: sigmaX[x] += w
                else: sigmaY[x] += w
                    
    return X, Y
