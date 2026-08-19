Your current Local Search uses a 1-flip neighborhood (moving one vertex at a time). It can get stuck in a local optimum where no single vertex wants to move,
but swapping two vertices (one from 𝑋 to 𝑌, and one from 𝑌 to 𝑋) would improve the cut.
Task: Implement a Swap Local Search. Instead of just evaluating single flips, evaluate the combined gain 
Δ(𝑢,𝑣) of swapping 𝑢∈𝑋 and 𝑣∈𝑌. The gain is Δ(𝑢,𝑣)=𝛿(𝑢)+𝛿(𝑣)−2⋅𝑤𝑢𝑣   (where 𝑤𝑢𝑣 is the edge weight between them, 
or 0 if no edge exists). Apply the best improving swap until no swap yields a positive gain.

def LocalSearchSwap(S, Sprime, G):
    n = G.n
    adj = G.adj
    S = set(S)
    Sprime = set(Sprime)
    
    # Precompute sigmas
    sigmaS = {v: 0 for v in range(1, n + 1)}
    sigmaSprime = {v: 0 for v in range(1, n + 1)}
    for v in range(1, n + 1):
        for (x, w) in adj[v]:
            if x in S: sigmaS[v] += w
            else: sigmaSprime[v] += w
            
    # Precompute edge weights in a dictionary for O(1) lookup
    edge_weights = {}
    for u in range(1, n + 1):
        for (v, w) in adj[u]:
            if u < v:
                edge_weights[(u, v)] = w
                edge_weights[(v, u)] = w

    while True:
        max_delta = 0
        best_u, best_v = None, None
        
        # Evaluate all possible swaps between S and Sprime
        for u in S:
            delta_u = sigmaSprime[u] - sigmaS[u] # Gain if u moves to Sprime
            for v in Sprime:
                delta_v = sigmaS[v] - sigmaSprime[v] # Gain if v moves to S
                
                w_uv = edge_weights.get((u, v), 0)
                total_delta = delta_u + delta_v - 2 * w_uv
                
                if total_delta > max_delta:
                    max_delta = total_delta
                    best_u, best_v = u, v
                    
        if best_u is None:
            break # No improving swap found, local optimum reached
            
        # Perform the swap
        S.remove(best_u)
        Sprime.add(best_u)
        Sprime.remove(best_v)
        S.add(best_v)
        
        # Update sigmas for neighbors of both swapped vertices
        for (x, w) in adj[best_u]:
            sigmaS[x] -= w
            sigmaSprime[x] += w
        for (x, w) in adj[best_v]:
            sigmaS[x] += w
            sigmaSprime[x] -= w
            
    return S, Sprime
