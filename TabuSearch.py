Standard Local Search can suffer from "cycling" or oscillation, where it flips a vertex v to Y, and then in the very next step flips it back to X
because it's still the best immediate move. Tabu Search solves this by introducing short-term memory. When a vertex is flipped, it is added to a 
"Tabu List" and is forbidden from being flipped back for the next T iterations (the "tenure").
Task: Implement a Tabu Search Local Search with a tenure of T=7 and compare it to the standard Best-Improvement Local Search.

def LocalSearchTabu(S, Sprime, G, tabu_tenure=7):
    n = G.n
    adj = G.adj
    S = set(S)
    Sprime = set(Sprime)
    
    sigmaS = {v: 0 for v in range(1, n + 1)}
    sigmaSprime = {v: 0 for v in range(1, n + 1)}
    for v in range(1, n + 1):
        for (x, w) in adj[v]:
            if x in S: sigmaS[v] += w
            else: sigmaSprime[v] += w
            
    # Initialize tabu list (0 means not tabu)
    tabu_list = {v: 0 for v in range(1, n + 1)}
    consecutive_no_improve = 0
    
    while consecutive_no_improve < n:
        maxdel = 0
        best_v = None
        
        for v in range(1, n + 1):
            # Skip if vertex is currently tabu
            if tabu_list[v] > 0:
                continue
                
            if v in S:
                delta = sigmaS[v] - sigmaSprime[v]
            else:
                delta = sigmaSprime[v] - sigmaS[v]
                
            if delta > maxdel:
                maxdel = delta
                best_v = v
                
        if best_v is None:
            # No improving non-tabu move found
            consecutive_no_improve += 1
            # Decrement tabu tenures for all vertices
            for v in range(1, n + 1):
                if tabu_list[v] > 0:
                    tabu_list[v] -= 1
            continue
            
        # Flip the best vertex
        if best_v in S:
            S.remove(best_v)
            Sprime.add(best_v)
            moved_to_Sprime = True
        else:
            Sprime.remove(best_v)
            S.add(best_v)
            moved_to_Sprime = False
            
        # Update sigmas for neighbors
        for (x, w) in adj[best_v]:
            if moved_to_Sprime:
                sigmaS[x] -= w
                sigmaSprime[x] += w
            else:
                sigmaS[x] += w
                sigmaSprime[x] -= w
                
        # Add to tabu list and reset counter
        tabu_list[best_v] = tabu_tenure
        consecutive_no_improve = 0
        
        # Decrement tabu tenures for all other vertices
        for v in range(1, n + 1):
            if v != best_v and tabu_list[v] > 0:
                tabu_list[v] -= 1
                
    return S, Sprime
