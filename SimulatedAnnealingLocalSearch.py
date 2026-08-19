Standard Local Search strictly rejects worsening moves (Δ≤0). Simulated Annealing allows the algorithm to escape deep local optima by occasionally accepting worse moves. 
The probability of accepting a worse move depends on a "Temperature" parameter T, which gradually cools down.
Task: Implement Simulated Annealing. If a move improves the cut (Δ>0), accept it. If it worsens the cut (Δ≤0), 
accept it with probability eΔ/T. Start with T0=100.0 and cool it by a factor of α=0.99 at each step until T<1.0.

import math

def LocalSearchSA(S, Sprime, G, T0=100.0, Tmin=1.0, alpha_cool=0.99):
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

    T = T0
    # Random permutation for scanning order
    perm = list(range(1, n + 1))
    random.shuffle(perm)
    idx = 0
    
    while T > Tmin:
        v = perm[idx]
        
        if v in S:
            delta = sigmaS[v] - sigmaSprime[v]
        else:
            delta = sigmaSprime[v] - sigmaS[v]
            
        accept = False
        if delta > 0:
            accept = True
        else:
            # Probability of accepting a worse move
            prob = math.exp(delta / T)
            if random.random() < prob:
                accept = True
                
        if accept:
            if v in S:
                S.remove(v)
                Sprime.add(v)
                moved_to_Sprime = True
            else:
                Sprime.remove(v)
                S.add(v)
                moved_to_Sprime = False
                
            for (x, w) in adj[v]:
                if moved_to_Sprime:
                    sigmaS[x] -= w
                    sigmaSprime[x] += w
                else:
                    sigmaS[x] += w
                    sigmaSprime[x] -= w
                    
        # Cool down temperature
        T *= alpha_cool
        idx = (idx + 1) % n
        
    return S, Sprime
