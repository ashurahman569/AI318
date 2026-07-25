import glob
import re
import csv
import random

KNOWN_BEST = {
    "g1": 12078, "g2": 12084, "g3": 12077,
    "g11": 627,  "g12": 621,  "g13": 645,
    "g14": 3187, "g15": 3169, "g16": 3172,
    "g22": 14123,"g23": 14129,"g24": 14131,
    "g32": 1560, "g33": 1537, "g34": 1541,
    "g35": 8000, "g36": 7996, "g37": 8009,
    "g43": 7027, "g44": 7022, "g45": 7020,
    "g48": 6000, "g49": 6000, "g50": 5988,
}

def natural_key(filepath):
    name = filepath.split('/')[-1].split('\\')[-1]  # handle both path styles
    match = re.search(r'(\d+)', name)
    return int(match.group(1)) if match else 0


class Graph:
    def __init__(self, n):
        self.n = n
        self.adj = {v: [] for v in range(1, n + 1)} 

    def addEdge(self, u, v, W):
        if 1 <= u <= self.n and 1 <= v <= self.n:
            self.adj[u].append((v, W))
            self.adj[v].append((u, W))  


def RandomizedMaxCut(G, times):
    totalCutWeight = 0
    n = G.n
    for i in range(1, times+1):
        X = set()
        Y = set()
        for j in range(1, n+1):       
            if random.choice([True, False]):
                X.add(j)
            else:
                Y.add(j)
        totalCutWeight += cutWeight(G, X, Y)

    # return average cut weight over trials
    return totalCutWeight / times

def GreedyMaxCut(G):
    n = G.n
    adj = G.adj

    maxu, maxv, maxw = -1, -1, -1
    for u in range(1, n + 1):
        for (v, w) in adj[u]:
            if w > maxw:
                maxw = w
                maxu, maxv = u, v

    X = {maxu}
    Y = {maxv}

    U = [z for z in range(1, n + 1) if z not in X and z not in Y]

    for z in U:
        wX = sum(w for (y, w) in adj[z] if y in Y)  # if z belongsto X
        wY = sum(w for (x, w) in adj[z] if x in X)  # if z belongsto Y

        if wX > wY:
            X.add(z)
        else:
            Y.add(z)

    return X, Y

def SemiGreedyMaxCut(G, alpha=0.5):
    n = G.n
    adj = G.adj

    maxu, maxv, maxw = -1, -1, -1
    for u in range(1, n + 1):
        for (v, w) in adj[u]:
            if w > maxw:
                maxw = w
                maxu, maxv = u, v

    X = {maxu}
    Y = {maxv}

    Vprime = set(range(1, n + 1)) - X - Y

    while Vprime:
        sigmaX = {}
        sigmaY = {}

        for v in Vprime:
            sigmaX[v] = sum(w for (x, w) in adj[v] if x in X)
            sigmaY[v] = sum(w for (y, w) in adj[v] if y in Y)

        greedy_val = {v: max(sigmaX[v], sigmaY[v]) for v in Vprime}

        sigma = list(sigmaX.values()) + list(sigmaY.values())
        wmin = min(sigma)
        wmax = max(sigma)
        u = wmin + alpha * (wmax - wmin)

        RCL = [z for z in Vprime if greedy_val[z] >= u]
        chosen = random.choice(RCL)

        if sigmaX[chosen] >= sigmaY[chosen]:
            X.add(chosen)
        else:
            Y.add(chosen)

        Vprime.remove(chosen)

    return X, Y

def GRASP(maxIter, G, alpha = 0.5):
    xStar = None
    wStar = None

    for i in range (1, maxIter+1):
        X, Y = SemiGreedyMaxCut(G, alpha)
        X, Y = LocalSearch(X, Y, G)
        w = cutWeight (G, X, Y)

        if i == 1:
            xStar = (X, Y)
            wStar = w
        elif w > wStar:
            xStar = (X, Y)
            wStar = w

    return xStar, wStar

def LocalSearch(S, Sprime, G):
    n = G.n
    adj = G.adj
    while(1):
        sigmaS = {}
        sigmaSprime = {}
        delta = {}
        for v in range(1, n+1):
            sigmaS[v] = sum(w for (x, w) in adj[v] if x in S)
            sigmaSprime[v] = sum(w for (y, w) in adj[v] if y in Sprime)
        maxdel = 0
        for v in range(1, n+1):
            if v in S:
                delta[v]= sigmaS[v] - sigmaSprime[v]
                newS, newSprime = S - {v} , Sprime | {v}
            else:
                delta[v]= sigmaSprime[v] - sigmaS[v]
                newS, newSprime = S | {v} , Sprime - {v}
            if delta[v] > maxdel:
                maxdel = delta[v]
                maxS = newS
                maxSprime = newSprime

        if maxdel == 0:
            return S, Sprime  
        else:
            S, Sprime = maxS, maxSprime

def cutWeight(G, X, Y):
    total = 0
    for u in X:
        for (v, w) in G.adj[u]:
            if v in Y:
                total += w
    return total

def loadGraph(filepath):
    with open(filepath, 'r') as f:
        n, m = map(int, f.readline().split())
        g = Graph(n)
        for _ in range(m):
            u, v, w = map(int, f.readline().split())
            g.addEdge(u, v, w)
    return g, n, m

def main():
    folder = "set1" 
    filepaths = sorted(glob.glob(f"{folder}/*.rud"), key=natural_key)

    if not filepaths:
        return

    results = []

    for filepath in filepaths:
        name = filepath.split('/')[-1].split('\\')[-1].replace('.rud', '')
        print(f"Processing {name}...")

        G, n, m = loadGraph(filepath)

        randomized_val = RandomizedMaxCut(G, 20) # avg 20 random solutions

        X, Y = GreedyMaxCut(G)
        greedy_val = cutWeight(G, X, Y)

        X, Y = SemiGreedyMaxCut(G, 0.5)
        semi_greedy_val = cutWeight(G, X, Y)

        local_search_vals = []
        for i in range(15): #15 local search iterations
            Xc, Yc = SemiGreedyMaxCut(G, 0.5)
            Xl, Yl = LocalSearch(Xc, Yc, G)
            local_search_vals.append(cutWeight(G, Xl, Yl))
        local_search_avg = sum(local_search_vals) / len(local_search_vals)


        (Xg, Yg), grasp_val = GRASP(10, G, 0.5)

        known_best = KNOWN_BEST.get(name, "")

        results.append({
            "Name": name,
            "|V|": n,
            "|E|": m,
            "Simple Randomized": round(randomized_val, 2),
            "Simple Greedy": greedy_val,
            "Semi-Greedy-1": semi_greedy_val,
            "Local Search Iterations": 15,
            "Local Search Avg Value": round(local_search_avg, 2),
            "GRASP Iterations": 10,
            "GRASP Best Value": grasp_val,
            "Known Best/Upper Bound": known_best,
        })

    out_file = "2205029.csv"
    with open(out_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone. Results written to {out_file}")

if __name__ == '__main__':
    main()
