import json
import math

with open("expert_dtw_trajectory.json", "r") as f:
    expert = json.load(f)

with open("learner_dtw_trajectory.json", "r") as f:
    learner = json.load(f)

n = len(expert)
m = len(learner)

def dist(a, b):
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2)

dp = [[float("inf")] * (m + 1) for _ in range(n + 1)]
parent = [[None] * (m + 1) for _ in range(n + 1)]

dp[0][0] = 0

for i in range(1, n + 1):
    for j in range(1, m + 1):
        cost = dist(expert[i - 1], learner[j - 1])

        options = [
            (dp[i - 1][j], (i - 1, j)),
            (dp[i][j - 1], (i, j - 1)),
            (dp[i - 1][j - 1], (i - 1, j - 1)),
        ]

        best_cost, best_parent = min(options, key=lambda x: x[0])
        dp[i][j] = cost + best_cost
        parent[i][j] = best_parent

# backtrack
path = []
i, j = n, m

while i > 0 and j > 0:
    path.append((i - 1, j - 1))
    i, j = parent[i][j]

path.reverse()

# learner_frame -> expert_frame
mapping = {}

for expert_idx, learner_idx in path:
    learner_frame = learner[learner_idx]["frame"]
    expert_frame = expert[expert_idx]["frame"]
    mapping[learner_frame] = expert_frame

# fill missing learner frames using last expert frame
filled_mapping = {}
last_expert = 0

max_learner_frame = max(p["frame"] for p in learner)

for lf in range(max_learner_frame + 1):
    if lf in mapping:
        last_expert = mapping[lf]
    filled_mapping[lf] = last_expert

with open("dtw_alignment.json", "w") as f:
    json.dump(filled_mapping, f, indent=2)

print("Saved -> dtw_alignment.json")
print(f"DTW path length: {len(path)}")
print(f"Expert points: {n}, Learner points: {m}")