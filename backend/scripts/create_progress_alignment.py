import json
import math

with open("expert_dtw_trajectory.json", "r") as f:
    expert = json.load(f)

expert = [p for p in expert if p["frame"] < 446]

with open("learner_dtw_trajectory.json", "r") as f:
    learner = json.load(f)

def cumulative_progress(points):
    distances = [0.0]
    total = 0.0

    for i in range(1, len(points)):
        dx = points[i]["x"] - points[i - 1]["x"]
        dy = points[i]["y"] - points[i - 1]["y"]
        d = math.sqrt(dx * dx + dy * dy)
        total += d
        distances.append(total)

    if total == 0:
        return [0.0 for _ in distances]

    return [d / total for d in distances]

expert_prog = cumulative_progress(expert)
learner_prog = cumulative_progress(learner)

mapping = {}

expert_i = 0

for li, lp in enumerate(learner_prog):
    while expert_i < len(expert_prog) - 1 and expert_prog[expert_i] < lp:
        expert_i += 1

    learner_frame = learner[li]["frame"]
    expert_frame = expert[expert_i]["frame"]

    mapping[learner_frame] = expert_frame

# fill missing learner frames
filled = {}
last_expert = 0
max_learner_frame = max(p["frame"] for p in learner)

for lf in range(max_learner_frame + 1):
    if lf in mapping:
        last_expert = mapping[lf]
    filled[lf] = last_expert

with open("progress_alignment.json", "w") as f:
    json.dump(filled, f, indent=2)

print("Saved -> progress_alignment.json")
print(f"Expert frames: {len(expert)}")
print(f"Learner frames: {len(learner)}")