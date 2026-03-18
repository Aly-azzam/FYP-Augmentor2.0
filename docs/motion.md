# 📄 AugMentor 2.0 — Motion Representation & Evaluation Pipeline

---

## 🧭 1. Overview

This document describes the implementation of:

* **Motion Representation Engine (Person 2)**
* **Alignment & Scoring (partial Person 3)**

The goal is to transform raw landmark data into:

* structured motion features
* normalized representation
* alignment-ready vectors
* similarity score between learner and expert

---

## 🧱 2. Global Pipeline

```
Video (future)
→ Landmarks (Perception Engine - Person 1)
→ Motion Representation (Person 2)
→ Alignment + Scoring (Person 3)
→ Feedback (future)
```

---

## 🧠 3. Responsibilities

### 👤 Person 2 (Ali)

Responsible for:

* sequence construction
* motion features (velocity, acceleration)
* normalization
* alignment preparation

### 🔄 Extended (implemented early)

* DTW alignment
* scoring system

---

## 📥 4. Input Format

```json
{
  "video_id": "learner_01",
  "fps": 30,
  "frames": [
    {
      "frame_index": 0,
      "timestamp": 0.0,
      "landmarks": {
        "wrist": [x, y, z],
        "index_tip": [x, y, z],
        "thumb_tip": [x, y, z]
      }
    }
  ]
}
```

---

## 📤 5. Outputs

### Motion Representation

```json
{
  "frames": [
    {
      "positions": {...},
      "velocity": {...},
      "speed": {...},
      "acceleration": {...},
      "trajectory_point": {...}
    }
  ]
}
```

### Alignment Vectors

```
[x1, y1, z1, x2, y2, z2, ...]
```

---

## ⚙️ 6. Implementation Steps

### Step 1 — Sequence

* Load JSON
* Build ordered frames
* Extract landmarks

---

### Step 2 — Trajectory

* Extract (x, y) paths
* Compute trajectory length

---

### Step 3 — Motion Features

Velocity:

```
v = (p_t - p_{t-1}) / dt
```

Speed:

```
|v|
```

Acceleration:

```
a = (v_t - v_{t-1}) / dt
```

---

### Step 4 — Enrichment

Each frame includes:

* positions
* velocity
* speed
* acceleration
* trajectory

---

### Step 5 — Normalization

Centering:

```
p' = p - wrist
```

Scaling:

```
p'' = p' / distance(wrist, index_tip)
```

Result:

```
wrist → [0, 0, 0]
```

---

### Step 6 — Alignment Prep

Flatten each frame:

```
[wrist_x, wrist_y, wrist_z,
 index_x, index_y, index_z,
 thumb_x, thumb_y, thumb_z]
```

---

## 🔁 7. DTW (Dynamic Time Warping)

Used to align sequences with different speeds.

```
cost[i][j] = dist + min(
    cost[i-1][j],
    cost[i][j-1],
    cost[i-1][j-1]
)
```

Output:

* distance
* normalized distance
* alignment path

---

## 📊 8. Scoring

```
score = max(0, 100 * (1 - normalized_distance))
```

---

## 🧪 9. Demo Pipeline

Runs full system:

```
Learner vs Expert → Score
```

---

## 📊 10. Demo Results

* GOOD → ~94
* MEDIUM → ~71
* POOR → ~7

---

## 🎓 11. Key Explanation

> We extract motion features, normalize them, align sequences using DTW, and compute a similarity score.

---

## 🏁 Conclusion

* Modular
* Scalable
* AI-ready
* Demo-ready

---
