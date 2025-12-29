import os, glob
import numpy as np
import cv2
import mediapipe as mp

DATA_ROOT = "data"
CLASSES = [("1", 1), ("13", 13), ("15", 15)]
EXTS = ("*.mp4", "*.MP4")

mp_hands = mp.solutions.hands

def video_to_feature(path: str, max_frames=120, stride=2):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"open failed: {path}")

    seq = []
    detected = 0

    last_vec = np.zeros((2, 21, 3), dtype=np.float32)
    has_last = False

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ) as hands:

        i = 0
        while len(seq) < max_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            i += 1
            if i % stride != 0:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            vec = np.zeros((2, 21, 3), dtype=np.float32)

            if res.multi_hand_landmarks:
                detected += 1
                for h_idx, hand_lms in enumerate(res.multi_hand_landmarks[:2]):
                    for p_idx, lm in enumerate(hand_lms.landmark):
                        vec[h_idx, p_idx, 0] = lm.x
                        vec[h_idx, p_idx, 1] = lm.y
                        vec[h_idx, p_idx, 2] = lm.z

                for h in range(2):
                    pts = vec[h]
                    if np.allclose(pts, 0):
                        continue
                    wrist = pts[0].copy()
                    pts = pts - wrist
                    scale = np.linalg.norm(pts[9]) + 1e-6
                    pts = pts / scale
                    vec[h] = pts

                last_vec = vec
                has_last = True
                seq.append(vec.reshape(-1))

            else:
                if has_last:
                    seq.append(last_vec.reshape(-1))
                else:
                    seq.append(vec.reshape(-1))

    cap.release()
    
    if detected == 0:
        raise RuntimeError(f"no hands detected: {path}")

    seq = np.stack(seq, axis=0)  # (T,126)

    d = np.diff(seq, axis=0)
    feat = np.concatenate([seq.mean(axis=0), seq.std(axis=0), d.mean(axis=0)], axis=0)
    return feat

X, y = [], []
skipped = 0

for folder, label in CLASSES:
    paths = []
    for ext in EXTS:
        paths += glob.glob(os.path.join(DATA_ROOT, folder, ext))

    print(f"[{folder}] files:", len(paths))

    for p in paths:
        try:
            X.append(video_to_feature(p))
            y.append(label)
        except Exception as e:
            skipped += 1
            print("SKIP:", p, "->", e)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

print("X:", X.shape, "y:", y.shape, "labels:", dict(zip(*np.unique(y, return_counts=True))))
print("skipped:", skipped)

np.save("X.npy", X)
np.save("y.npy", y)
print("saved: X.npy, y.npy")
