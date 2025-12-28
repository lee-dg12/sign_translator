import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["GLOG_minloglevel"] = "2"

import cv2
import time
import numpy as np
import joblib
import mediapipe as mp

MODEL_PATH = "artifacts/hand_svm.joblib"

id_to_label = {
    1: "HELLO",
    13: "THANK YOU",
    15: "SORRY",
}

mp_hands = mp.solutions.hands

def normalize_vec(vec: np.ndarray) -> np.ndarray:
    # vec: (2,21,3)
    for h in range(2):
        pts = vec[h]
        if np.allclose(pts, 0):
            continue
        wrist = pts[0].copy()
        pts = pts - wrist
        scale = np.linalg.norm(pts[9]) + 1e-6
        pts = pts / scale
        vec[h] = pts
    return vec

def frames_to_feature(frames, stride=2):
    seq = []
    last_vec = np.zeros((2, 21, 3), dtype=np.float32)
    has_last = False
    detected = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ) as hands:
        for i, frame in enumerate(frames, start=1):
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

                vec = normalize_vec(vec)
                last_vec = vec
                has_last = True
                seq.append(vec.reshape(-1))  # (126,)
            else:
                if has_last:
                    seq.append(last_vec.reshape(-1))
                else:
                    seq.append(vec.reshape(-1))

    if detected == 0 or len(seq) < 5:
        return None

    seq = np.stack(seq, axis=0)  # (T,126)
    d = np.diff(seq, axis=0)
    if d.shape[0] == 0:
        d_mean = np.zeros((seq.shape[1],), dtype=np.float32)
    else:
        d_mean = d.mean(axis=0)

    feat = np.concatenate([seq.mean(axis=0), seq.std(axis=0), d_mean], axis=0)  # (378,)
    return feat.reshape(1, -1)

def record_clip(cap, seconds=2.0):
    frames = []
    start = time.time()
    while time.time() - start < seconds:
        ret, fr = cap.read()
        if not ret or fr is None:
            break
        fr = cv2.flip(fr, 1)
        frames.append(fr)
        cv2.imshow("KSL 3-signs", fr)
        cv2.waitKey(1)
    return frames

def main():
    model = joblib.load(MODEL_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    last_text = "SPACE: recognize / ESC: quit"

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        cv2.putText(frame, last_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow("KSL 3-signs", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        if key == 32:
            frames = record_clip(cap, seconds=2.0)
            feat = frames_to_feature(frames, stride=2)

            if feat is None:
                last_text = "No hands detected. Check lighting/distance."
                continue

            pred = int(model.predict(feat)[0])
            conf = float(np.max(model.predict_proba(feat)[0]))
            label = id_to_label.get(pred, "UNKNOWN")

            last_text = f"{label} (conf={conf:.2f})"
            print("PRED:", pred, label, "conf", conf)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
