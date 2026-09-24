import cv2
import numpy as np
import librosa
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

# thresholds 
SAMPLE_INTERVAL_SEC = 0.1          
MIN_CORRELATION_CONFIDENCE = 0.3   
SINGLE_FACE_MIN_CONFIDENCE = 0.1   
MIN_SAMPLES_REQUIRED = 10         
SMOOTHING_WINDOW = 9              
DETECTION_CONFIDENCE = 0.3         
FACE_CROP_MARGIN = 0.3             

# Mediapipe FaceMesh landmark indices for the mouth (upper and lower
# inner lip center points)
UPPER_LIP_IDX = 13
LOWER_LIP_IDX = 14


def get_mouth_openness_per_face(frame, detection_model, mesh_model, max_faces: int = 5):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detection_results = detection_model.process(rgb)

    if not detection_results.detections:
        return []

    faces = []
    for detection in detection_results.detections[:max_faces]:
        rel_bbox = detection.location_data.relative_bounding_box

        x1 = max(0, int(rel_bbox.xmin * w))
        y1 = max(0, int(rel_bbox.ymin * h))
        x2 = min(w, int((rel_bbox.xmin + rel_bbox.width) * w))
        y2 = min(h, int((rel_bbox.ymin + rel_bbox.height) * h))

        if x2 <= x1 or y2 <= y1:
            continue

        margin_x = int((x2 - x1) * FACE_CROP_MARGIN)
        margin_y = int((y2 - y1) * FACE_CROP_MARGIN)
        cx1 = max(0, x1 - margin_x)
        cy1 = max(0, y1 - margin_y)
        cx2 = min(w, x2 + margin_x)
        cy2 = min(h, y2 + margin_y)

        face_crop = rgb[cy1:cy2, cx1:cx2]
        if face_crop.size == 0:
            continue

        mesh_results = mesh_model.process(face_crop)
        if not mesh_results.multi_face_landmarks:
            continue

        landmarks = mesh_results.multi_face_landmarks[0].landmark
        crop_h, crop_w = face_crop.shape[:2]

        ys = [lm.y * crop_h for lm in landmarks]
        face_height = max(ys) - min(ys)
        if face_height <= 0:
            continue

        upper_lip = landmarks[UPPER_LIP_IDX]
        lower_lip = landmarks[LOWER_LIP_IDX]
        lip_distance_px = abs((lower_lip.y - upper_lip.y) * crop_h)
        mouth_openness = lip_distance_px / face_height

        faces.append({
            "bbox": (x1, y1, x2, y2),
            "mouth_openness": mouth_openness,
        })

    return faces


def track_faces_over_segment(video_path: str, sample_interval_sec: float = SAMPLE_INTERVAL_SEC):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    sample_interval_frames = max(1, int(fps * sample_interval_sec))

    timestamps = []
    tracks = []

    with mp_face_detection.FaceDetection(min_detection_confidence=DETECTION_CONFIDENCE) as detection_model, \
         mp_face_mesh.FaceMesh(
             static_image_mode=True,
             max_num_faces=1,
             refine_landmarks=True,
             min_detection_confidence=0.2,
         ) as mesh_model:

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval_frames == 0:
                t = frame_idx / fps
                timestamps.append(t)
                faces = get_mouth_openness_per_face(frame, detection_model, mesh_model)

                matched_track_ids = set()

                for face in faces:
                    cx = (face["bbox"][0] + face["bbox"][2]) / 2
                    cy = (face["bbox"][1] + face["bbox"][3]) / 2

                    matched_idx = None
                    best_dist = float("inf")
                    for idx, track in enumerate(tracks):
                        if idx in matched_track_ids:
                            continue
                        last_bbox = track["last_bbox"]
                        last_cx = (last_bbox[0] + last_bbox[2]) / 2
                        last_cy = (last_bbox[1] + last_bbox[3]) / 2
                        dist = ((cx - last_cx) ** 2 + (cy - last_cy) ** 2) ** 0.5
                        if dist < best_dist and dist < 150:
                            best_dist = dist
                            matched_idx = idx

                    if matched_idx is None:
                        tracks.append({
                            "last_bbox": face["bbox"],
                            "mouth_openness_series": [None] * (len(timestamps) - 1) + [face["mouth_openness"]],
                        })
                        matched_track_ids.add(len(tracks) - 1)
                    else:
                        tracks[matched_idx]["last_bbox"] = face["bbox"]
                        tracks[matched_idx]["mouth_openness_series"].append(face["mouth_openness"])
                        matched_track_ids.add(matched_idx)

                for idx, track in enumerate(tracks):
                    if idx not in matched_track_ids:
                        track["mouth_openness_series"].append(None)

            frame_idx += 1

    cap.release()
    return timestamps, tracks


def get_audio_energy_envelope(audio_path: str, timestamps: list):
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    energy_at_timestamps = np.interp(timestamps, rms_times, rms)
    return list(energy_at_timestamps)


def smooth_signal_array(values: list, window: int = SMOOTHING_WINDOW) -> np.ndarray:
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def correlate_face_with_audio(mouth_series: list, audio_energy: list) -> float:
    valid_indices = [i for i, m in enumerate(mouth_series) if m is not None]

    if len(valid_indices) < MIN_SAMPLES_REQUIRED:
        return 0.0

    mouth_vals = [mouth_series[i] for i in valid_indices]
    audio_vals = [audio_energy[i] for i in valid_indices]

    # mouth MOVEMENT: absolute frame-to-frame change, not raw position
    mouth_movement = [abs(mouth_vals[i] - mouth_vals[i - 1]) for i in range(1, len(mouth_vals))]
    audio_vals_aligned = audio_vals[1:]  # align lengths after the diff

    mouth_smoothed = smooth_signal_array(mouth_movement)
    audio_smoothed = smooth_signal_array(audio_vals_aligned)

    if np.std(mouth_smoothed) == 0 or np.std(audio_smoothed) == 0:
        return 0.0

    correlation = np.corrcoef(mouth_smoothed, audio_smoothed)[0, 1]
    return correlation if not np.isnan(correlation) else 0.0


def select_speaking_face(video_path: str, audio_path: str):
    timestamps, tracks = track_faces_over_segment(video_path)

    if not tracks:
        print("[SpeakerDetection] No faces detected in segment.")
        return None

    cap = cv2.VideoCapture(video_path)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    audio_energy = get_audio_energy_envelope(audio_path, timestamps)

    best_track = None
    best_score = 0.0

    for track in tracks:
        score = correlate_face_with_audio(track["mouth_openness_series"], audio_energy)
        x1, y1, x2, y2 = track["last_bbox"]
        width_ratio = (x2 - x1) / frame_w
        print(f"[SpeakerDetection] Face at {track['last_bbox']}: "
              f"correlation = {score:.3f}, width_ratio = {width_ratio:.3f}")

        if score > best_score:
            best_score = score
            best_track = track

    threshold = SINGLE_FACE_MIN_CONFIDENCE if len(tracks) == 1 else MIN_CORRELATION_CONFIDENCE

    if best_track is None or best_score < threshold:
        print(f"[SpeakerDetection] No face met confidence threshold ({threshold}). "
              f"Best score was {best_score:.3f}.")
        return None

    print(f"[SpeakerDetection] Selected face at {best_track['last_bbox']} with confidence {best_score:.3f}")
    return {
        "bbox": best_track["last_bbox"],
        "confidence": best_score,
        "num_faces": len(tracks)
    }

