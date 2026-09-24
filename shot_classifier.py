import cv2
import wave
import subprocess
import os
import mediapipe as mp
import webrtcvad
from speaker_detection import select_speaking_face

mp_face_detection = mp.solutions.face_detection

#thresholds 
DETECTION_CONFIDENCE = 0.1
VAD_AGGRESSIVENESS = 2
VAD_SAMPLE_RATE = 16000
VAD_FRAME_MS = 30
SAMPLE_INTERVAL_SEC = 0.5
FACE_PRESENT_WINDOW_SEC = 0.75
MIN_SHOT_DURATION_SEC = 1.0
MAX_SILENCE_GAP_SEC = 1.0
TRANSITION_BUFFER_SEC = 1.0


# audio: voice activity detection
def extract_audio_for_vad(video_path: str, output_wav: str):
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ar", str(VAD_SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le",
        output_wav
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run_vad(wav_path: str):
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

    with wave.open(wav_path, "rb") as wf:
        sample_rate = wf.getframerate()
        assert sample_rate == VAD_SAMPLE_RATE, f"Expected {VAD_SAMPLE_RATE}Hz, got {sample_rate}Hz"
        pcm_data = wf.readframes(wf.getnframes())

    frame_size = int(VAD_SAMPLE_RATE * VAD_FRAME_MS / 1000) * 2
    results = []
    offset = 0
    frame_idx = 0

    while offset + frame_size <= len(pcm_data):
        frame = pcm_data[offset:offset + frame_size]
        is_speech = vad.is_speech(frame, VAD_SAMPLE_RATE)
        time_sec = frame_idx * VAD_FRAME_MS / 1000
        results.append({"time_sec": round(time_sec, 3), "speech": is_speech})
        offset += frame_size
        frame_idx += 1

    return results


def get_speech_at_time(vad_results, time_sec: float, tolerance: float = 0.25) -> bool:
    closest = min(vad_results, key=lambda r: abs(r["time_sec"] - time_sec), default=None)
    if closest is None or abs(closest["time_sec"] - time_sec) > tolerance:
        return False
    return closest["speech"]


#video: face presence
def face_present(frame, detector) -> bool:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)
    return bool(results.detections)


#combined classification
def classify_video(video_path: str, sample_interval_sec: float = SAMPLE_INTERVAL_SEC):
    temp_wav = video_path + "_vad_temp.wav"
    extract_audio_for_vad(video_path, temp_wav)
    vad_results = run_vad(temp_wav)
    os.remove(temp_wav)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps else 0

    sample_interval_frames = max(1, int(fps * sample_interval_sec))
    results = []

    with mp_face_detection.FaceDetection(min_detection_confidence=DETECTION_CONFIDENCE) as detector:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval_frames == 0:
                time_sec = frame_idx / fps
                has_face = face_present(frame, detector)
                has_speech = get_speech_at_time(vad_results, time_sec)
                results.append({
                    "time_sec": round(time_sec, 2),
                    "speech": has_speech,
                    "face_present": has_face,
                })

            frame_idx += 1

    cap.release()

    raw_face_present = [r["face_present"] for r in results]
    raw_times = [r["time_sec"] for r in results]

    for i, r in enumerate(results):
        widened = any(
            abs(raw_times[j] - r["time_sec"]) <= FACE_PRESENT_WINDOW_SEC and raw_face_present[j]
            for j in range(len(results))
        )
        r["face_present"] = widened
        r["apply_lipsync"] = r["speech"] and widened

    return results, duration


def group_into_shots(frame_results, duration, min_shot_duration: float = MIN_SHOT_DURATION_SEC):
    if not frame_results:
        return []

    shots = []
    current_start = 0.0
    current_value = frame_results[0]["apply_lipsync"]

    for i in range(1, len(frame_results)):
        if frame_results[i]["apply_lipsync"] != current_value:
            shots.append({
                "start": round(current_start, 2),
                "end": round(frame_results[i]["time_sec"], 2),
                "apply_lipsync": current_value,
            })
            current_start = frame_results[i]["time_sec"]
            current_value = frame_results[i]["apply_lipsync"]

    shots.append({
        "start": round(current_start, 2),
        "end": round(duration, 2),
        "apply_lipsync": current_value,
    })

    merged = []
    for shot in shots:
        shot_duration = shot["end"] - shot["start"]
        if merged and shot_duration < min_shot_duration:
            merged[-1]["end"] = shot["end"]
        else:
            merged.append(shot)

    return merged


def fill_short_silence_gaps(shots, max_gap_duration: float = MAX_SILENCE_GAP_SEC):
    if len(shots) < 3:
        return shots

    filled = [dict(shots[0])]
    for i in range(1, len(shots) - 1):
        shot = dict(shots[i])
        duration = shot["end"] - shot["start"]
        prev_lipsync = filled[-1]["apply_lipsync"]
        next_lipsync = shots[i + 1]["apply_lipsync"]

        if not shot["apply_lipsync"] and duration <= max_gap_duration and prev_lipsync and next_lipsync:
            shot["apply_lipsync"] = True

        filled.append(shot)

    filled.append(dict(shots[-1]))

    merged = [filled[0]]
    for shot in filled[1:]:
        if shot["apply_lipsync"] == merged[-1]["apply_lipsync"]:
            merged[-1]["end"] = shot["end"]
        else:
            merged.append(shot)

    return merged


def resolve_speaker_for_shot(video_path: str, audio_path: str, shot: dict) -> dict:
    result = select_speaking_face(video_path, audio_path)

    if result is None:
        print(f"[Speaker Resolution] No confident speaker match for shot "
              f"{shot['start']}-{shot['end']}s -- downgrading to AUDIO ONLY.")
        shot["apply_lipsync"] = False
        shot["speaker_bbox"] = None
        shot["multi_person"] = False
    else:
        shot["speaker_bbox"] = result["bbox"]
        shot["speaker_confidence"] = result["confidence"]
        shot["multi_person"] = result.get("num_faces", 1) > 1

    return shot


def classify_shots(video_path: str, audio_path: str = None):
    print(f"[Stage 4] Analyzing {video_path}...")
    frame_results, duration = classify_video(video_path)
    shots = group_into_shots(frame_results, duration)
    shots = fill_short_silence_gaps(shots)

    if audio_path is not None:
        tmp_dir = os.path.join(os.path.dirname(video_path), "speaker_check_tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        for shot in shots:
            if shot["apply_lipsync"]:
                tmp_video = os.path.join(tmp_dir, f"shot_{shot['start']}_{shot['end']}_v.mp4")
                tmp_audio = os.path.join(tmp_dir, f"shot_{shot['start']}_{shot['end']}_a.wav")
                duration_sec = shot["end"] - shot["start"]

                full_duration = duration_sec
                if full_duration > TRANSITION_BUFFER_SEC * 3:
                    check_start = shot["start"] + TRANSITION_BUFFER_SEC
                    check_duration = full_duration - (TRANSITION_BUFFER_SEC * 2)
                else:
                    check_start = shot["start"]
                    check_duration = full_duration

                subprocess.run([
                    "ffmpeg", "-y", "-i", video_path,
                    "-ss", str(check_start), "-t", str(check_duration),
                    "-c:v", "libx264", "-c:a", "aac", tmp_video
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                # apply noise reduction to correlation audio only --
                # preserves original audio for final output
                subprocess.run([
                    "ffmpeg", "-y", "-i", audio_path,
                    "-ss", str(check_start), "-t", str(check_duration),
                    "-af", "afftdn=nf=-40",
                    "-ar", "44100", "-ac", "2", tmp_audio
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                resolve_speaker_for_shot(tmp_video, tmp_audio, shot)

    print(f"[Stage 4] Found {len(shots)} shot(s):")
    for shot in shots:
        label = "LIP-SYNC" if shot["apply_lipsync"] else "AUDIO ONLY"
        speaker_note = ""
        if shot.get("speaker_bbox"):
            speaker_note = f"  (speaker face: {shot['speaker_bbox']}, confidence {shot['speaker_confidence']:.2f})"
        print(f"    {shot['start']:>6.2f}s - {shot['end']:>6.2f}s  ->  {label}{speaker_note}")

    return shots

