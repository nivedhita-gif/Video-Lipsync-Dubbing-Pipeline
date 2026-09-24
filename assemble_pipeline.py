import os
import sys
import subprocess
import json
import librosa
import numpy as np
from shot_classifier import classify_shots

WORK_DIR = r"C:\lipsync\test_clips\segments"
os.makedirs(WORK_DIR, exist_ok=True)

MAX_MUSETALK_SEGMENT_SEC = 25.0


def cut_segment(input_path: str, start: float, duration: float, output_path: str, is_audio: bool = False):
    """Cuts a segment from input_path starting at start for duration seconds."""
    if is_audio:
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ss", str(start), "-t", str(duration),
            "-c", "copy", output_path,
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-ss", str(start), "-t", str(duration),
            "-c:v", "libx264", "-c:a", "aac", output_path,
        ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def find_best_split_point(audio_path: str, total_duration: float) -> float:
    """Finds the quietest moment within 3 seconds of the midpoint."""
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

    target = total_duration / 2
    window = 3.0
    mask = (times >= target - window) & (times <= target + window)

    if not np.any(mask):
        return target

    window_rms = rms[mask]
    window_times = times[mask]
    best_idx = np.argmin(window_rms)

    print(f"[AutoSplit] Best split point near {target:.1f}s: {window_times[best_idx]:.2f}s "
          f"(RMS energy: {window_rms[best_idx]:.4f})")
    return float(window_times[best_idx])


def split_segment_recursively(input_vid: str, input_aud: str, seg_idx: int,
                               part_label: str, duration: float) -> list:
    """Recursively splits a segment into chunks no longer than MAX_MUSETALK_SEGMENT_SEC."""
    if duration <= MAX_MUSETALK_SEGMENT_SEC:
        output = os.path.join(WORK_DIR, f"seg_{seg_idx:03d}_musetalk_out{part_label}.mp4")
        return [{"video": input_vid, "audio": input_aud, "output": output}]

    split_t = find_best_split_point(input_aud, duration)

    vid_a = os.path.join(WORK_DIR, f"seg_{seg_idx:03d}_lipsync_video{part_label}_a.mp4")
    vid_b = os.path.join(WORK_DIR, f"seg_{seg_idx:03d}_lipsync_video{part_label}_b.mp4")
    aud_a = os.path.join(WORK_DIR, f"seg_{seg_idx:03d}_audio{part_label}_a.wav")
    aud_b = os.path.join(WORK_DIR, f"seg_{seg_idx:03d}_audio{part_label}_b.wav")

    cut_segment(input_vid, 0, split_t, vid_a)
    cut_segment(input_vid, split_t, duration - split_t, vid_b)
    cut_segment(input_aud, 0, split_t, aud_a, is_audio=True)
    cut_segment(input_aud, split_t, duration - split_t, aud_b, is_audio=True)

    parts_a = split_segment_recursively(vid_a, aud_a, seg_idx, f"{part_label}_a", split_t)
    parts_b = split_segment_recursively(vid_b, aud_b, seg_idx, f"{part_label}_b", duration - split_t)

    return parts_a + parts_b


def crop_video_to_speaker(video_path: str, speaker_bbox: tuple, output_path: str, padding_ratio: float = 1.2):
    """
    Crops video to the speaker's face region for MuseTalk input.
    Only used for multi-person shots.
    Returns: (crop_x1, crop_y1, crop_w, crop_h)
    """
    probe_cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0", video_path
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
    frame_w, frame_h = map(int, result.stdout.strip().split("x"))

    x1, y1, x2, y2 = speaker_bbox
    box_w = x2 - x1
    box_h = y2 - y1

    pad_x = int(box_w * padding_ratio)
    pad_top = int(box_h * padding_ratio * 0.5)
    pad_bottom = int(box_h * padding_ratio * 1.5)

    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_top)
    crop_y2 = min(frame_h, y2 + pad_bottom)
    crop_w = min(box_w + 2 * pad_x, frame_w - crop_x1)
    crop_h = crop_y2 - crop_y1

    crop_w -= crop_w % 2
    crop_h -= crop_h % 2

    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"crop={crop_w}:{crop_h}:{crop_x1}:{crop_y1}",
        "-c:v", "libx264", "-c:a", "aac", output_path,
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return (crop_x1, crop_y1, crop_w, crop_h)


def composite_crop_back(original_video_path: str, musetalk_crop_output_path: str,
                         crop_region: tuple, output_path: str):
    """
    Composites MuseTalk's cropped output back onto the original full-frame video.
    """
    x1, y1, crop_w, crop_h = crop_region

    filter_complex = (
        f"[1:v]scale={crop_w}:{crop_h}[scaled];"
        f"[0:v][scaled]overlay={x1}:{y1}[out]"
    )

    subprocess.run([
        "ffmpeg", "-y",
        "-i", original_video_path,
        "-i", musetalk_crop_output_path,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:v", "libx264",
        "-an", output_path,
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def prepare_segments(video_path: str, hindi_audio_path: str, mode: str = "default",
                     classification_audio_path: str = None):
    """
    Runs shot classification using original audio for accurate speaker detection,
    but cuts LIP-SYNC segments using the Hindi dubbed audio for lip-sync.
    """
    # use original audio for classification if provided, otherwise use hindi audio
    audio_for_classification = classification_audio_path if classification_audio_path else hindi_audio_path
    shots = classify_shots(video_path, audio_for_classification)

    for i, shot in enumerate(shots):
        start, end = shot["start"], shot["end"]
        tag = "lipsync" if shot["apply_lipsync"] else "audioonly"

        video_seg_path = os.path.join(WORK_DIR, f"seg_{i:03d}_{tag}_video.mp4")
        cut_segment(video_path, start, end - start, video_seg_path)
        shot["video_segment_path"] = video_seg_path

        if not shot["apply_lipsync"]:
            continue

        # cut audio from Hindi dubbed track for lip-sync
        audio_seg_path = os.path.join(WORK_DIR, f"seg_{i:03d}_audio.wav")
        cut_segment(hindi_audio_path, start, end - start, audio_seg_path, is_audio=True)
        shot["audio_segment_path"] = audio_seg_path

        if shot.get("speaker_bbox") and shot.get("multi_person", False):
            cropped_path = os.path.join(WORK_DIR, f"seg_{i:03d}_lipsync_video_cropped.mp4")
            crop_region = crop_video_to_speaker(video_seg_path, shot["speaker_bbox"], cropped_path)
            shot["musetalk_input_video_path"] = cropped_path
            shot["crop_region"] = list(crop_region)
        else:
            shot["musetalk_input_video_path"] = video_seg_path
            shot["crop_region"] = None

        shot["musetalk_output_path"] = os.path.join(WORK_DIR, f"seg_{i:03d}_musetalk_out.mp4")

        seg_duration = end - start
        if seg_duration > MAX_MUSETALK_SEGMENT_SEC:
            shot["musetalk_split_parts"] = split_segment_recursively(
                shot["musetalk_input_video_path"], audio_seg_path, i, "", seg_duration
            )
        else:
            shot["musetalk_split_parts"] = None

    manifest_path = os.path.join(WORK_DIR, f"manifest_{mode}.json")
    with open(manifest_path, "w") as f:
        json.dump(shots, f, indent=2)

    print(f"\n[Stage 4+6 prep] {len(shots)} segments prepared.")
    print(f"[Stage 4+6 prep] Manifest saved to: {manifest_path}")
    print(f"\n[Stage 4+6 prep] LIP-SYNC segments needing MuseTalk (upload these to Colab):")

    colab_pairs = []
    for shot in shots:
        if not shot["apply_lipsync"]:
            continue
        if shot.get("musetalk_split_parts"):
            for part in shot["musetalk_split_parts"]:
                print(f"    {part['video']}")
                print(f"    {part['audio']}")
                colab_pairs.append((
                    os.path.basename(part['video']),
                    os.path.basename(part['audio']),
                    os.path.basename(part['output'])
                ))
        else:
            print(f"    {shot['musetalk_input_video_path']}")
            print(f"    {shot['audio_segment_path']}")
            colab_pairs.append((
                os.path.basename(shot['musetalk_input_video_path']),
                os.path.basename(shot['audio_segment_path']),
                os.path.basename(shot['musetalk_output_path'])
            ))

    print(f"\n[Stage 4+6 prep] Automated Colab batch inference cell:")
    print("=" * 60)
    print("%%bash")
    print("export PATH=\"/usr/local/miniconda3/bin:$PATH\"")
    print("source /usr/local/miniconda3/etc/profile.d/conda.sh")
    print("conda activate musetalk")
    print("cd /content/MuseTalk")
    print("export MPLBACKEND=Agg")
    print()
    for video, audio, output in colab_pairs:
        print(f"cat > configs/inference/test.yaml << 'EOF'")
        print(f"task:")
        print(f"  video_path: \"/content/MuseTalk/{video}\"")
        print(f"  audio_path: \"/content/MuseTalk/{audio}\"")
        print(f"EOF")
        print(f"python -m scripts.inference \\")
        print(f"  --inference_config configs/inference/test.yaml \\")
        print(f"  --result_dir results \\")
        print(f"  --unet_model_path models/musetalkV15/unet.pth \\")
        print(f"  --unet_config models/musetalk/musetalk.json \\")
        print(f"  --version v15 < /dev/null")
        print(f"echo 'Done: {video}'")
        print()
    print("echo 'ALL SEGMENTS COMPLETE'")
    print("=" * 60)

    return shots


def stitch_final(video_path: str, hindi_audio_path: str, output_path: str, mode: str = "default"):
    """
    Reads the manifest, composites MuseTalk outputs back onto full frames,
    concatenates all segments, and overlays the final audio track.
    """
    manifest_path = os.path.join(WORK_DIR, f"manifest_{mode}.json")
    with open(manifest_path, "r") as f:
        shots = json.load(f)

    clip_list_path = os.path.join(WORK_DIR, "concat_list.txt")

    with open(clip_list_path, "w") as f:
        for i, shot in enumerate(shots):
            if shot["apply_lipsync"]:
                if shot.get("musetalk_split_parts"):
                    concat_path = os.path.join(WORK_DIR, f"seg_{i:03d}_concat_list.txt")
                    with open(concat_path, "w") as cf:
                        for part in shot["musetalk_split_parts"]:
                            cf.write(f"file '{part['output']}'\n")
                    merged_path = os.path.join(WORK_DIR, f"seg_{i:03d}_musetalk_merged.mp4")
                    subprocess.run([
                        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                        "-i", concat_path, "-c:v", "libx264", "-c:a", "aac", merged_path
                    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    musetalk_result = merged_path
                else:
                    musetalk_result = shot["musetalk_output_path"]

                if shot.get("crop_region"):
                    composited_path = os.path.join(WORK_DIR, f"seg_{i:03d}_composited.mp4")
                    composite_crop_back(
                        shot["video_segment_path"], musetalk_result,
                        tuple(shot["crop_region"]), composited_path
                    )
                    source = composited_path
                else:
                    source = musetalk_result
            else:
                source = shot["video_segment_path"]

            silent_path = os.path.join(WORK_DIR, f"seg_{i:03d}_silent.mp4")
            subprocess.run([
                "ffmpeg", "-y", "-i", source, "-an",
                "-c:v", "libx264",
                "-vf", "scale=1920:1080,format=yuv420p",
                "-colorspace", "bt709",
                "-color_primaries", "bt709",
                "-color_trc", "bt709",
                "-force_key_frames", "expr:gte(t,0)",
                silent_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            f.write(f"file '{silent_path}'\n")

    concatenated_path = os.path.join(WORK_DIR, "concatenated_silent.mp4")

    input_args = []
    for i in range(len(shots)):
        input_args += ["-i", os.path.join(WORK_DIR, f"seg_{i:03d}_silent.mp4")]

    filter_str = "".join([f"[{i}:v]" for i in range(len(shots))]) + f"concat=n={len(shots)}:v=1:a=0[out]"

    subprocess.run([
        "ffmpeg", "-y",
        *input_args,
        "-filter_complex", filter_str,
        "-map", "[out]",
        "-c:v", "libx264",
        concatenated_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    subprocess.run([
        "ffmpeg", "-y",
        "-i", concatenated_path,
        "-i", hindi_audio_path,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f"\n[Stage 6] Final video assembled: {output_path}")
    return output_path


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "one_person_video"

    if len(sys.argv) < 3:
        print("Usage: python assemble_pipeline.py <video_path> <audio_path> [output_path]")
        sys.exit(1)

    VIDEO_PATH = sys.argv[1]
    HINDI_AUDIO_PATH = sys.argv[2]
    OUTPUT_PATH = sys.argv[3] if len(sys.argv) > 3 else r"C:\lipsync\outputs\final_dubbed_output.mp4"

    mode = os.path.splitext(os.path.basename(VIDEO_PATH))[0]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # extract original video audio for accurate shot classification
    original_audio_path = os.path.join(os.path.dirname(VIDEO_PATH), "original_audio_tmp.wav")
    subprocess.run([
        "ffmpeg", "-y", "-i", VIDEO_PATH,
        "-vn", "-ar", "44100", "-ac", "2", original_audio_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    manifest_path = os.path.join(WORK_DIR, f"manifest_{mode}.json")
    needs_prepare = True

    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            existing = json.load(f)
        if existing and existing[0].get("video_segment_path", "").find("seg_") != -1:
            all_musetalk_ready = all(
                os.path.exists(shot["musetalk_output_path"])
                for shot in existing
                if shot.get("apply_lipsync") and shot.get("musetalk_output_path")
                and not shot.get("musetalk_split_parts")
            )
            if all_musetalk_ready:
                needs_prepare = False
                print(f"[Pipeline] Manifest found with all MuseTalk outputs ready -- running stitch_final")

    if needs_prepare:
        print(f"[Pipeline] Running prepare_segments for {mode}...")
        prepare_segments(VIDEO_PATH, HINDI_AUDIO_PATH, mode,
                        classification_audio_path=original_audio_path)
        print("[Pipeline] Upload the listed segments to Colab, run MuseTalk, download outputs, then run this script again.")
    else:
        stitch_final(VIDEO_PATH, HINDI_AUDIO_PATH, OUTPUT_PATH, mode)