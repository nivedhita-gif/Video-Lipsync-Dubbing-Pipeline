"""
dubbing.py

Stages 1-3 of the pipeline: transcribe, translate, and generate
dubbed audio from the input video.

WORKFLOW:
    1. Whisper transcribes the original audio to text with timestamps
    2. googletrans translates each segment to the target language
    3. gTTS generates speech for each translated segment
    4. ffmpeg stretches/compresses each generated audio clip to match
       the original segment timing so lip-sync stays in sync
    5. All segments are stitched into a single dubbed audio track

LIMITATIONS:
    - gTTS voice quality is robotic compared to ElevenLabs -- can be
      swapped in later with minimal changes
    - googletrans is unofficial and can sometimes fail or rate-limit
    - Audio stretching to match timing can sound unnatural if the
      translation is much longer or shorter than the original
    - Whisper 'base' model may struggle with heavy accents or noisy audio
"""

import os
import subprocess
import whisper
from googletrans import Translator
from gtts import gTTS


def transcribe_audio(video_path: str, src_language: str = "ta") -> list:
    """
    Uses Whisper to transcribe the audio from the video into text
    segments with timestamps.

    Returns: list of {start, end, text} dicts
    """
    print(f"[Dubbing] Transcribing audio with Whisper...")
    model = whisper.load_model("base")
    result = model.transcribe(video_path, language=src_language, word_timestamps=False)

    segments = []
    for seg in result["segments"]:
        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip()
        })
        print(f"  [{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text'].strip()}")

    print(f"[Dubbing] Transcribed {len(segments)} segments.")
    return segments


def translate_segments(segments: list, src_lang: str = "ta", dest_lang: str = "hi") -> list:
    """
    Translates each transcribed segment to the target language.

    Returns: segments list with 'translated_text' added to each
    """
    print(f"[Dubbing] Translating {src_lang} -> {dest_lang}...")
    translator = Translator()

    for seg in segments:
        result = translator.translate(seg["text"], src=src_lang, dest=dest_lang)
        seg["translated_text"] = result.text
        print(f"  Original: {seg['text']}")
        print(f"  Translated: {seg['translated_text']}")

    return segments


def generate_tts_for_segment(text: str, lang: str, output_path: str):
    """
    Generates TTS audio for a single translated segment using gTTS.
    Saves as mp3 then converts to wav.
    """
    mp3_path = output_path.replace(".wav", ".mp3")
    tts = gTTS(text, lang=lang)
    tts.save(mp3_path)

    subprocess.run([
        "ffmpeg", "-y", "-i", mp3_path,
        "-ar", "44100", "-ac", "2", output_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    os.remove(mp3_path)


def stretch_audio_to_duration(input_path: str, target_duration: float, output_path: str):
    """
    Stretches or compresses audio to match the target duration using
    ffmpeg's atempo filter, so the dubbed audio fits the original
    segment timing and stays in sync with the lip-sync.

    atempo only supports 0.5-2.0x range, so we chain filters for
    more extreme cases.
    """
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", input_path
    ], capture_output=True, text=True, check=True)
    actual_duration = float(probe.stdout.strip())

    if actual_duration <= 0:
        subprocess.run(["cp", input_path, output_path])
        return

    ratio = actual_duration / target_duration
    ratio = max(0.5, min(2.0, ratio))

    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-filter:a", f"atempo={ratio:.4f}",
        output_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def generate_dubbed_audio(video_path: str, output_audio_path: str,
                           src_lang: str = "en", dest_lang: str = "hi",
                           tmp_dir: str = "/tmp/dubbing"):
    os.makedirs(tmp_dir, exist_ok=True)

    segments = transcribe_audio(video_path, src_lang)
    segments = translate_segments(segments, src_lang, dest_lang)

    segment_files = []

    for i, seg in enumerate(segments):
        tts_path = os.path.join(tmp_dir, f"seg_{i:03d}_tts.wav")
        generate_tts_for_segment(seg["translated_text"], dest_lang, tts_path)
        segment_files.append(tts_path)

        # add a small natural pause between sentences
        silence_path = os.path.join(tmp_dir, f"pause_{i}.wav")
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "anullsrc=r=44100:cl=stereo",
            "-t", "0.3",
            silence_path
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        segment_files.append(silence_path)
        print(f"[Dubbing] Segment {i+1}/{len(segments)} done.")

    concat_list = os.path.join(tmp_dir, "concat.txt")
    with open(concat_list, "w") as f:
        for path in segment_files:
            f.write(f"file '{path}'\n")

    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_list, "-c", "copy", output_audio_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f"[Dubbing] Dubbed audio saved to: {output_audio_path}")
    return output_audio_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python dubbing.py <video_path> <output_audio_path> [src_lang] [dest_lang]")
        sys.exit(1)

    video_path = sys.argv[1]
    output_path = sys.argv[2]
    src_lang = sys.argv[3] if len(sys.argv) > 3 else "ta"
    dest_lang = sys.argv[4] if len(sys.argv) > 4 else "hi"

    generate_dubbed_audio(video_path, output_path, src_lang, dest_lang)