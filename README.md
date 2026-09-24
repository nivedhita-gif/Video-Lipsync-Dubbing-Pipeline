# Video-Lipsync-Dubbing-Pipeline
AI pipeline that transcribes, translates, and lip-syncs video from one language to another - with automatic shot classification and multi-person speaker detection.
# Automated Video Dubbing & Lip-Sync Pipeline

An end-to-end pipeline that dubs video from one language to another and automatically syncs the speaker's mouth movements to match the translated audio.

## What it does

- Transcribes speech from the input video using OpenAI Whisper
- Translates the transcription to the target language
- Generates dubbed audio using Text-to-Speech
- Automatically detects which scenes need lip-sync (shot classification)
- Identifies the active speaker in multi-person scenes (speaker detection)
- Runs lip-sync using MuseTalk on Google Colab
- Composites and stitches the final dubbed video

## Works for

- Single-speaker videos
- Multi-person videos (correctly identifies and animates only the active speaker)

## Project Structure

| File | Description |
|------|-------------|
| `pipeline/assemble_pipeline.py` | Main pipeline — shot cutting, compositing, stitching |
| `pipeline/shot_classifier.py` | Stage 4 — detects which scenes need lip-sync |
| `pipeline/speaker_detection.py` | Speaker identification for multi-person scenes |
| `pipeline/dubbing.py` | Stages 1-3 — transcription, translation, TTS |
| `pipeline/musetalk_patches.py` | Patches for MuseTalk compatibility |

## How it works

### Shot Classification
Uses WebRTC VAD (voice activity detection) combined with MediaPipe face detection to automatically classify each part of the video as either LIP-SYNC (face visible + speech happening) or AUDIO ONLY (no face or no speech). Only LIP-SYNC segments are sent to MuseTalk.

### Speaker Detection
For multi-person scenes, identifies the active speaker by correlating each person's mouth movement velocity with the audio energy over time using Pearson correlation. The video is cropped to the speaker's face before being sent to MuseTalk, preventing it from animating the wrong person.

### Lip-Sync (MuseTalk)
MuseTalk regenerates the speaker's mouth region frame by frame to match the dubbed audio's phonemes. Runs on GPU via Google Colab. Long segments are automatically split at natural silence points to prevent face-tracking drift.

### Assembly
After MuseTalk, the pipeline composites the lip-synced face back onto the original full frame (for multi-person shots), normalizes all segments to consistent resolution and color space, concatenates everything in timeline order, and overlays the final dubbed audio track.

## Usage

```bash
python pipeline/assemble_pipeline.py <video_path> <audio_path> [output_path]

# Example
python pipeline/assemble_pipeline.py input_video.mp4 dubbed_audio.wav output.mp4
```

## Requirements

- Python 3.10
- MuseTalk (run on Google Colab — GPU required)
- MediaPipe
- OpenCV
- WebRTC VAD
- librosa
- OpenAI Whisper
- gTTS
- ffmpeg

## Tech Stack

Python · MuseTalk · OpenAI Whisper · MediaPipe · OpenCV · WebRTC VAD · librosa · gTTS · ffmpeg · Google Colab

## Limitations

- MuseTalk works best on frontal faces — angled shots may produce blur
- gTTS voice quality is robotic — designed to be replaced with ElevenLabs for production use
- Speaker detection relies on audio-mouth correlation — noisy audio can reduce accuracy
- Boundary transitions between AUDIO ONLY and LIP-SYNC segments have a subtle visual difference
  
## Testing

To test the pipeline you'll need:
1. A short video clip (10-30 seconds) with a person speaking clearly to camera
2. Either the same video's audio (to test lip-sync quality) or a dubbed audio track in another language

Good sources for free test videos:
- [Pexels](https://www.pexels.com/videos/) — free CC0 licensed videos
- [Internet Archive Prelinger Collection](https://archive.org/details/prelinger) — public domain films with clear dialogue

Then run:
```bash
python pipeline/assemble_pipeline.py your_video.mp4 your_audio.wav output.mp4
```

Note: MuseTalk requires GPU — you'll need to run the lip-sync step on Google Colab. The pipeline will print the exact Colab cell to run after preparing segments.

## Acknowledgements

Lip-sync powered by [MuseTalk](https://github.com/TMElyralab/MuseTalk)
