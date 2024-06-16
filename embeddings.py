import ntpath
import os
import tempfile

import cv2
import librosa
import numpy as np
import moviepy.editor as mp
import torch
from PIL import Image


def get_embeddings_vit(frames, feature_extractor_l, model_l):
    """
    Get embeddings from frames using vit transformer
    :param frames:
    :param feature_extractor_l: model like ViTFeatureExtractor
    :param model_l: model like VIT transformer
    :return:
    """
    # Преобразуем кадры из формата OpenCV в формат PIL
    images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
    inputs = feature_extractor_l(images=images, return_tensors="pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Перемещаем данные на GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model_l(**inputs)
    return outputs.last_hidden_state.mean(dim=1)


def extract_frame_embeddings_vit(video_path: str, model_l, feature_extractor_l, frame_interval=None) -> np.ndarray:
    """
    Extract frame embeddings from video
    :param video_path: path to the video
    :param model_l: model like VIT transformer
    :param feature_extractor_l: model like ViTFeatureExtractor
    :param frame_interval: interval between frames to extract frame
    :return: embeddings of frames
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if frame_interval is None:
        frame_interval = int(fps)
    print(f"Framerate is {frame_interval}")

    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frames.append(frame)

        frame_count += 1

    cap.release()
    outputs = get_embeddings_vit(frames, feature_extractor_l, model_l)
    return outputs.detach().cpu().squeeze().numpy()


def get_sound_embedding(audio, start_time, end_time, model_audio, n=10):
    start_sample = int(start_time * 44100)
    end_sample = int(end_time * 44100)

    audio_segment = audio[start_sample:end_sample]
    _, emb = model_audio.inference(audio_segment[None, :])
    return np.array([emb[0]] * n)


def extract_audio_from_mp4(file_path: str, temp_dir) -> np.ndarray:
    """
    Extract audio from mp4 file to wav
    :param file_path: path to the audio file
    :param temp_dir: temporary directory
    :return: path to the new file
    """
    video = mp.VideoFileClip(file_path)

    temp_audio_path = os.path.join(temp_dir.name, "temp_audio.wav")
    video.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
    video.close()
    (audio, _) = librosa.core.load(temp_audio_path, sr=44100, mono=True)
    return audio


def get_video_embeddings(filename: str, model_l, feature_extractor_l, model_audio) -> dict:
    """
    Get embeddings from video
    :param filename: name of the file
    :param model_l: model like VIT transformer
    :param feature_extractor_l: model like ViTFeatureExtractor
    :param model_audio: model for audio sources
    :return:
    """
    video_embeddings_l = np.array([]).reshape(0, 768)
    audio_embeddings_l = np.array([]).reshape(0, 2048)
    segments = []
    filenames = []
    video = mp.VideoFileClip(filename)
    video_duration = video.duration
    segment_duration = 10
    start_time = 0
    segment_index = 0
    temp_dir = tempfile.TemporaryDirectory()
    try:
        video_embeddings_l = np.concatenate((video_embeddings_l,
                                             extract_frame_embeddings_vit(filename, model_l,
                                                                          feature_extractor_l)), axis=0)
        audio = extract_audio_from_mp4(filename, temp_dir)
        while start_time < video_duration:
            end_time = min(start_time + segment_duration, video_duration)
            if end_time - start_time != segment_duration:
                start_time += segment_duration
                continue
            # segment = video.subclip(start_time, end_time)
            # temp_video_path = os.path.join(temp_dir.name, "temp_segment.mp4")
            # segment.write_videofile(temp_video_path, fps=video.fps)
            # stack = np.vstack
            # if len(audio_embeddings_l) == 0:
            #     stack = np.hstack
            audio_embeddings_l = np.concatenate((audio_embeddings_l,
                                                 get_sound_embedding(audio, start_time, end_time, model_audio)), axis=0)

            segments.extend([segment_index] * 10)
            filename_base = ntpath.basename(filename)
            filenames.extend([filename_base] * 10)
            start_time += segment_duration
            segment_index += 1
    finally:
        video.close()
        temp_dir.cleanup()
    min_length = min(len(video_embeddings_l), len(audio_embeddings_l), len(segments), len(filenames))
    video_embeddings_l = video_embeddings_l[:min_length]
    audio_embeddings_l = audio_embeddings_l[:min_length]
    segments = segments[:min_length]
    filenames = filenames[:min_length]
    return {"video": np.array(video_embeddings_l), "audio": np.array(audio_embeddings_l),
            "segments": segments, "filenames": filenames}
