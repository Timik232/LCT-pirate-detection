import os

import lancedb
import librosa
import numpy as np
import pyarrow as pa
import moviepy.editor as mp
import cv2
from PIL import Image
import torch
from transformers import ViTModel, ViTFeatureExtractor


def get_embeddings_vit(frames, feature_extractor_l, model_l):
    # Преобразуем кадры из формата OpenCV в формат PIL
    images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]
    inputs = feature_extractor_l(images=images, return_tensors="pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Перемещаем данные на GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model_l(**inputs)
    return outputs.last_hidden_state.mean(dim=1)


def extract_frame_embeddings_vit(video_path, model_l, feature_extractor_l, frame_interval=None):
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


def get_sound_embedding(audio_path):
    sr = 22050
    y, sr = librosa.load(audio_path, sr=sr)

    # Длина одного фрейма в секундах
    frame_length = sr

    # Количество фреймов (секунд) в аудиофайле
    num_frames = len(y) // frame_length

    spectrograms = []

    for i in range(num_frames):
        # Извлечь одну секунду аудио
        frame = y[i * frame_length: (i + 1) * frame_length]

        # Преобразовать в спектрограмму
        spectrogram = librosa.feature.melspectrogram(y=frame, sr=sr)

        # Преобразовать спектрограмму в логарифмическую шкалу
        log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        flat_spectrogram = log_spectrogram.flatten()
        spectrograms.append(flat_spectrogram)

    return np.array(spectrograms)


def extract_audio_from_mp4(file_path):
    video = mp.VideoFileClip(file_path)
    temp_audio_path = "temp_audio.wav"
    video.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)

    return temp_audio_path


def get_video_embeddings(filename, model_l, feature_extractor_l):
    video_embeddings_l = np.array([])
    audio_embeddings_l = np.array([])
    segments = []
    filenames = []
    video = mp.VideoFileClip(filename)
    video_duration = video.duration
    segment_duration = 10
    start_time = 0
    segment_index = 0
    while start_time < video_duration:
        end_time = min(start_time + segment_duration, video_duration)
        if end_time - start_time != segment_duration:
            start_time += segment_duration
            continue
        segment = video.subclip(start_time, end_time)
        segment.write_videofile("temp_segment.mp4", fps=video.fps)
        stack = np.vstack
        if len(audio_embeddings_l) == 0:
            stack = np.hstack
        audio_embeddings_l = stack((audio_embeddings_l,
                                    get_sound_embedding(extract_audio_from_mp4("temp_segment.mp4")).flatten()))
        video_embeddings_l = stack((video_embeddings_l,
                                    extract_frame_embeddings_vit("temp_segment.mp4", model_l,
                                                                 feature_extractor_l).flatten()))
        segments.append(segment_index)
        filenames.append(filename)
        start_time += segment_duration
        segment_index += 1
    return {"video": np.array(video_embeddings_l), "audio": np.array(audio_embeddings_l),
            "segments": segments, "filenames": filenames}


def create_lance_table(table_name, vector_dim):
    schema = pa.schema([
        ('vector', pa.list_(pa.float64(), vector_dim)),
        ('segment_time', pa.int64()),
        ('filename', pa.string())
    ])
    print(f"Vector dim {vector_dim}")
    db_path = "data/"
    if not os.path.exists(db_path):
        os.makedirs(db_path)
    db = lancedb.connect(db_path)
    _ = db.create_table(table_name, schema=schema)
    return db


def append_vector_to_table(db: lancedb.DBConnection, table_name: str, vector, segments, filenames):
    table = db.open_table(table_name)

    for i in range(len(vector)):
        segment_time_array = pa.array([segments[i]])
        filename_array = pa.array([filenames[i]])
        vector_array = pa.array([vector[i]])
        t = pa.Table.from_arrays([vector_array, segment_time_array, filename_array], schema=table.schema)
        table.add(t)


model = ViTModel.from_pretrained('google/vit-base-patch16-224')
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


dict_data = get_video_embeddings("ded3d179001b3f679a0101be95405d2c.mp4",
                                 model, feature_extractor)
dict_data_1 = get_video_embeddings("f4b1fd188fe77f9f56de07e867128b13.mp4",
                                   model, feature_extractor)

_ = create_lance_table("video_embeddings", len(dict_data["video"][0]))
database = create_lance_table("audio_embeddings", len(dict_data["audio"][0]))

append_vector_to_table(database, "video_embeddings", dict_data["video"], dict_data["segments"], dict_data["filenames"])
append_vector_to_table(database, "audio_embeddings", dict_data["audio"], dict_data["segments"], dict_data["filenames"])
table_video = database.open_table("video_embeddings")
table_audio = database.open_table("audio_embeddings")
percent_video = 0
percent_audio = 0
for batch in dict_data_1["video"]:
    result = table_video.search(batch, vector_column_name="vector").limit(10).to_list()
