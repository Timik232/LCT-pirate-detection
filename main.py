import json
import os
import tempfile
import pandas as pd
from numba import njit
import lancedb
import librosa
import numpy as np
import pyarrow as pa
import moviepy.editor as mp
import operator
import cv2
from PIL import Image
import torch
from transformers import ViTModel, ViTFeatureExtractor
from panns_inference import AudioTagging
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

from scipy.interpolate import PchipInterpolator
from scipy.signal import find_peaks, peak_widths

model_audio = AudioTagging(checkpoint_path=None, device='cuda' if torch.cuda.is_available() else 'cpu')


def make_plt_rows(matrix_l):
    """
    Make plot of rows
    :param matrix_l:
    :return:
    """
    points_dict = {}

    # Проходимся по строкам матрицы косинусных расстояний
    for i in range(max(matrix_l.shape[0], matrix_l.shape[1])):
        points_dict[i] = 0
    max_value_global = np.max(np.abs(matrix_l))
    mean_value_global = np.mean(np.abs(matrix_l))
    for i in range(matrix_l.shape[0]):
        max_index = np.argmax(matrix_l[i])
        max_value = matrix_l[i, max_index]
        points_dict[max_index] = max_value

    print(points_dict)
    # Преобразуем словарь в списки для построения графика
    x_points = list(points_dict.keys())
    y_points = list(points_dict.values())
    # Убедимся, что все значения y положительны
    y_points = np.maximum(y_points, 0)

    sorted_indices = np.argsort(x_points)
    print(max(x_points))
    x_points = np.take(np.array(x_points), sorted_indices)
    y_points = np.take(np.array(y_points), sorted_indices)
    print(max(x_points))

    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode='same')

    window_size = max(int(y_points.shape[0] * 0.05), 3)
    y_points_smoothed = moving_average(y_points, window_size)

    f_interp = PchipInterpolator(x_points, y_points_smoothed)
    x_smooth = np.linspace(x_points.min(), x_points.max(), len(x_points))
    y_smooth = f_interp(x_smooth)

    peaks, _ = find_peaks(y_smooth)
    widths_half_max = peak_widths(y_smooth, peaks, rel_height=0.50)

    max_peak_idx = np.argmax(y_smooth[peaks])
    max_peak_height = y_smooth[peaks][max_peak_idx]

    max_width_idx = np.argmax(widths_half_max[0])
    max_peak_width = widths_half_max[0][max_width_idx]
    left_ips_x = 0
    right_ips_x = 0
    if peaks[max_peak_idx] == peaks[max_width_idx]:
        print(f"Пик одновременно самый высокий и самый широкий: высота={max_peak_height}, ширина={max_peak_width}")

        left_ips_x = x_smooth[int(widths_half_max[2][max_width_idx])]
        right_ips_x = x_smooth[int(widths_half_max[3][max_width_idx])]

        print(f"Границы пика относительно оси x: {left_ips_x}, {right_ips_x}")
    elif peaks[max_width_idx]:  #and widths_half_max[0][max_peak_idx] > 50:
        left_ips_x = x_smooth[int(widths_half_max[2][max_peak_idx])]
        right_ips_x = x_smooth[int(widths_half_max[3][max_peak_idx])]

        print(f"Границы пика относительно оси x: {left_ips_x}, {right_ips_x}")
    plt.plot(x_smooth, y_smooth)
    plt.xlabel('Index of Minimum Cosine Distance')
    plt.ylabel('Sum of Max Value - Min Value')
    plt.title('Graph of Minimum Cosine Distances with Peaks')
    plt.grid(True)
    plt.show()
    return f"{left_ips_x}-{right_ips_x}"


def make_plt_columns(matrix_l):
    """
    Make plot of columns
    :param matrix_l:
    :return:
    """
    points_dict = {}
    for j in range(max(matrix_l.shape[0], matrix_l.shape[1])):
        points_dict[j] = 0
    print(matrix_l.shape[1])
    max_value_global = np.max(np.abs(matrix_l))
    mean_value_global = np.mean(np.abs(matrix_l))
    # Проходимся по строкам матрицы косинусных расстояний
    for j in range(matrix_l.shape[1]):
        max_index = np.argmax(matrix_l[:, j])
        max_value = matrix_l[max_index, j]
        points_dict[max_index] = max_value

    x_points = list(points_dict.keys())
    y_points = list(points_dict.values())

    sorted_indices = np.argsort(x_points)
    print(max(x_points))
    x_points = np.take(np.array(x_points), sorted_indices)
    y_points = np.take(np.array(y_points), sorted_indices)
    print(max(x_points))

    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode='same')

    window_size = max(int(y_points.shape[0] * 0.05), 3)
    y_points_smoothed = moving_average(y_points, window_size)

    f_interp = PchipInterpolator(x_points, y_points_smoothed)
    x_smooth = np.linspace(x_points.min(), x_points.max(), len(x_points))
    y_smooth = f_interp(x_smooth)

    peaks, _ = find_peaks(y_smooth)
    widths_half_max = peak_widths(y_smooth, peaks, rel_height=0.50)

    max_peak_idx = np.argmax(y_smooth[peaks])
    max_peak_height = y_smooth[peaks][max_peak_idx]

    max_width_idx = np.argmax(widths_half_max[0])
    max_peak_width = widths_half_max[0][max_width_idx]
    left_ips_x = 0
    right_ips_x = 0
    if peaks[max_peak_idx] == peaks[max_width_idx]:
        print(f"Пик одновременно самый высокий и самый широкий: высота={max_peak_height}, ширина={max_peak_width}")

        left_ips_x = x_smooth[int(widths_half_max[2][max_width_idx])]
        right_ips_x = x_smooth[int(widths_half_max[3][max_width_idx])]

        print(f"Границы пика относительно оси x: {left_ips_x}, {right_ips_x}")
    elif peaks[max_width_idx]:  #and widths_half_max[0][max_peak_idx] > 50#:
        left_ips_x = x_smooth[int(widths_half_max[2][max_peak_idx])]
        right_ips_x = x_smooth[int(widths_half_max[3][max_peak_idx])]

        print(f"Границы пика относительно оси x: {left_ips_x}, {right_ips_x}")

    plt.plot(x_smooth, y_smooth)
    plt.xlabel('Index of Minimum Cosine Distance')
    plt.ylabel('Sum of Max Value - Min Value')
    plt.title('Graph of Minimum Cosine Distances with Peaks')
    plt.grid(True)
    plt.show()
    return f"{left_ips_x}-{right_ips_x}"


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
    :param frame_interval:
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


def get_sound_embedding(audio_path: str, n=10) -> np.ndarray:
    """
    Extract sound embedding from audio file
    :param audio_path: path to the audio file
    :param n:
    :return: embedding of audio
    """
    (audio, _) = librosa.core.load(audio_path, sr=44100, mono=True)
    _, emb = model_audio.inference(audio[None, :])
    return np.array([emb[0]] * n)


def extract_audio_from_mp4(file_path: str, temp_dir) -> str:
    """
    Extract audio from mp4 file to wav
    :param file_path: path to the audio file
    :param temp_dir: temporary directory
    :return: path to the new file
    """
    video = mp.VideoFileClip(file_path)
    temp_audio_path = os.path.join(temp_dir.name, "temp_audio.wav")
    video.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)

    return temp_audio_path


def get_video_embeddings(filename: str, model_l, feature_extractor_l) -> dict:
    """
    Get embeddings from video
    :param filename: name of the file
    :param model_l: model like VIT transformer
    :param feature_extractor_l: model like ViTFeatureExtractor
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
    while start_time < video_duration:
        end_time = min(start_time + segment_duration, video_duration)
        if end_time - start_time != segment_duration:
            start_time += segment_duration
            continue
        segment = video.subclip(start_time, end_time)
        temp_video_path = os.path.join(temp_dir.name, "temp_segment.mp4")
        segment.write_videofile(temp_video_path, fps=video.fps)
        # stack = np.vstack
        # if len(audio_embeddings_l) == 0:
        #     stack = np.hstack
        audio_embeddings_l = np.concatenate((audio_embeddings_l,
                                             get_sound_embedding(extract_audio_from_mp4(temp_video_path, temp_dir))), axis=0)
        video_embeddings_l = np.concatenate((video_embeddings_l,
                                             extract_frame_embeddings_vit(temp_video_path, model_l,
                                                                          feature_extractor_l)), axis=0)
        segments.extend([segment_index] * 10)
        filenames.extend([filename] * 10)
        start_time += segment_duration
        segment_index += 1
    temp_dir.cleanup()
    return {"video": np.array(video_embeddings_l), "audio": np.array(audio_embeddings_l),
            "segments": segments, "filenames": filenames}


def create_lance_table(table_name: str, vector_dim) -> lancedb.DBConnection:
    """
    create vector table
    :param table_name: name of the table
    :param vector_dim: dimension of the vector
    :return: database connection
    """
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
    if table_name not in db.table_names():
        _ = db.create_table(table_name, schema=schema)
    return db


def append_vector_to_table(db: lancedb.DBConnection, table_name: str, vector, segments, filenames: list) -> None:
    """
    append vector of embeddings to table
    :param db: lancedb connection
    :param table_name: name of the table
    :param vector:
    :param segments:
    :param filenames:
    :return:
    """
    table = db.open_table(table_name)

    for i in range(len(segments)):
        segment_time_array = pa.array([segments[i]])
        filename_array = pa.array([filenames[i]])
        vector_array = pa.array([vector[i]])
        t = pa.Table.from_arrays([vector_array, segment_time_array, filename_array], schema=table.schema)
        table.add(t)


def calculate_f1_score(true_positives: int, false_positives: int, false_negatives: int) -> float:
    """
    Calculate the F1 score based on true positives, false positives, and false negatives.
    """
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1_score


def find_license_by_pirate_name(df: pd.DataFrame, file_name: str):
    """
    Find the ID of a file with a certain name in a Pandas DataFrame.
    :param df: Pandas DataFrame
    :param file_name: name of the file
    :return: ID of the file
    """
    matching_row = df[df['ID_piracy'] == file_name]
    if matching_row.empty:
        return None
    else:
        return matching_row['ID_license'].values[0]


def f1_for_all_search(model, feature_extractor, database, threshold: float) -> float:
    """
    calculate f1 for all search video
    :param model: model like vit transformer
    :param feature_extractor: model like ViTFeatureExtractor
    :param database: database connection
    :param threshold: threshold for confidence of the found video
    :return:
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    table_video = database.open_table("video_embeddings")
    table_audio = database.open_table("audio_embeddings")
    threshold_video = 0.6
    threshold_audio = 0.08
    csv_path = "piracy_val.csv"
    ground_truth = pd.DataFrame(pd.read_csv(csv_path))
    pirate_video = "val/"
    pirate_files = os.listdir(pirate_video)
    for file in pirate_files:
        percent_dict = {}
        if file.endswith(".mp4"):
            dict_data = get_video_embeddings(os.path.join(pirate_video, file), model, feature_extractor)
            for batch in dict_data["video"]:
                result = table_video.search(batch, vector_column_name="vector").metric("cosine").limit(10).to_list()
                if result[0]["_distance"] < threshold_video:
                    if not percent_dict.get(result[0]["filename"]):
                        percent_dict[result[0]["filename"]] = 1
                    else:
                        percent_dict[result[0]["filename"]] += 1
            for batch in dict_data["audio"]:
                result = table_audio.search(batch, vector_column_name="vector").metric("cosine").limit(10).to_list()
                if result[0]["_distance"] < threshold_audio:
                    if not percent_dict.get(result[0]["filename"]):
                        percent_dict[result[0]["filename"]] = 1
                    else:
                        percent_dict[result[0]["filename"]] += 1
            for key, _ in percent_dict.items():
                percent_dict[key] = percent_dict[key] / (len(dict_data["video"] * 2))
            predicted_license_video = max(percent_dict.items(), key=operator.itemgetter(1))[0]
            if percent_dict[predicted_license_video] < threshold:
                predicted_license_video = None
            if predicted_license_video is None:
                false_negatives += 1
            else:
                true_license_id = find_license_by_pirate_name(ground_truth, file)
                if predicted_license_video == true_license_id:
                    true_positives += 1
                else:
                    false_positives += 1
    return calculate_f1_score(true_positives, false_positives, false_negatives)


def get_embeddings_for_directory(directory: str, model_l, feature_extractor_l) -> lancedb.DBConnection:
    """
    Get embeddings for all files in the directory
    :param directory: path to the directory
    :param model_l: model like VIT transformer
    :param feature_extractor_l: model like ViTFeatureExtractor
    :return: database connection
    """
    if not os.path.exists("indexed_files.json"):
        indexed_files = {"indexed_files": []}
    else:
        indexed_files = json.load(open("indexed_files.json", "r", encoding="utf-8"))
    database = None
    files = os.listdir(directory)
    for file in files:
        if file.endswith(".mp4"):
            if file in indexed_files["indexed_files"]:
                continue
            else:
                indexed_files["indexed_files"].append(file)
            dict_data = get_video_embeddings(os.path.join(directory, file), model_l, feature_extractor_l)
            _ = create_lance_table("video_embeddings", len(dict_data["video"][0]))
            database = create_lance_table("audio_embeddings", len(dict_data["audio"][0]))

            append_vector_to_table(database, "video_embeddings", dict_data["video"],
                                   dict_data["segments"], dict_data["filenames"])
            append_vector_to_table(database, "audio_embeddings", dict_data["audio"],
                                   dict_data["segments"], dict_data["filenames"])
            with open("indexed_files.json", "w", encoding="utf-8") as f:
                json.dump(indexed_files, f, indent=4)
    return database


def check_similarity(model, feature_extractor):
    """
    Check similarity between two videos
    :param model: model like vit transformer
    :param feature_extractor: model like ViTFeatureExtractor
    :return:
    """

    dict_data = get_video_embeddings("ydcrodwtz3mstjq1vhbdflx6kyhj3y0p.mp4",
                                     model, feature_extractor)
    dict_data_1 = get_video_embeddings("ded3d179001b3f679a0101be95405d2c.mp4",
                                       model, feature_extractor)

    _ = create_lance_table("video_embeddings", len(dict_data_1["video"][0]))
    database = create_lance_table("audio_embeddings", len(dict_data_1["audio"][0]))

    append_vector_to_table(database, "video_embeddings", dict_data_1["video"],
                           dict_data_1["segments"], dict_data_1["filenames"])
    append_vector_to_table(database, "audio_embeddings", dict_data_1["audio"],
                           dict_data_1["segments"], dict_data_1["filenames"])

    table_video = database.open_table("video_embeddings")
    table_audio = database.open_table("audio_embeddings")
    threshold_video = 0.6
    threshold_audio = 0.08
    percent_dict = {}
    for batch in dict_data["video"]:
        result = table_video.search(batch, vector_column_name="vector").metric("cosine").limit(10).to_list()
        if result[0]["_distance"] < threshold_video:
            if not percent_dict.get(result[0]["filename"]):
                percent_dict[result[0]["filename"]] = 1
            else:
                percent_dict[result[0]["filename"]] += 1
    print("-" * 40)

    for batch in dict_data["audio"]:
        result = table_audio.search(batch, vector_column_name="vector").metric("cosine").limit(10).to_list()
        if result[0]["_distance"] < threshold_audio:
            if not percent_dict.get(result[0]["filename"]):
                percent_dict[result[0]["filename"]] = 1
            else:
                percent_dict[result[0]["filename"]] += 1
    for key, _ in percent_dict.items():
        percent_dict[key] = percent_dict[key] / (len(dict_data["video"] * 2))
    print(percent_dict)
    matrix = cosine_similarity(dict_data["video"], dict_data_1["video"])
    matrix_audio = cosine_similarity(dict_data["audio"], dict_data_1["audio"])
    martix = matrix + matrix_audio
    print(max(percent_dict.items(), key=operator.itemgetter(1))[0]) # add threshold for final result
    make_plt_rows(matrix)
    make_plt_columns(matrix)


if "__main__" == __name__:
    model = ViTModel.from_pretrained('google/vit-base-patch16-224')
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    CONFIDENCE_THRESHOLD = 0.05
    database = get_embeddings_for_directory("index/", model, feature_extractor)
    # check_similarity(model, feature_extractor)
    print(f1_for_all_search(model, feature_extractor, database, CONFIDENCE_THRESHOLD))