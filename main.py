import pandas as pd

from indexer import *
from embeddings import *
import operator

import torch
from transformers import ViTModel, ViTFeatureExtractor
from panns_inference import AudioTagging

from peaks import *
from db_utils import *
from sklearn.metrics.pairwise import cosine_similarity

model_audio = AudioTagging(checkpoint_path=None, device='cuda' if torch.cuda.is_available() else 'cpu')


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
    csv_path = "piracy_val.csv"
    ground_truth = pd.DataFrame(pd.read_csv(csv_path))
    pirate_video = "val/"
    pirate_files = os.listdir(pirate_video)
    for file in pirate_files:
        percent_dict = {}
        if file.endswith(".mp4"):
            dict_data = get_video_embeddings(os.path.join(pirate_video, file), model, feature_extractor, model_audio)
            for table_name in database.table_names():
                table = database.open_table(table_name)
                table_filename = table_name.split("$")[1]
                full_embedding_video = table.search().where(f"filename = {table_filename}").to_list()
                full_embedding_video_vec = [x["vector_video"] for x in full_embedding_video]
                full_embedding_audio = table.search().where(f"filename = {table_filename}").to_list()
                full_embedding_audio_vec = [x["vector_audio"] for x in full_embedding_audio]
                matrix = cosine_similarity(dict_data["video"], full_embedding_video_vec)
                matrix_audio = cosine_similarity(dict_data["audio"], full_embedding_audio_vec)
                matrix = matrix + matrix_audio
                result_peaks_columns = make_plt_columns(matrix)
                if result_peaks_columns["interval"] == "":
                    continue
                else:
                    result_peaks_rows = make_plt_rows(matrix)
                    interval1 = result_peaks_columns["interval"]
                    interval2 = result_peaks_rows["interval"]
                    intervals = f"{interval1} {interval2}"
                    percent_dict[table_filename] = {
                        "score": result_peaks_columns["width"] + result_peaks_columns["height"],
                        "intervals": f"{intervals}"}

            predicted_license_video = max(percent_dict.items(), key=lambda item: item[1]["score"])[0]
            if percent_dict[predicted_license_video]["score"] < threshold:
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


# def check_similarity(model, feature_extractor):
#     """
#     Check similarity between two videos
#     :param model: model like vit transformer
#     :param feature_extractor: model like ViTFeatureExtractor
#     :return:
#     """
#
#     dict_data = get_video_embeddings("ydcrodwtz3mstjq1vhbdflx6kyhj3y0p.mp4",
#                                      model, feature_extractor, model_audio)
#     dict_data_1 = get_video_embeddings("ded3d179001b3f679a0101be95405d2c.mp4",
#                                        model, feature_extractor, model_audio)
#
#     _ = create_lance_table("video_embeddings", len(dict_data_1["video"][0]))
#     database = create_lance_table("audio_embeddings", len(dict_data_1["audio"][0]))
#
#     append_vector_to_table(database, "video_embeddings", dict_data_1["video"],
#                            dict_data_1["segments"], dict_data_1["filenames"])
#     append_vector_to_table(database, "audio_embeddings", dict_data_1["audio"],
#                            dict_data_1["segments"], dict_data_1["filenames"])
#
#     table_video = database.open_table("video_embeddings")
#     table_audio = database.open_table("audio_embeddings")
#     threshold_video = 0.6
#     threshold_audio = 0.08
#     percent_dict = {}
#     for batch in dict_data["video"]:
#         result = table_video.search(batch, vector_column_name="vector").metric("cosine").limit(10).to_list()
#         if result[0]["_distance"] < threshold_video:
#             if not percent_dict.get(result[0]["filename"]):
#                 percent_dict[result[0]["filename"]] = 1
#             else:
#                 percent_dict[result[0]["filename"]] += 1
#     print("-" * 40)
#
#     for batch in dict_data["audio"]:
#         result = table_audio.search(batch, vector_column_name="vector").metric("cosine").limit(10).to_list()
#         if result[0]["_distance"] < threshold_audio:
#             if not percent_dict.get(result[0]["filename"]):
#                 percent_dict[result[0]["filename"]] = 1
#             else:
#                 percent_dict[result[0]["filename"]] += 1
#     for key, _ in percent_dict.items():
#         percent_dict[key] = percent_dict[key] / (len(dict_data["video"] * 2))
#     print(percent_dict)
#     matrix = cosine_similarity(dict_data["video"], dict_data_1["video"])
#     matrix_audio = cosine_similarity(dict_data["audio"], dict_data_1["audio"])
#     matrix = matrix + matrix_audio
#     print(max(percent_dict.items(), key=operator.itemgetter(1))[0])  # add threshold for final result
#     make_plt_rows(matrix, True)
#     make_plt_columns(matrix, True)
#

if "__main__" == __name__:
    model = ViTModel.from_pretrained('google/vit-base-patch16-224')
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    CONFIDENCE_THRESHOLD = 0.05
    database = get_embeddings_for_directory("index/", model, feature_extractor, model_audio)
    # check_similarity(model, feature_extractor)
    print(f1_for_all_search(model, feature_extractor, database, CONFIDENCE_THRESHOLD))
