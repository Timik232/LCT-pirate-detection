import json
import os
from db_utils import *
from embeddings import *


def get_embeddings_for_directory(directory: str, model_l, feature_extractor_l, model_audio) -> lancedb.DBConnection:
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
            dict_data = get_video_embeddings(os.path.join(directory, file), model_l, feature_extractor_l, model_audio)
            _ = create_lance_table("video_embeddings", len(dict_data["video"][0]))
            database = create_lance_table("audio_embeddings", len(dict_data["audio"][0]))

            append_vector_to_table(database, "video_embeddings", dict_data["video"],
                                   dict_data["segments"], dict_data["filenames"])
            append_vector_to_table(database, "audio_embeddings", dict_data["audio"],
                                   dict_data["segments"], dict_data["filenames"])
            with open("indexed_files.json", "w", encoding="utf-8") as f:
                json.dump(indexed_files, f, indent=4)
                print(f"Indexed {file}")
    return database
