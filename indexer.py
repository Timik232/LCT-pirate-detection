import json
import os
from db_utils import *
from embeddings import *

AUDIO_EMBEDDINGS_TABLE = "audio_embeddings"
AUDIO_EMBEDDINGS_DIM = 2048
VIDEO_EMBEDDINGS_TABLE = "video_embeddings"
VIDEO_EMBEDDINGS_DIM = 768


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
    database = create_lance_db()
    files = os.listdir(directory)
    for file in files:
        if file.endswith(".mp4"):
            if file in indexed_files["indexed_files"]:
                continue
            else:
                indexed_files["indexed_files"].append(file)
            dict_data = get_video_embeddings(os.path.join(directory, file), model_l, feature_extractor_l, model_audio)
            append_vectors_to_table(database, f"video_embeddings${file}", dict_data["video"], dict_data["audio"],
                                    dict_data["segments"], dict_data["filenames"], VIDEO_EMBEDDINGS_DIM,
                                    AUDIO_EMBEDDINGS_DIM)
            with open("indexed_files.json", "w", encoding="utf-8") as f:
                json.dump(indexed_files, f, indent=4)
                print(f"Indexed {file}")
    return database
