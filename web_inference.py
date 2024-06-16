import operator
import tempfile
import threading
import time
from queue import Queue

import requests
from flask import request
from tqdm import tqdm
from flask import jsonify
from transformers import ViTModel, ViTFeatureExtractor
from panns_inference import AudioTagging
import torch
import hashlib

from app import app
from indexer import *
from sklearn.metrics.pairwise import cosine_similarity
from peaks import *

AUDIO_EMBEDDINGS_TABLE = "audio_embeddings"
AUDIO_EMBEDDINGS_DIM = 2048
VIDEO_EMBEDDINGS_TABLE = "video_embeddings"
VIDEO_EMBEDDINGS_DIM = 768


def md5_checksum(filepath):
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def run_application():
    if __name__ == '__main__':
        threading.Thread(target=lambda: app.run(debug=False)).start()


class Task:
    def __init__(self, task_type, func, kwargs: dict[str, any]):
        task_id = hashlib.md5(str(time.time()).encode('utf-8')).hexdigest()
        self.data = {"id": task_id, "type": task_type, "status": "process", "content": {}}
        self.func = func
        self.kwargs = kwargs
        self.status_code = 200

    def set_result(self, content, status_code):
        if content is None:
            self.data["content"] = {}
            self.data["status"] = "abort"
            self.status_code = status_code
        else:
            self.data["status"] = "ready"
            self.data["content"] = content
            self.status_code = status_code


class MainApplication:
    def __init__(self):
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        self.model_audio = AudioTagging(checkpoint_path=None, device='cuda' if torch.cuda.is_available() else 'cpu')
        self.db = create_lance_table(VIDEO_EMBEDDINGS_TABLE, VIDEO_EMBEDDINGS_DIM)
        self.db = create_lance_table(AUDIO_EMBEDDINGS_TABLE, AUDIO_EMBEDDINGS_DIM)
        self.queue = Queue()
        self.cache = {}

    def start(self):
        while True:
            if self.queue.empty():
                continue
            task = self.queue.get_nowait()
            content, status_code = task.func(task.kwargs)
            task.set_result(content, status_code)
            self.cache[task.data["id"]] = task

    def add_task(self, task: Task):
        self.queue.put(task)
        self.cache[task.data["id"]] = task

    def index_in_db(self, temp_dir):
        db = get_embeddings_for_directory(temp_dir, self.vit, self.feature_extractor, self.model_audio)
        return db is not None

    def search_in_db(self, filename):
        dict_data = get_video_embeddings(filename,
                                         self.vit, self.feature_extractor, self.model_audio)
        table_video = self.db.open_table(VIDEO_EMBEDDINGS_TABLE)
        table_audio = self.db.open_table(AUDIO_EMBEDDINGS_TABLE)
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

        for batch in dict_data["audio"]:
            result = table_audio.search(batch, vector_column_name="vector").metric("cosine").limit(10).to_list()
            if result[0]["_distance"] < threshold_audio:
                if not percent_dict.get(result[0]["filename"]):
                    percent_dict[result[0]["filename"]] = 1
                else:
                    percent_dict[result[0]["filename"]] += 1
        for key, _ in percent_dict.items():
            percent_dict[key] = percent_dict[key] / (len(dict_data["video"] * 2))

        file_piracy = max(percent_dict.items(), key=operator.itemgetter(1))[0]  # add threshold for final result
        full_embedding_video = table_video.search().where(f"filename = {file_piracy}").to_list()
        full_embedding_video_vec = [x["vector"] for x in full_embedding_video]
        full_embedding_audio = table_audio.search().where(f"filename = {file_piracy}").to_list()
        full_embedding_audio_vec = [x["vector"] for x in full_embedding_audio]

        matrix = cosine_similarity(dict_data["video"], full_embedding_video_vec)
        matrix_audio = cosine_similarity(dict_data["audio"], full_embedding_audio_vec)
        matrix = matrix + matrix_audio

        return f"{make_plt_rows(matrix)} {make_plt_columns(matrix)}", file_piracy


application = MainApplication()


def index_in_db_wrapper(kwargs):
    result = application.index_in_db(kwargs.get("temp_dir").name)
    kwargs.get("temp_dir").cleanup()
    if not result:
        return {"error": "error while indexing video"}, 500
    return {"indexed": True}, 200


def search_in_db_wrapper(kwargs):
    result = application.index_in_db(kwargs.get("filename"))
    kwargs.get("temp_dir").cleanup()
    if not result:
        return {"error": "error while searching video"}, 500
    return {"intervals": result[0], "filename": result[1]}, 200


@app.route('/set_video_download', methods=['POST'])
def set_video_download():
    temp_dir = tempfile.TemporaryDirectory()
    download_url = request.json.get("download_url")
    filename = request.json.get("filename")
    hashsum_md5 = request.json.get("md5")
    purpose = request.json.get("purpose")
    response = requests.get(download_url, stream=True)

    if download_url == "":
        return jsonify({"error": "download url cant be empty"}), 422
    with open(os.path.join(temp_dir.name, filename), "wb") as handle:
        for data in tqdm(response.iter_content()):
            handle.write(data)

    hashsum = md5_checksum(os.path.join(temp_dir.name, filename))
    if hashsum != hashsum_md5:
        return jsonify({"error": "file integrity cant be verified"}), 500

    if purpose not in ["index", "val"]:
        return jsonify({"error": "purpose can be either index or val"}), 422

    if purpose == "index":
        task = Task("index_in_db", index_in_db_wrapper, {"temp_dir": temp_dir})
        application.add_task(task)
    if purpose == "val":
        task = Task("search_in_db", index_in_db_wrapper, {"temp_dir": temp_dir,
                                                          "filename": os.path.join(temp_dir.name, filename)})
        application.add_task(task)


@app.route('/task_status', methods=['POST'])
def task_status():
    task_id = request.json.get('task_id')
    if task_id is None or task_id == "" or task_id not in application.cache.keys():
        return jsonify({"result": "error", "error_str": "task_id cant be empty"}), 422
    return jsonify(application.cache[task_id].data), application.cache[task_id].status_code


run_application()
application.start()
