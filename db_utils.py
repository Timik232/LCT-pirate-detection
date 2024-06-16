import lancedb
import pyarrow as pa
import os


def create_lance_db() -> lancedb.DBConnection:
    """
    create vector table
    :param table_name: name of the table
    :param vector_dim: dimension of the vector
    :return: database connection
    """

    db_path = "data/"
    if not os.path.exists(db_path):
        os.makedirs(db_path)
    db = lancedb.connect(db_path)

    return db


def append_vectors_to_table(db: lancedb.DBConnection, table_name: str, vector_video, vector_audio, segments, filenames: list,
                           video_dim, audio_dim) -> None:
    """
    append vector of embeddings to table
    :param db: lancedb connection
    :param table_name: name of the table
    :param vector_video:
    :param vector_audio:
    :param segments:
    :param filenames:
    :param video_dim:
    :param audio_dim:
    :return:
    """

    schema = pa.schema([
        ('vector_video', pa.list_(pa.float64(), video_dim)),
        ('vector_audio', pa.list_(pa.float64(), audio_dim)),
        ('segment_time', pa.int64()),
        ('filename', pa.string())
    ])

    print(f"Start append f{vector_video.shape}")
    if not (len(vector_video) == len(segments) == len(filenames) == len(vector_audio)):
        raise ValueError(
            f"Length mismatch: vector ({len(vector_video)}), segments ({len(segments)}), filenames ({len(filenames)})"
            f"vector audio ({len(vector_audio)})")

    if table_name not in db.table_names():
        _ = db.create_table(table_name, schema=schema)
    table = db.open_table(table_name)

    # Create PyArrow arrays for all data at once
    vector_array = pa.array(vector_video.tolist())
    vector_audio_array = pa.array(vector_audio.tolist())
    segment_time_array = pa.array(segments)
    filename_array = pa.array(filenames)

    # Create a single PyArrow Table from all arrays
    t = pa.Table.from_arrays([vector_array, vector_audio_array, segment_time_array, filename_array], schema=table.schema)

    # Add the entire table in one operation
    table.add(t)
