import lancedb
import pyarrow as pa
import os


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
    print(f"Start append f{vector.shape}")
    if not (len(vector) == len(segments) == len(filenames)):
        raise ValueError(
            f"Length mismatch: vector ({len(vector)}), segments ({len(segments)}), filenames ({len(filenames)})")

    table = db.open_table(table_name)

    # Create PyArrow arrays for all data at once
    vector_array = pa.array(vector.tolist())
    segment_time_array = pa.array(segments)
    filename_array = pa.array(filenames)

    # Create a single PyArrow Table from all arrays
    t = pa.Table.from_arrays([vector_array, segment_time_array, filename_array], schema=table.schema)

    # Add the entire table in one operation
    table.add(t)
