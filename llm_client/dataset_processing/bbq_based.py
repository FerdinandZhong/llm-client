import pandas as pd


def load_file_as_df(file_path):
    raw_df = pd.read_json(path_or_buf=file_path, lines=True)
