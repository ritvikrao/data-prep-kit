from charm4py import charm, ray

from data_processing_ray.runtime.ray import RayTransformLauncher
from data_processing.utils import ParamsUtils
import sys
import json
import pandas as pd

import os
import pyarrow as pa
import pyarrow.parquet as pq

from datasets import load_dataset

import uuid
from data_processing.utils import TransformUtils
from collections import defaultdict

from pathlib import Path
from ededup_transform_ray import EdedupRayTransformConfiguration
import pprint

def read_parquet_bulk(dir_path):
    data_dir = Path(dir_path)
    # Get the list of all Parquet files in the directory
    parquet_files = list(data_dir.glob('*.parquet'))
    # Check if the directory contains any Parquet files
    if not parquet_files:
        raise ValueError(f"No Parquet files found in directory: {dir_path}") 
    # Concatenate all Parquet files into a single DataFrame
    full_df = pd.concat(
        pd.read_parquet(parquet_file)
        for parquet_file in parquet_files
    ).reset_index(drop=True)
    
    return full_df

## Converts a subset of a Hugging Face dataset to a Parquet file, optionally mapping and renaming columns.
def hf_dataset_to_parquet(ds, skip, nrows, file_name, mapper=None, renamed_columns=[]):
    dst_ = ds.skip(skip).take(nrows)
    
    data_dict = defaultdict(list)

    dst = dst_.map(mapper)

    for data in dst:
        for k, v in data.items():
            data_dict[k].append(v)

    for old, new in renamed_columns:
        data_dict[new] = data_dict[old]
        del data_dict[old]

    table = pa.Table.from_pydict(data_dict)
    pq.write_table(table, file_name)

def row_mapper(row):
    return {
            'ext': TransformUtils.get_file_extension(row['path'])[1],
            'document_id': str(uuid.uuid4())
            }

def main(args):
    #Default parameters for computation
    worker_options = {"num_cpus": 0.8}
    common_config_params = {
        "run_locally": True,
        "runtime_worker_options": ParamsUtils.convert_to_ast(worker_options),
        "runtime_num_workers": 2,
    }

    DATASET_NAME='codeparrot/github-code'

    ds = load_dataset(DATASET_NAME, streaming=True, split="train", trust_remote_code=True)
    parquet_data_output = "sample_data/hf_2_parquet"
    total_files = 20
    rows_per_file = 20
    for num in range(total_files):
        file_name = os.path.join(
            f"{parquet_data_output}",
            f"data_{num}.parquet"
        )
        print (f"Writing {file_name}")
        hf_dataset_to_parquet(ds, 
                            1 * rows_per_file,
                            rows_per_file,
                            file_name=file_name,
                            mapper=row_mapper,
                            renamed_columns=[("code", "contents"),
                                            ("path", "title")])
    input_df=read_parquet_bulk(parquet_data_output)
    print("No of rows, No of columns",input_df.shape)
    print("Sample data \n ")
    input_df.head(1)

    input_folder = parquet_data_output # Output of previous stage is used as input.
    output_folder = "sample_data/ededup_out"

    local_conf = {
        "input_folder": input_folder,
        "output_folder": output_folder,
    }

    ededup_params = {
        # ededup parameters
        "ededup_hash_cpu": 0.5,
        "ededup_num_hashes": 2,
        "ededup_doc_column": "contents",
        "data_local_config": ParamsUtils.convert_to_ast(local_conf)
    }
    params = common_config_params | ededup_params
    sys.argv = ParamsUtils.dict_to_req(d=params)
    ededup_launcher = RayTransformLauncher(EdedupRayTransformConfiguration()) 
    ededup_launcher.launch()
    def read_metadata(path):
        with open(path, 'r') as file:
            metadata = json.load(file)
            pprint.pp(metadata)
    read_metadata(f"{output_folder}/metadata.json")

    exit()

charm.start(main)