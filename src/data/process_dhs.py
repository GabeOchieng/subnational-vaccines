import json
import logging
from pathlib import Path
import sys

import pandas as pd
from pandas.io.stata import StataReader
from tqdm import tqdm

def load_stata_file(filepath,
                    index_cols):
    """ Load data and metadata from Stata file"""
    data = pd.read_stata(filepath, convert_categoricals=False).set_index(index_cols)

    with StataReader(filepath) as reader:
        reader.value_labels()

        mapping = {col: reader.value_label_dict[t] for col, t in
                   zip(reader.varlist, reader.lbllist)
                   if t in reader.value_label_dict}

        data.replace(mapping, inplace=True)

        # convert the categorical variables into
        # the category type
        for c in data.columns:
            if c in mapping:
                data[c] = data[c].astype('category')

        # read the actual questions that were asked for reference
        questions = reader.variable_labels()

    return data, questions


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    dhs_raw_folder = Path(sys.argv[1])
    dhs_out_folder = Path(sys.argv[2])

    logger.info(f'Loading data from {dhs_raw_folder} and saving to {dhs_out_folder}...')
    for folder in tqdm(list(dhs_raw_folder.iterdir())):
        # process children's recodes and bith recodes for the most recent wave.
        # see https://dhsprogram.com/data/File-Types-and-Names.cfm
        if (
            (('KR7' in folder.name) or
             ('BR7' in folder.name))
            and folder.name.endswith('DT')
            ):
            # print(folder.name)
            survey_id = folder.stem[:-2]
            df, questions = load_stata_file(folder/f"{survey_id}FL.DTA", ['caseid'])

            # create dir tree if it doesn't exist
            (dhs_out_folder/survey_id).mkdir(exist_ok=True, parents=True)

            # write csv and questions
            df.to_csv(dhs_out_folder/survey_id/f"{survey_id}.csv")
            with open(dhs_out_folder/survey_id/f"{survey_id}.json", "w") as f:
                json.dump(questions, f)

