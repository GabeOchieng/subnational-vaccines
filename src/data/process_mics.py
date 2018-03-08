# -*- coding: utf-8 -*-
import json
import logging
import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
import savReaderWriter as srw
from tqdm import tqdm


@click.command()
@click.argument('data_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
def run_mics_process(data_dir, output_dir):
    data_dir = Path(data_dir)

    mics_countries = []
    mics_surveys = list((data_dir).iterdir())
    for f in tqdm(mics_surveys, desc='countries'):
        if f.name.startswith('.'):
            continue

        country = f.parts[-1].split("_")[0]
        mics_countries.append(country)

        sav_files = list(f.glob('*SPSS Datasets/*.sav'))
        for spss_file in tqdm(sav_files, desc=" > savs"):
            # if spss_file.stem in ['hh', 'hl']:
            country_dir = Path(output_dir)/country
            country_dir.mkdir(exist_ok=True)

            output_path = country_dir/f"{spss_file.stem}.csv"
            spss_to_csv(spss_file, output_path, w_col='hhweight')


def spss_to_csv(input_filepath, output_filepath, upsample_size=100000, w_col='Weight'):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    with srw.SavHeaderReader(input_filepath, ioUtf8=True) as header:
        metadata = header.all()
        value_replacements = metadata.valueLabels

        nan_replacements = dict()
        for k, v in metadata.missingValues.items():
            if v and 'values' in v:
                nan_values = dict()
                if isinstance(v['values'], list):
                    for nan_val in v['values']:
                        nan_values[nan_val] = np.nan
                else:
                    nan_values[v['values']] = np.nan

                nan_replacements[k] = nan_values

        questions = metadata.varLabels

    with srw.SavReader(input_filepath, ioUtf8=True) as reader:
        header = reader.header
        records = reader.all()

    df = pd.DataFrame(records, columns=header)
    df.replace(value_replacements, inplace=True)
    df.replace(nan_replacements, inplace=True)

    df.to_csv(output_filepath)

    questions_file = Path(output_filepath).with_suffix('.json')
    with open(questions_file, 'w') as qf:
        json.dump(questions, qf)

    # upsample the survey for a "representative" sample for analysis
    # if specified
    if upsample_size and upsample_size > 0 and w_col in df.columns:
        rng = np.random.RandomState(12345)

        smpl = rng.choice(df.index, upsample_size, p=df[w_col] / df[w_col].sum())
        df_resampled = df.loc[smpl, :]

        output_path = Path(output_filepath)
        df_resampled.to_csv(output_path.with_name(output_path.stem + '_upsampled.csv'))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    run_mics_process()
