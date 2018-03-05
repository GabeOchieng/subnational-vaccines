# -*- coding: utf-8 -*-
import json
import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd
import savReaderWriter as srw


@click.command()
@click.argument('data_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
def run_mics_process(data_dir, output_dir):
    data_dir = Path(data_dir)

    mics_countries = []
    for f in (data_dir).iterdir():
        if f.name.startswith('.'):
            continue

        country = f.parts[-1].split("_")[0]
        mics_countries.append(country)

        for spss_file in f.glob('*SPSS Datasets/*.sav'):
            if spss_file.stem in ['hh', 'hl']:
                spss_to_csv(spss_file, f'test_here_{spss_file.stem}.csv')


def spss_to_csv(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    # load metadata
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
    # rng = np.random.RandomState(12345)

    # w_col = 'Weight' if 'Weight' in df.columns else 'weight'

    # smpl = rng.choice(df.index, 240000, p=df[w_col] / df[w_col].sum())
    # df_resampled = df.loc[smpl, :]

    # output_path = Path(output_filepath)
    # df_resampled.to_csv(output_path.with_name(output_path.stem + '_upsampled.csv'))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    run_mics_process()
