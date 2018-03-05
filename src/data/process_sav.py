# -*- coding: utf-8 -*-
import contextlib
import json
import os
import logging
from pathlib import Path
import sys

import click
from dotenv import find_dotenv, load_dotenv
import numpy as np
import pandas as pd
import savReaderWriter as srw


@contextlib.contextmanager
def configure_savReaderWriter():
    logger = logging.getLogger(__name__)
    logger.info(f"{os.environ.get('DYLD_LIBRARY_PATH', '')}")
    yield

    # if sys.platform.lower() != 'darwin':
    #     logger.warn("Configuration for savReaderWriter is only implemented for MacOS (darwin). "
    #                 "Other OSes may work without additional configuration per the "
    #                 "savReaderWriter documentation pages. Your platform is {}".format(sys.platform))

    #     return

    # _old_environ = dict(os.environ)

    # os.environ['LC_ALL'] = 'en_US.UTF-8'

    # spss_new_path = Path(srw.__file__).parent / "spssio" / "macos"
    # assert spss_new_path.exists()

    # dyld_path = os.environ.get('DYLD_LIBRARY_PATH', '')
    # os.environ['DYLD_LIBRARY_PATH'] = os.pathsep.join([str(spss_new_path), dyld_path])

    # try:
    #     yield
    # finally:
    #     os.environ.clear()
    #     os.environ.update(_old_environ)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def spss_to_csv(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # configure savReaderWriter
    with configure_savReaderWriter():
        # load metadata

        with srw.SavHeaderReader(input_filepath, ioUtf8=True) as header:
            metadata = header.all()
            value_replacements = metadata.valueLabels

            nan_replacements = dict()
            for k, v in metadata.missingValues.items():
                if v and 'values' in v:
                    nan_values = dict()
                    if isinstance(v['values'], list):
                        for nan_val in v:
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


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    spss_to_csv()
