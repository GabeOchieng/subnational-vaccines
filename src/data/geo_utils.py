import json
import os
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
from geopy.geocoders import (Nominatim, GeoNames, GoogleV3)
import pandas as pd


load_dotenv(find_dotenv())

ISO_CODES_PATH = Path(__file__).parents[2]/'data'/'external'/'iso-codes.csv'
GEOCACHE_PATH = Path(__file__).parents[2]/'data'/'interim'/'geo_query_cache.json'


class Geocache(object):
    def __init__(self, path, raise_on_error=True):
        self.path = Path(path).resolve()
        self.raise_on_error = raise_on_error

        if self.path.exists():
            with open(self.path, 'r') as f:
                self.cache = json.load(f)

        else:
            self.cache = dict()

    def _query_geocoder(self, geocoder, query):
        a2, a1, country = '', '', ''

        try:
            response = geocoder.geocode(query, exactly_one=True, language='en')

            if isinstance(geocoder, GoogleV3):
                components = response.raw['address_components']

                for c in components:
                    if 'administrative_area_level_2' in c['types']:
                        a2 = c['long_name']
                    elif 'administrative_area_level_1' in c['types']:
                        a1 = c['long_name']
                    elif 'country' in c['types']:
                        country = c['long_name']

            elif isinstance(geocoder, Nominatim):
                a2, a1, country = map(str.strip, response.raw['display_name'].split(','))
            else:
                raise ValueError("This type of Geocoder is not supported")

        except:
            if self.raise_on_error:
                raise

        return({'admin2': a2, 'admin1': a1, 'country': country})

    def _update(self, update_dict):
        """ Dict with keys that are the query and values that are the results."""
        self.cache.update(update_dict)

        with open(self.path, 'w') as f:
            json.dump(self.cache, f)

    def lookup(self, geocoder, query):
        result = self._query_geocoder(geocoder, query)
        self._update({query: result})
        return result

    def batch_lookup(self, geocoder, queries):
        results = [self._query_geocoder(geocoder, q) for q in queries]
        update_dict = dict(zip(queries, results))
        self._update(update_dict)
        return results


class CountryNames(object):
    def __init__(self, path):
        """ Loads isocode data from path. Data originally downloaded from:
            https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.csv
        """
        self.code_map = pd.read_csv(path).set_index('alpha-3')['name']

    def lookup(self, code):
        return self.code_map[code].values[0]

    def batch_lookup(self, codes):
        return self.code_map[codes].values


def _get_manual_remap():
    """ Some regions are not properly geocoded, so we map those explicitly here:
    """
    return {
        "Almaar  Afghanistan": "Almar  Afghanistan"
    }


def canonicalize_dataframe_geographies(df,
                                       country_col,
                                       admin1_col,
                                       admin2_col,
                                       iso_country=True,
                                       timeout=10,
                                       geocoder_class=GoogleV3,
                                       return_failures=True):
    """ Takes a dataframe and returns a canonicalized representation of the country, admin1
        and admin2.

        `timeout` is the numer of seconds to wait for a response (default 10)
        `geocoder_class` is the class of geocoder from the geopy package to use
        If `return_failures` the rows that fail will also be returned as blank; otherwise, they will raise an exception.
        If `iso_country` the country column is the iso code

        Returns a dataframe with the same index as input. Columns are country, admin1, and admin2 geocoded.
    """
    cn = CountryNames(ISO_CODES_PATH)

    # don't manipulate existing dataframe inplace
    df = df.copy()

    if iso_country:
        df[country_col] = cn.batch_lookup(df[country_col].values)

    gc = Geocache(GEOCACHE_PATH, raise_on_error=(not return_failures))

    queries = (df[admin2_col].fillna('').astype('str').str.replace("-", ' ') + " " +
               df[admin1_col].fillna('').astype('str').str.replace("-", ' ') + " " +
               df[country_col].fillna('').astype('str'))

    # some queries we need to update to get them coded :'(
    queries.replace(_get_manual_remap(), inplace=True)

    geocoder = geocoder_class(timeout=timeout, api_key=os.environ.get('GMAPS_KEY', ''))
    results = gc.batch_lookup(geocoder, queries)

    results = pd.DataFrame(index=df.index, data=results)
    failures = df.loc[(results == '').all(axis=1), :]

    return (results, failures) if return_failures else results
