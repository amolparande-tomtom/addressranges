import os
import re
import numpy as np
import pandas as pd
import psycopg2
import typing
from shapely.geometry import LineString, Point
from math import radians, sin, cos, sqrt, atan2
from typing import Callable
from pandas.core.frame import DataFrame
import datetime
from shapely import wkb
import geopandas as gpd
from shapely.wkt import loads
from map_content.utils import utils


def csvFileWriter(pandasDataFrame, filename, outputpath):
    if not os.path.exists(outputpath + filename):
        pandasDataFrame.to_csv(outputpath + filename, mode='w', index=False, encoding="utf-8")

    else:
        pandasDataFrame.to_csv(outputpath + filename, mode='a', header=False, index=False, encoding="utf-8")


def oldinterpolationTypeHandalling(get_hnr_df_DF: DataFrame, correct_hnr_array) -> DataFrame:
    for index, row in get_hnr_df_DF.iterrows():
        interpolation = row['interpolation']
        intermediates = row['intermediates']
        min_hsn_numeric = row['min_hsn_numeric']
        max_hsn_numeric = row['max_hsn_numeric']
        hnr_numeric_mixed_array = row['hnr_numeric_mixed_array']

        # Check if 'interpolation' is ['irregular', 'odd', 'numeric_mixed', 'even'] is not null
        if (interpolation in ['irregular', 'odd', 'numeric_mixed', 'even']) and pd.notnull(intermediates).any():
            # Create the 'hnr_array' by adding 'min_hsn_numeric' and 'max_hsn_numeric' to 'intermediates'
            hnr_array = [min_hsn_numeric, max_hsn_numeric] + intermediates
            get_hnr_df_DF.at[index, 'hnr_array'] = hnr_array
        else:
            # Copy the records from the existing 'hnr_numeric_mixed_array' column
            get_hnr_df_DF.at[index, 'hnr_array'] = hnr_numeric_mixed_array

    get_hnr_df_DF['hnr_ranges'] = get_hnr_df_DF['hnr_array']

    # Apply the correction function to the "hnr_array" column
    get_hnr_df_DF['hnr_array'] = get_hnr_df_DF['hnr_array'].apply(correct_hnr_array)

    # Remove square brackets and convert array to string using lambda function
    get_hnr_df_DF['street'] = get_hnr_df_DF['street'].apply(lambda x: x[0] if isinstance(x, list) else x)

    selectedColumnsGetHnr_DF = get_hnr_df_DF[
        ['osm_id', 'place_name', 'street', 'way', 'min_hsn', 'max_hsn', 'hnr_array', 'hnr_ranges',
         'hnr_numeric_mixed_array', 'PointLocation', 'interpolation', 'intermediates', 'place_value']]

    # Explode functionality for Array
    df_exploded = selectedColumnsGetHnr_DF.explode('hnr_array')
    df_exploded['hnr_Number'] = df_exploded['hnr_array']

    df_exploded.reset_index(drop=True, inplace=True)
    return df_exploded

def interpolationTypeHandalling(get_hnr_df_DF: DataFrame, correct_hnr_array) -> DataFrame:
    # Ensure 'intermediates' column has lists (convert NaNs to empty lists)
    get_hnr_df_DF['intermediates'] = get_hnr_df_DF['intermediates'].apply(lambda x: x if isinstance(x, list) else [])

    for index, row in get_hnr_df_DF.iterrows():
        interpolation = row['interpolation']
        intermediates = row['intermediates']
        min_hsn_numeric = row['min_hsn_numeric']
        max_hsn_numeric = row['max_hsn_numeric']
        hnr_numeric_mixed_array = row['hnr_numeric_mixed_array']

        # Check if 'interpolation' is ['irregular', 'odd', 'numeric_mixed', 'even'] is not null
        if (interpolation in ['irregular', 'odd', 'numeric_mixed', 'even']) and pd.notnull(intermediates).any():
            # Create the 'hnr_array' by adding 'min_hsn_numeric' and 'max_hsn_numeric' to 'intermediates'
            hnr_array = [min_hsn_numeric, max_hsn_numeric] + intermediates
            get_hnr_df_DF.at[index, 'hnr_array'] = hnr_array
        else:
            # Copy the records from the existing 'hnr_numeric_mixed_array' column
            get_hnr_df_DF.at[index, 'hnr_array'] = hnr_numeric_mixed_array

    get_hnr_df_DF['hnr_ranges'] = get_hnr_df_DF['hnr_array']

    # Ensure 'hnr_array' column has lists (convert NaNs to empty lists)
    get_hnr_df_DF['hnr_array'] = get_hnr_df_DF['hnr_array'].apply(lambda x: x if isinstance(x, list) else [])

    # Apply the correction function to the "hnr_array" column
    get_hnr_df_DF['hnr_array'] = get_hnr_df_DF['hnr_array'].apply(correct_hnr_array)

    # Remove square brackets and convert array to string using lambda function
    get_hnr_df_DF['street'] = get_hnr_df_DF['street'].apply(lambda x: x[0] if isinstance(x, list) else x)

    selectedColumnsGetHnr_DF = get_hnr_df_DF[
        ['osm_id', 'place_name', 'street', 'way', 'min_hsn', 'max_hsn', 'hnr_array', 'hnr_ranges',
         'hnr_numeric_mixed_array', 'PointLocation', 'interpolation', 'intermediates', 'place_value']]

    # Explode functionality for Array
    df_exploded = selectedColumnsGetHnr_DF.explode('hnr_array')
    df_exploded['hnr_Number'] = df_exploded['hnr_array']

    df_exploded.reset_index(drop=True, inplace=True)
    return df_exploded


# from map_content.utils.openmap import get_alphabetic_hnr_df, get_numeric_hnr_df
def truncate(n: float, decimals: int = 0) -> int:
    """Simple function which truncates an incoming float value (n) and returns its integer value
    based on its multiplier value, later divided by the same multiplier value.
    :param n: Input numerical value, float type.
    :type n: flaot
    :param decimals: Input integer which defines the number of decimal point to return, defaults to 0
    :type decimals: int, optional
    :return:Output integer value after the truncate process of the input value (n).
    :rtype: int
    """

    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier


def haversine_distance(lt1: float,
                       ln1: float,
                       lt2: float,
                       ln2: float) -> float:
    """Function which calculates the distance in meters between two XY coodinates. The function requires
    the latitude and longitude for each point. At the end of the function, we truncate the output distance
    value with a maximun of 4 decimal points.
    :param lt1: Latitude value of first coordinate point.
    :type lt1: float
    :param ln1: Longitude value of first coordinate point.
    :type ln1: float
    :param lt2: Latitude value of second coordinate point.
    :type lt2: float
    :param ln2: Longitude value of second coordinate point.
    :type ln2: float
    :return: Distance between point in meters.
    :rtype: float
    """

    R = 6373.0  # approximate radius of earth in km
    lat1 = radians(lt1)
    lon1 = radians(ln1)
    lat2 = radians(lt2)
    lon2 = radians(ln2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = truncate(R * c, 4) * 1000
    return distance


def StartAndEndHNRSame(dexploded_Df):
    """
    Starting and Ending House Number Same
    :return:
    """
    houseNumberError = pd.DataFrame()
    # Check conditions and append output records
    for index, row in dexploded_Df.iterrows():
        if row['min_hsn'] == row['max_hsn']:
            row['Error'] = 'Array Issue'
            houseNumberError = houseNumberError.append(row, ignore_index=True)
    # # Reset index of Error DataFrame
    # houseNumberError.reset_index(drop=True, inplace=True)
    # Remove multiple columns
    arrayColumnRemove = ['way', 'min_hsn', 'max_hsn', 'hnr_array', 'Error']
    if houseNumberError.empty:
        return pd.DataFrame()
    else:
        houseNumberArrayRemove = houseNumberError.drop(arrayColumnRemove, axis=1)
        # Array Reorder the columns
        array_columns = ['osm_id', 'hnr_Number', 'street', 'place_name', 'hnr_numeric_mixed_array', 'PointLocation',
                         'place_value']
        houseNumberArray = houseNumberArrayRemove[array_columns].copy()
        # Remove duplicates and keep the first occurrence
        houseNumberArray.drop_duplicates(subset=['osm_id'], keep='first', inplace=True)
        return houseNumberArray


# Function to correct the "hnr_array" column
def Old_correct_hnr_array(arr):
    corrected_arr = []
    for item in arr:
        if ';' in item:
            corrected_arr.extend(item.split(';'))
        else:
            corrected_arr.append(item)
    return corrected_arr


def old_v1correct_hnr_array(arr):
    if arr is None:
        return []
    corrected_arr = []
    for item in arr:
        if ';' in item:
            corrected_arr.extend(item.split(';'))
        else:
            corrected_arr.append(item)
    return corrected_arr


def old_v2correct_hnr_array(arr):
    if arr is None or isinstance(arr, float):  # Check for None or float
        return []
    corrected_arr = []
    for item in arr:
        if isinstance(item, int):
            corrected_arr.append(item)
        elif ';' in item:
            corrected_arr.extend(item.split(';'))
        else:
            corrected_arr.append(item)
    return corrected_arr


def oldcorrect_hnr_array(arr):
    if pd.isna(arr):  # Check for NaN ('NAType')
        return []

    if isinstance(arr, float):  # Check for None or float
        return []

    corrected_arr = []
    for item in arr:
        if isinstance(item, int):
            corrected_arr.append(item)
        elif pd.notna(item):  # Check for non-NaN items
            if ';' in item:
                corrected_arr.extend(item.split(';'))
            else:
                corrected_arr.append(item)
    return corrected_arr

def correct_hnr_array(arr):
    if isinstance(arr, float):  # Check for None or float
        return []

    if isinstance(arr, (list, np.ndarray)):
        corrected_arr = []
        for item in arr:
            if pd.notna(item):  # Check for non-NaN items
                if isinstance(item, int):
                    corrected_arr.append(item)
                elif isinstance(item, str) and ';' in item:
                    corrected_arr.extend(item.split(';'))
                else:
                    corrected_arr.append(item)
        return corrected_arr

    return []


# Function to calculate the center point of a LineString
def calculate_center_point(geometry):
    line = LineString(geometry)
    return line.centroid


def parse_hnr_tags(df: pd.DataFrame) -> pd.DataFrame:
    """Parses the OSM tags to extract house number range info from way

    :param df: DataFrame from OSM result
    :type df: pd.DataFrame
    :return: DataFrame with added columns for hnr components
    :rtype: pd.DataFrame
    """

    # Create copy of
    df_copy = df.copy()

    # Split tags into list for both end point nodes and network
    df_copy["tags_list"] = df_copy.tags.str.split('",')
    df_copy["tags_list"] = df_copy.tags_list.apply(
        lambda x: [tag.split("=>") for tag in x]
    )

    df_copy["tags_network_list"] = df_copy.tags_network.str.split('",')
    df_copy["tags_network_list"] = df_copy.tags_network_list.apply(
        lambda x: [tag.split("=>") for tag in x]
    )

    # Parse components
    df_copy["constant"] = df_copy["tags_network_list"].apply(
        lambda x: [tag[1].replace('"', "").strip() for tag in x if "constant" in tag[0]]
    )
    df_copy["constant"] = df_copy["constant"].apply(lambda x: x[0] if x else None)

    df_copy["interpolation_value"] = df_copy["tags_network_list"].apply(
        lambda x: [
            tag[1].replace('"', "").strip() for tag in x if "interpolation" in tag[0]
        ]
    )
    df_copy["interpolation_value"] = df_copy["interpolation_value"].apply(
        lambda x: x if x else None
    )

    # Replace interpolation
    df_copy["interpolation"] = df_copy["interpolation"].fillna(
        df_copy["interpolation_value"]
    )

    df_copy["intermediates"] = df_copy["tags_network_list"].apply(
        lambda x: [
            tag[1].replace('"', "").strip() for tag in x if "intermediate" in tag[0]
        ]
    )
    df_copy["intermediates"] = df_copy.intermediates.apply(
        lambda x: [hsn for value in x for hsn in value.split(",")] if x else None
    )

    df_copy["street"] = df_copy.tags_list.apply(
        lambda x: [tag[1].replace('"', "").strip() for tag in x if "street" in tag[0]]
    )
    df_copy["street"] = df_copy.street.apply(lambda x: x if x else None)

    return df_copy.drop(columns=["tags_list"])


def get_numeric_house_number_column(x: pd.Series) -> typing.List[str]:
    """Extracts the numeric component of a house number

    :param x: column containing house numbers
    :type x: pd.Series
    :return: list containing the parsed housenumbers
    :rtype: typing.List[str]
    """
    numeric_component = x.str.extract("(\d+)[^\d]*(\d+)?", expand=False).fillna("")
    return [" ".join(j).strip() for j in numeric_component.values.tolist()]


def preprocess_hnr_hsn(hnr_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the house numbers for a set of address ranges
    responses

    :param hnr_df: DataFrame, result of openmap_hnr_lookup
    :type hnr_df: pd.DataFrame
    :return: DataFrame with processed house numbers
    :rtype: pd.DataFrame
    """

    country_hnr_df_lookup = hnr_df.copy()

    # Split into list and get smallest and biggest HSN in the range
    country_hnr_df_lookup["min_hsn_numeric"] = get_numeric_house_number_column(
        country_hnr_df_lookup["min_hsn"]
    )

    country_hnr_df_lookup["max_hsn_numeric"] = get_numeric_house_number_column(
        country_hnr_df_lookup["max_hsn"]
    )

    return country_hnr_df_lookup


def compute_alphabetic_hnr(x: pd.Series) -> typing.List[str]:
    """Generates a list of alphabetic address range in the Orbis ecosystem

    :param x: Series where the interpolation value is 'alphabetic'
    :type x: pd.Series
    :return: list containing alphabetic housenumber ranges
    :rtype: typing.List[str]
    """

    # If endpoints are equal, return list with one value
    if x["min_hsn"] == x["max_hsn"]:
        return [x["min_hsn"]]

    # If cannot convert to characters, return endpoints
    if (not isinstance(x["first_char"], str)) or (not isinstance(x["last_char"], str)):
        return [x["min_hsn"], x["max_hsn"]]

    # Iterate over chars in order
    variable_part = [
        chr(i)
        for i in range(
            min([ord(x["first_char"].lower()), ord(x["last_char"].lower())]),
            max([ord(x["first_char"].lower()), ord(x["last_char"].lower())]) + 1,
        )
    ]

    # Add constant value
    hnr_array = [x["min_hsn_numeric"] + char for char in variable_part]

    return hnr_array


def numeric_mixed_array(x: pd.Series) -> typing.List[str]:
    """Parses numeric mixed array according to different posibilities

    :param x:  Series containing constant value, and parseable numeric mixed arrays
    :type x: pd.Series
    :return: list containing the parsed address range
    :rtype: typing.List[str]
    """

    # No constant value, hence iteration over endpoints
    if x["constant"] == "":
        return x["hnr_array"]

    # Constant value separated by '-'
    if "-" in x["constant"]:
        return [x["constant"] + hnr for hnr in x["hnr_array"]]

    # Numeric constant value that it's possibly not separated by a dash
    if len(x["constant"]) >= 2 and x["min_hsn_variable"] == "":
        min_hsn_variable = 1
        max_hsn_variable = int(re.search("([0-9]+)", x["max_hsn_variable"]).group())
        variable_array = list(range(min_hsn_variable, max_hsn_variable + 1, 1))
        hnr_mixed_array = [x["min_hsn"]] + [
            x["constant"] + "-" + str(hnr) for hnr in variable_array
        ]
        return hnr_mixed_array

    # Last case, take endpoints and add integers between them
    min_hsn_variable = int(re.search("([0-9]+)", x["min_hsn"]).group())
    max_hsn_variable = int(re.search("([0-9]+)", x["max_hsn"]).group())
    variable_array = list(range(min_hsn_variable + 1, max_hsn_variable + 1, 1))
    hnr_mixed_array = [x["min_hsn"]] + variable_array + [x["max_hsn"]]

    return hnr_mixed_array


def get_alphabetic_hnr_df(hnr_df: pd.DataFrame) -> pd.DataFrame:
    """Produces the housenumber array for an alphabetic address range

    :param hnr_df: DataFrame, result of openmap_hnr_lookup, after preprocessed
    :type hnr_df: pd.DataFrame
    :return: DataFrame containing alphabetic variance housenumber ranges
    :rtype: pd.DataFrame
    """
    country_alpha_hnr_df = hnr_df.copy()

    # Return same input if DataFrame is empty
    if country_alpha_hnr_df.shape[0] == 0:
        return pd.DataFrame(columns=hnr_df.columns.tolist() + ["hnr_array"])

    # Compute alphabetic hnr
    country_alpha_hnr_df["first_char"] = (
        country_alpha_hnr_df["min_hsn"]
            .str.replace("[^a-zA-Z]", "", regex=True)
            .str[0]
            .replace({"": "a"})
    )

    country_alpha_hnr_df["last_char"] = (
        country_alpha_hnr_df["max_hsn"]
            .str.replace("[^a-zA-Z]", "", regex=True)
            .str[0]
            .replace({"": "a"})
    )

    country_alpha_hnr_df["hnr_array"] = country_alpha_hnr_df.apply(
        compute_alphabetic_hnr, axis=1
    )

    return country_alpha_hnr_df


def old_get_numeric_hnr_df(hnr_df: pd.DataFrame) -> pd.DataFrame:
    """Produces the housenumber array for a numeric address range. These are
    address ranges with variance: 'even', 'odd', 'numeric_mixed', 'irregular'

    :param hnr_df: DataFrame, result of openmap_hnr_lookup, after preprocessed
    :type hnr_df: pd.DataFrame
    :return: DataFrame containing numeric variance housenumber ranges
    :rtype: pd.DataFrame
    """
    country_hnr_df_lookup = hnr_df.copy()

    # Return emtpy DataFrame with necessary columns if input is empty
    if country_hnr_df_lookup.shape[0] == 0:
        return pd.DataFrame(columns=hnr_df.columns.tolist() + ["hnr_array"])

    country_hnr_df_lookup["min_hsn_numeric"] = (
        country_hnr_df_lookup["min_hsn_numeric"]
            .apply(lambda x: min(x.split(" ")))
            .astype(int)
    )

    country_hnr_df_lookup["max_hsn_numeric"] = (
        country_hnr_df_lookup["max_hsn_numeric"]
            .apply(lambda x: max(x.split(" ")))
            .astype(int)
    )

    # Recompute lowest and max depending on how the info was captured
    country_hnr_df_lookup["min_hsn_hnr"] = country_hnr_df_lookup[
        ["min_hsn_numeric", "max_hsn_numeric"]
    ].min(axis=1)
    country_hnr_df_lookup["max_hsn_hnr"] = country_hnr_df_lookup[
        ["min_hsn_numeric", "max_hsn_numeric"]
    ].max(axis=1)

    # Convert to odd number or even number depending on interpolation
    country_hnr_df_lookup["min_hsn_hnr"] = np.where(
        country_hnr_df_lookup["interpolation"] == "even",
        country_hnr_df_lookup["min_hsn_hnr"] // 2 * 2,
        np.where(
            country_hnr_df_lookup["interpolation"] == "odd",
            country_hnr_df_lookup["min_hsn_hnr"] // 2 * 2 + 1,
            country_hnr_df_lookup["min_hsn_hnr"],
        ),
    )

    country_hnr_df_lookup["max_hsn_hnr"] = np.where(
        country_hnr_df_lookup["interpolation"] == "even",
        country_hnr_df_lookup["max_hsn_hnr"] // 2 * 2,
        np.where(
            country_hnr_df_lookup["interpolation"] == "odd",
            country_hnr_df_lookup["max_hsn_hnr"] // 2 * 2 + 1,
            country_hnr_df_lookup["max_hsn_hnr"],
        ),
    )

    # Compute cadency and fill null values for constant
    country_hnr_df_lookup["cadency"] = np.where(
        ~country_hnr_df_lookup["interpolation"].isin(["even", "odd"]), 1, 2
    )
    country_hnr_df_lookup["constant"] = country_hnr_df_lookup["constant"].fillna("")

    # HNR Array for even and odd cases
    country_hnr_df_lookup["hnr_array"] = country_hnr_df_lookup.apply(
        lambda x: list(
            np.arange(x["min_hsn_hnr"], x["max_hsn_hnr"] + x["cadency"], x["cadency"])
        ),
        axis=1,
    )
    country_hnr_df_lookup["hnr_array"] = country_hnr_df_lookup["hnr_array"].apply(
        lambda x: [str(j) for j in x]
    )
    country_hnr_df_lookup["hnr_array"] = country_hnr_df_lookup.apply(
        lambda x: x["hnr_array"] + [str(j) for j in x["intermediates"]]
        if x["intermediates"]
        else x["hnr_array"],
        axis=1,
    )

    # Compute numeric mixed array
    country_hnr_df_lookup["min_hsn_variable"] = country_hnr_df_lookup.apply(
        lambda x: x["min_hsn"].replace(x["constant"], ""), axis=1
    )
    country_hnr_df_lookup["max_hsn_variable"] = country_hnr_df_lookup.apply(
        lambda x: x["max_hsn"].replace(x["constant"], ""), axis=1
    )
    country_hnr_df_lookup["hnr_numeric_mixed_array"] = country_hnr_df_lookup.apply(
        lambda x: numeric_mixed_array(x)
        if x["interpolation"] == "numeric_mixed"
        else x["hnr_array"],
        axis=1,
    )

    # Determine final array depending on the type of interpolation
    country_hnr_df_lookup["hnr_array"] = np.where(
        country_hnr_df_lookup["interpolation"] == "numeric_mixed",
        country_hnr_df_lookup["hnr_numeric_mixed_array"],
        country_hnr_df_lookup["hnr_array"],
    )

    country_hnr_df_lookup["hnr_array"] = country_hnr_df_lookup.hnr_array.apply(
        lambda x: [str(j) for j in x]
    )

    return country_hnr_df_lookup

def get_numeric_hnr_df(hnr_df: pd.DataFrame) -> pd.DataFrame:
    """Produces the housenumber array for a numeric address range. These are
    address ranges with variance: 'even', 'odd', 'numeric_mixed', 'irregular'

    :param hnr_df: DataFrame, result of openmap_hnr_lookup, after preprocessed
    :type hnr_df: pd.DataFrame
    :return: DataFrame containing numeric variance housenumber ranges
    :rtype: pd.DataFrame
    """
    country_hnr_df_lookup = hnr_df.copy()

    # Return empty DataFrame with necessary columns if input is empty
    if country_hnr_df_lookup.shape[0] == 0:
        return pd.DataFrame(columns=hnr_df.columns.tolist() + ["hnr_array"])

    # Convert numeric columns to integers after handling missing values
    country_hnr_df_lookup["min_hsn_numeric"] = (
        country_hnr_df_lookup["min_hsn_numeric"]
        .apply(lambda x: min(x.split(" ")) if x else np.nan)
        .astype(float)
        .astype(pd.Int32Dtype())
    )

    country_hnr_df_lookup["max_hsn_numeric"] = (
        country_hnr_df_lookup["max_hsn_numeric"]
        .apply(lambda x: max(x.split(" ")) if x else np.nan)
        .astype(float)
        .astype(pd.Int32Dtype())
    )

    # Recompute lowest and max depending on how the info was captured
    country_hnr_df_lookup["min_hsn_hnr"] = country_hnr_df_lookup[
        ["min_hsn_numeric", "max_hsn_numeric"]
    ].min(axis=1)
    country_hnr_df_lookup["max_hsn_hnr"] = country_hnr_df_lookup[
        ["min_hsn_numeric", "max_hsn_numeric"]
    ].max(axis=1)

    # Convert to odd number or even number depending on interpolation
    country_hnr_df_lookup["min_hsn_hnr"] = np.where(
        country_hnr_df_lookup["interpolation"] == "even",
        country_hnr_df_lookup["min_hsn_hnr"] // 2 * 2,
        np.where(
            country_hnr_df_lookup["interpolation"] == "odd",
            country_hnr_df_lookup["min_hsn_hnr"] // 2 * 2 + 1,
            country_hnr_df_lookup["min_hsn_hnr"],
        ),
    )

    country_hnr_df_lookup["max_hsn_hnr"] = np.where(
        country_hnr_df_lookup["interpolation"] == "even",
        country_hnr_df_lookup["max_hsn_hnr"] // 2 * 2,
        np.where(
            country_hnr_df_lookup["interpolation"] == "odd",
            country_hnr_df_lookup["max_hsn_hnr"] // 2 * 2 + 1,
            country_hnr_df_lookup["max_hsn_hnr"],
        ),
    )

    # Compute cadency and fill null values for constant
    country_hnr_df_lookup["cadency"] = np.where(
        ~country_hnr_df_lookup["interpolation"].isin(["even", "odd"]), 1, 2
    )
    country_hnr_df_lookup["constant"] = country_hnr_df_lookup["constant"].fillna("")

    # HNR Array for even and odd cases
    country_hnr_df_lookup["hnr_array"] = country_hnr_df_lookup.apply(
        lambda x: list(
            np.arange(x["min_hsn_hnr"], x["max_hsn_hnr"] + x["cadency"], x["cadency"])
        ),
        axis=1,
    )
    country_hnr_df_lookup["hnr_array"] = country_hnr_df_lookup["hnr_array"].apply(
        lambda x: [str(j) for j in x]
    )
    country_hnr_df_lookup["hnr_array"] = country_hnr_df_lookup.apply(
        lambda x: x["hnr_array"] + [str(j) for j in x["intermediates"]]
        if x["intermediates"]
        else x["hnr_array"],
        axis=1,
    )

    # Compute numeric mixed array
    country_hnr_df_lookup["min_hsn_variable"] = country_hnr_df_lookup.apply(
        lambda x: x["min_hsn"].replace(x["constant"], ""), axis=1
    )
    country_hnr_df_lookup["max_hsn_variable"] = country_hnr_df_lookup.apply(
        lambda x: x["max_hsn"].replace(x["constant"], ""), axis=1
    )
    country_hnr_df_lookup["hnr_numeric_mixed_array"] = country_hnr_df_lookup.apply(
        lambda x: numeric_mixed_array(x)
        if x["interpolation"] == "numeric_mixed"
        else x["hnr_array"],
        axis=1,
    )

    # Determine final array depending on the type of interpolation
    country_hnr_df_lookup["hnr_array"] = np.where(
        country_hnr_df_lookup["interpolation"] == "numeric_mixed",
        country_hnr_df_lookup["hnr_numeric_mixed_array"],
        country_hnr_df_lookup["hnr_array"],
    )

    country_hnr_df_lookup["hnr_array"] = country_hnr_df_lookup.hnr_array.apply(
        lambda x: [str(j) for j in x]
    )

    return country_hnr_df_lookup

def get_hnr_df(hnr_df: pd.DataFrame) -> pd.DataFrame:
    """Produces the housenumber array, first separating into alphabetic and
    numeric interpolation, and then concat them to produce the address range

    :param hnr_df: DataFrame, result of openmap_hnr_lookup, after preprocessed
    :type hnr_df: pd.DataFrame
    :return: DataFrame containing house number range
    :rtype: pd.DataFrame
    """

    country_hnr_df_lookup = hnr_df.copy()

    # Split into alphabetic and numeric HNR and compute array
    country_alpha_hnr_df = country_hnr_df_lookup.loc[
        country_hnr_df_lookup["interpolation"] == "alphabetic"
        ].reset_index(drop=True)

    country_numeric_hnr_df = country_hnr_df_lookup.loc[
        country_hnr_df_lookup["interpolation"] != "alphabetic"
        ].reset_index(drop=True)

    country_alpha_hnr_df = get_alphabetic_hnr_df(country_alpha_hnr_df)
    country_numeric_hnr_df = get_numeric_hnr_df(country_numeric_hnr_df)

    # Join both dataframes
    country_hnr_df_lookup = pd.concat(
        [country_alpha_hnr_df, country_numeric_hnr_df], ignore_index=True
    )

    return country_hnr_df_lookup


def find_openmap_schema(
        country: str, latest: bool or None = True, credentials: dict or None = None
) -> pd.DataFrame:
    """Gets the schema in the Openmap's 3G database to make a spatial query

    :param country: country code in ISO2 or ISO3
    :type country: str
    :param latest: boolean or None indicating whether to return the latest version of OSM product, defaults to True
    :type latest: bool or None, optional
    :param credentials: dictionary containing the credentials for a connection
    :type credentials: dict or None, optional
    :return: DataFrame with the relevant country schemas
    :rtype: pd.DataFrame
    """

    # Standarize country input to ISO3
    # country_iso3 = country_converter.convert(country, to="ISO3")
    country_iso3 = 'gbr'
    # Initialize connection if not passed as parameter
    if credentials is None:
        conn = postgres_db_connection()
    else:
        conn = psycopg2.connect(**credentials)

    # Query all schemas in Openmap database
    schemas_df = pd.read_sql("SELECT nspname FROM pg_catalog.pg_namespace", conn)
    conn.close()

    # Filter relevant country
    country_schemas = schemas_df.loc[
        schemas_df.nspname.str.contains("_" + country_iso3, case=False)
    ].reset_index(drop=True)

    country_schemas["date"] = country_schemas.nspname.str.extract("([0-9]+)")
    country_schemas["schema"] = country_schemas.nspname.str.replace(
        "_[0-9]+.*", "", regex=True
    )
    country_schemas["is_latest"] = (
            country_schemas.date == country_schemas.groupby("schema").date.max()[0]
    )
    country_schemas["country"] = country

    # Return schemas
    if latest:
        return country_schemas.loc[country_schemas.is_latest == latest].reset_index(
            drop=True
        )
    else:
        return country_schemas


# con = psycopg2.connect(
#     host="10.137.173.68",
#     port="5432",
#     database="ggg",
#     user="ggg",
#     password="ok"
# )

def postgres_db_connection():
    """
    :param db_url: Postgres Server
    :return: DB Connection
    """
    try:
        con = psycopg2.connect(
            host="10.137.173.70",
            port="5432",
            database="ggg",
            user="ggg",
            password="ok"
        )
        return con
    except Exception as error:
        print("Oops! An exception has occured:", error)
        print("Exception TYPE:", type(error))


schemaname = find_openmap_schema("gbr").nspname[0]


# Query all schemas in Openmap database

def ovAdminAreaOrder8Area(schema):
    adminArea = """SELECT osm_id ,admin_level,boundary,"name",place,country, ST_AsText(way) as geometry
    FROM "{schema_name}".planet_osm_polygon 
    where boundary= 'administrative' and admin_level like '8'"""
    adminAreaAa8 = adminArea.replace("{schema_name}", str(schema))
    AdminOrdr8Area = pd.read_sql(adminAreaAa8, postgres_db_connection())
    # Convert the WKB coordinates to Shapely geometries
    # AdminOrdr8Area['geometry'] = AdminOrdr8Area['way'].apply(wkb.loads)
    return AdminOrdr8Area


def distanceCalculatioByGroup(input_df: pd.DataFrame) -> pd.DataFrame:
    global _, row
    # Calculate the distance between points within each group
    distances = []
    current_group = None
    for _, row in input_df.iterrows():
        if row['group_id'] != current_group:
            current_group = row['group_id']
            prev_point = row['PointLocation']
            distances.append(0)  # Assign a default distance of 0 for the first point in each group
        else:
            curr_point = row['PointLocation']
            prev_coords = prev_point.coords[0]
            curr_coords = curr_point.coords[0]
            distance = haversine_distance(prev_coords[1], prev_coords[0], curr_coords[1], curr_coords[0])
            distances.append(distance)
            prev_point = curr_point
    # Assign the calculated distances to a new column
    input_df['distance'] = distances

    return input_df


def finalAddressRanges(df_exploded: DataFrame) -> DataFrame:
    # Select the columns of interest
    columns_to_check = ['hnr_Number', 'street', 'place_name']

    # Check for duplicate records based on the selected columns
    duplicates = df_exploded.duplicated(subset=columns_to_check, keep=False)

    # Filter the DataFrame to select only the duplicate records
    duplicate_records = df_exploded[duplicates]

    # Assign a unique ID to each duplicate group
    duplicate_records['group_id'] = duplicate_records.groupby(['hnr_Number', 'street', 'place_name']).ngroup()

    # Sort the DataFrame based on 'hnr_Number', 'street', and 'place_name' columns
    sorted_duplicates = duplicate_records.sort_values(by=['hnr_Number', 'street', 'place_name'])

    # Reorder the columns
    reordered_columns = ['osm_id', 'hnr_Number', 'street', 'place_name', 'group_id', 'min_hsn', 'max_hsn',
                         'hnr_array', 'interpolation', 'intermediates',
                         'hnr_ranges', 'hnr_numeric_mixed_array', 'PointLocation', 'way', 'place_value']
    sorted_df = sorted_duplicates[reordered_columns]

    # Drop multiple columns
    houseNumberRemove = ['way', 'hnr_array', 'hnr_numeric_mixed_array']
    hnrAddressSorted = sorted_df.drop(houseNumberRemove, axis=1)

    hnrAddfiltered_two = pd.DataFrame(
        columns=hnrAddressSorted.columns)  # DataFrame for groups with exactly two rows
    hnrAddfiltered_other = pd.DataFrame(columns=hnrAddressSorted.columns)  # DataFrame for groups with other lengths

    for _, group in hnrAddressSorted.groupby('group_id'):
        if len(group) == 2:  # Check if the group has exactly two rows remove Same Duplicate
            hnrAddfiltered_two = hnrAddfiltered_two.append(
                group)  # Append the group to hnrAddfiltered_two DataFrame
        else:
            hnrAddfiltered_other = hnrAddfiltered_other.append(
                group)  # Append the group to hnrAddfiltered_other DataFrame

    hnrAddfiltered_two = hnrAddfiltered_two.drop_duplicates(
        subset=['osm_id', 'hnr_Number', 'street', 'place_name', 'group_id'], keep=False)

    frames = [hnrAddfiltered_two, hnrAddfiltered_other]
    # Merge DataFrame
    AddrssRangesDuplicateHNR = pd.concat(frames)

    # Convert single 'group_id' to int dtype.
    AddrssRangesDuplicateHNR['group_id'] = AddrssRangesDuplicateHNR['group_id'].astype('int')

    # Convert single 'osm_id' to int dtype.
    AddrssRangesDuplicateHNR['osm_id'] = AddrssRangesDuplicateHNR['osm_id'].astype(np.int64)

    # Create rank based on group_id and osm_id
    AddrssRangesDuplicateHNR["rank"] = AddrssRangesDuplicateHNR.groupby("group_id")["osm_id"].rank(method="dense",
                                                                                                   ascending=False).astype(
        int)

    # Assuming AddrssRangesDuplicateHNR is your DataFrame
    sorted_df = AddrssRangesDuplicateHNR.sort_values(by=["group_id", "rank"])
    return sorted_df


# Create a GeoPandas DataFrame
# spatial_query_result = gpd.GeoDataFrame(AdminOrdr8Area, geometry='geometry')

# Create query for addresses to reverse lookup

query = """

with sample as (select index_searched_query
                        ,st_geomfromtext (coordinates, 4326)  coordinates
                         from (VALUES (0, '{query_coordinates}')) as t (index_searched_query, coordinates))

, tags as (
select distinct skeys(tags) keys
from "{schema_name}".planet_osm_polygon pop 
where admin_level  in ('4', '8')
)


, name_tags as (
select * 
from tags
where (keys like '%name:%' or keys like '%alt%name') and keys not like '%pronunciation%'
)   


, hsn_tags as (
select distinct skeys(tags) keys 
from "{schema_name}".planet_osm_point
where "addr:housenumber" is not null or tags::text like '%addr:housenumber%'

)



, hsn_keys as (
select * from hsn_tags where (keys like '%addr:housenumber%')

)
,buffers as (
select 
    sample.index_searched_query
,   sample.coordinates
,   coordinates as buffer
,   road.road as road_name
,   road.name_tags_array as road_names
from sample


left join lateral (
                SELECT name as road, array_remove(tags->array((select keys from name_tags)), null) as name_tags_array
                FROM "{schema_name}".planet_osm_line road
                where name is not null
                and highway  in ('motorway','motorway_link','trunk','trunk_link','primary','primary_link','secondary','secondary_link','tertiary','tertiary_link','unclassified','residential','service','living_street','road','steps', 'footway', 'path', 'pedestrian', 'bridleway', 'cycleway', 'track')
                ORDER BY road.way <-> sample.coordinates
                LIMIT 1
                ) AS road 
 on true
 )


,  address_ranges as (
select 
buffers.index_searched_query
,   buffers.coordinates
,   buffers.road_name
,   buffers.road_names
,   hnr.osm_id
,   ST_astext(hnr.way) way
,   hnr."addr:interpolation" as interpolation
,   hnr.tags
,   hnr.tags->'addr:street' as road_name_way
,   hnr.tags->'addr:interpolation' as interpolation_tag
,   hnr."name" 
,   unnest(ways.nodes) nodes

from "{schema_name}".planet_osm_line hnr

join buffers on ST_Intersects(buffers.buffer, hnr.way)

join "{schema_name}".planet_osm_ways ways  on ways.id = hnr.osm_id

where hnr."addr:interpolation" is not null 
)

,   hsn as (
select 
pop.tags as tags_hsn
,   array_remove(array_append(pop.tags -> array((select keys from hsn_keys )), pop."addr:housenumber"), null) as range_hsn
, address_ranges.*

from address_ranges

left join "{schema_name}".planet_osm_point pop 
on pop.osm_id = address_ranges.nodes

where pop.tags is not null
)


,   hsn_long as (
select 
hsn.osm_id
,   hsn.index_searched_query
,   hsn.coordinates
,   hsn.tags as tags_network
,   hsn.road_name_way
,   hsn.road_name
,   hsn.road_names
,   hsn.interpolation
,   hsn.interpolation_tag
,   hsn.way
,   hsn.name
,   first_value(tags_hsn) over(partition by osm_id) as first_tags_hsn
,   unnest(range_hsn) as range_hsn

from hsn 
)
,addressrangesfinal as (select 
	hsn_long.osm_id
,   hsn_long.road_name
,   hsn_long.way
,   min(range_hsn) as min_hsn
,   max(range_hsn) as max_hsn	
,   hsn_long.index_searched_query
,   '{date}' as date
,   '{version}' as version
,   ST_AsText(hsn_long.coordinates) as coordinates
,   hsn_long.tags_network
,   hsn_long.road_name_way
,   hsn_long.interpolation
,   hsn_long.road_names
,   hsn_long.interpolation_tag
,   hsn_long.name
,   first_tags_hsn as tags
,   array_agg(distinct range_hsn) as intermediates
from hsn_long
group by 
	hsn_long.osm_id
,   hsn_long.index_searched_query
,   coordinates
,   hsn_long.road_name
,   hsn_long.road_names
,   hsn_long.tags_network
,   hsn_long.road_name_way
,   hsn_long.interpolation
,   hsn_long.interpolation_tag
,   hsn_long.way
,   hsn_long.name
,   first_tags_hsn
order by hsn_long.osm_id)

select 
plarea.osm_id as place_osm_id ,
plarea.name as place_name, 
plarea.reg_code as reg_code, 
plarea.region as place_region, 
plarea.cntry_code as place_cntry_code,
plarea.country as place_country, 
plarea.place as place_value,
ST_AsText(plarea.way) as place_way,
addressrangesfinal.*
from "{schema_name}".planet_osm_polygon as plarea
INNER JOIN addressrangesfinal ON ST_Intersects(ST_SetSRID(addressrangesfinal.way, 4326), ST_SetSRID(plarea.way, 4326))
where plarea.tags->'index:level'= '8' and plarea.tags->'index:priority:8'='30'

"""

# Get Geometry From Admin area

if __name__ == '__main__':
    for i, arow in ovAdminAreaOrder8Area(schemaname).iterrows():
        print(f"Id Number is: {i}, OSM ID is {arow['osm_id']}: Admin Area: ,{arow['name']}")

        adminOrder8areaName = arow['name']
        query_coordinates = arow['geometry']

        # query_coordinates = ovAdminAreaOrder8Area(schemaname).head(1).geometry.values[0]

        # Replace "schema_name" in SQL
        adminAreaAa8 = query.replace("{schema_name}", str(schemaname))

        # Replace Polygon "Geometry" in SQL
        addresRanges = adminAreaAa8.replace('{query_coordinates}', query_coordinates)

        # Hit to Postgres Database nad created pandas DataFrame
        AdminOrdr8Area = pd.read_sql(addresRanges, postgres_db_connection())
        if AdminOrdr8Area.empty:
            print(f"Empty Data Frame : Id Number is: {i}Admin Area: ,{arow['name']}")
        else:
            parse_hnr_tags_df = parse_hnr_tags(AdminOrdr8Area)

            preprocess_hnr_hsn_df = preprocess_hnr_hsn(parse_hnr_tags_df)

            get_hnr_df_DF = get_hnr_df(preprocess_hnr_hsn_df)

            # Apply the correction function to the "hnr_array" column
            get_hnr_df_DF['hnr_array'] = get_hnr_df_DF['hnr_array'].apply(correct_hnr_array)

            # Apply the correction function to the "intermediates" column
            get_hnr_df_DF['intermediates'] = get_hnr_df_DF['intermediates'].apply(correct_hnr_array)

            #### intermediates,nterpolation == irregular ######

            # Apply the function to the 'way' column and save the result in 'PointLocation' column
            get_hnr_df_DF['PointLocation'] = get_hnr_df_DF['way'].apply(lambda x: calculate_center_point(
                Point(float(coord.split()[0]), float(coord.split()[1])) for coord in
                x.strip('LINESTRING()').split(',')))

            # Check if 'interpolation' is ['irregular', 'odd', 'numeric_mixed', 'even'] is not null

            df_exploded = interpolationTypeHandalling(get_hnr_df_DF, correct_hnr_array)

            houseNumberArray = StartAndEndHNRSame(df_exploded)

            #############################################################
            ####################Array Issue#####################
            #############################################################
            filenamearray = 'GBR_AddressRanges_array.csv'
            outputpath = r"E:\\Amol\\9_addressRangesPython\\"
            filename = 'GBR_AddressRanges_singla_Array.csv'
            if not houseNumberArray.empty:
                pass
            else:
                csvFileWriter(houseNumberArray, filenamearray, outputpath)

            print("House Number Ranges Done, Admin Area {}".format(adminOrder8areaName))

            # houseNumberArray.to_csv(r"E:\\Amol\\9_addressRangesPython\\1.ArrayExplodAddrssRanges.csv")
            print("Array Done")
            ###########################House Number Duplicate#############################
            # Remaning Processing
            sorted_df = finalAddressRanges(df_exploded)
            # calculate distance in Meter by each group
            AddressRangesFinal = distanceCalculatioByGroup(sorted_df)

            # Create a dictionary to map old column names to new column names
            finalColumnNames = {'place_name': 'area_name',
                                'min_hsn': 'first hsn',
                                'max_hsn': 'last hsn',
                                'hnr_Number': 'Dup_hnr_Number'
                                }

            # Rename the columns using the dictionary
            AddressRangesFinaldf = AddressRangesFinal.rename(columns=finalColumnNames)

            AddressRangesFinaldf['Country code'] = 'GBR'
            AddressRangesFinaldf['Index level'] = '8'
            AddressRangesFinaldf['AA8Name'] = adminOrder8areaName
            # Reorder Column
            finalReorder = ['Country code', 'place_value', 'Index level', 'area_name', 'street', 'osm_id'
                , 'interpolation', 'first hsn', 'last hsn', 'intermediates', 'Dup_hnr_Number', 'hnr_ranges'
                , 'group_id', 'distance', 'PointLocation', 'rank', 'AA8Name']

            addressRangesOutPut = AddressRangesFinaldf[finalReorder]

            # addressRangesOutPut.to_csv(r"E:\\Amol\\9_addressRangesPython\\AddressRangesFinal.csv")
            # Display the duplicate records
            #############################################################
            # Get the current time
            current_time = datetime.datetime.now()
            # Format the time as a string (optional)
            formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
            #############################################################
            ####################Adress Ranges Output#####################
            #############################################################
            outputpath = r"E:\\Amol\\9_addressRangesPython\\"
            filename = 'GBR_AddressRanges_multiple_duplicate.csv'


            csvFileWriter(addressRangesOutPut, filename, outputpath)
            print(f"House Number Ranges Done, Admin Area: {adminOrder8areaName}, Current time: {formatted_time}")
# creatting Geomaty for Admin area, Plance name , Address Ranges
# def PandasToGeopandasGeoemtryExport():
#     """
#     :return: # creatting Geomaty for Admin area, Plance name , Address Ranges
#     """
#     # # Create new geometry column from the "way" column
#     get_hnr_df_DF['geometry'] = get_hnr_df_DF['way'].apply(lambda way: loads(way.split(';')[0]))
#     #
#     # # Create a GeoPandas DataFrame
#     spatial_query_result = gpd.GeoDataFrame(get_hnr_df_DF, geometry='geometry')
#     # Convert non-compatible columns to string
#     non_compatible_types = ['object', 'bool']  # Add more types if needed
#     non_string_columns = spatial_query_result.select_dtypes(
#         exclude=['string', 'int', 'float', 'datetime', 'geometry']).columns
#     for column in non_string_columns:
#         if spatial_query_result[column].dtype.name in non_compatible_types:
#             spatial_query_result[column] = spatial_query_result[column].astype(str)
#     # export to line
#     # Export to GeoPackage Adress Ranges
#     pathline = r"E:\\Amol\\9_addressRangesPython\\AddrssRangeslineArray.gpkg"
#     spatial_query_result.to_file(pathline, layer='AddrssRanges', driver='GPKG')
#     #### create Polygon Geometry Admin order 8 area
#     AA8gdf = spatial_query_result.copy()
#     # Remove the existing "coordinates" column
#     AA8gdf.drop(columns='geometry', inplace=True)
#     #
#     AA8gdf_duplicates = AA8gdf.drop_duplicates(subset=['coordinates'])
#     # Create new geometry column from the "way" column
#     AA8gdf_duplicates['geometry'] = AA8gdf_duplicates['coordinates'].apply(
#         lambda coordinates: loads(coordinates.split(';')[0]))
#     # # Create a GeoPandas DataFrame
#     admiAreaOrder8AreaGDF = gpd.GeoDataFrame(AA8gdf_duplicates, geometry='geometry')
#     # # Export to GeoPackage
#     admiAreaOrder8AreaGDF.to_file(pathline, layer='AdminOrder8Area', driver='GPKG')
#     #### create Polygon Geometry Placename
#     placegdf = spatial_query_result.copy()
#     # Remove the existing "coordinates" column
#     placegdf.drop(columns='geometry', inplace=True)
#     #
#     placegdf_duplicates = placegdf.drop_duplicates(subset=['place_way'])
#     # Create new geometry column from the "way" column
#     placegdf_duplicates['geometry'] = placegdf_duplicates['place_way'].apply(
#         lambda place_way: loads(place_way.split(';')[0]))
#     # # Create a GeoPandas DataFrame
#     placeGDF = gpd.GeoDataFrame(placegdf_duplicates, geometry='geometry')
#     # Export to GeoPackage
#     placeGDF.to_file(pathline, layer='place', driver='GPKG')
#
# PandasToGeopandasGeoemtryExport()
