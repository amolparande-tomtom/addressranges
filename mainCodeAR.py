import re
import numpy as np
import pandas as pd
import psycopg2
import typing
from shapely import wkb
import geopandas as gpd
from shapely.wkt import loads
from map_content.utils import utils
# from map_content.utils.openmap import get_alphabetic_hnr_df, get_numeric_hnr_df

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


def get_numeric_hnr_df(hnr_df: pd.DataFrame) -> pd.DataFrame:
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


con = psycopg2.connect(
    host="10.137.173.71",
    port="5432",
    database="ggg",
    user="ggg",
    password="ok"
)


def postgres_db_connection():
    """
    :param db_url: Postgres Server
    :return: DB Connection
    """
    try:
        con = psycopg2.connect(
            host="10.137.173.71",
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


query_coordinates = ovAdminAreaOrder8Area(schemaname).head(1).geometry.values[0]

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

, buffers as (
select 
    sample.index_searched_query
,   sample.coordinates
,   coordinates as buffer
,   postcode.postcode as postal_code
,   city.name as city_name
,   city.name_tags_array as city_names
,   state.name as state_name
,   state.name_tags_array as state_names
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


left outer join(

    select index_searched_query, array_agg(distinct postcode) as postcode 
    from
            (Select
            sample.index_searched_query,
            way,
        --   array_agg( array_remove(ARRAY[tags->'postal_code',tags->'addr:postal_code',tags->'postcode', tags->'addr:postcode'], null) ) postcode
            unnest(array_remove(tags->array['postal_code', 'addr:postal_code', 'postcode', 'addr:postcode'], null)) as postcode
            From "{schema_name}".planet_osm_polygon polygon

            join sample on ST_Intersects(sample.coordinates, polygon.way)

            Where (tags?|ARRAY['postal_code','addr:postal_code','postcode','addr:postcode'] AND boundary = 'administrative')
            OR boundary = 'postal_code'

            ) a

    where postcode is not null
    group by index_searched_query    

                ) AS postcode
on postcode.index_searched_query = sample.index_searched_query


left outer join (
                select admin_level , place, name, way,
                array_remove(tags->array((select keys from name_tags)), null) as name_tags_array
                from "{schema_name}".planet_osm_polygon pop 
                where admin_level in ('8')
                and boundary ='administrative'

                ) city 
on ST_Intersects(sample.coordinates, city.way)


left outer join (
    select admin_level , place, name, way,
    array_remove(tags->array((select keys from name_tags)), null) as name_tags_array
    from "{schema_name}".planet_osm_polygon pop 
    where admin_level in ('4')
    and boundary ='administrative'

    ) state 
        on ST_Intersects(sample.coordinates, state.way)
)


,  address_ranges as (
select 
buffers.index_searched_query
,   buffers.coordinates
,   buffers.postal_code
,   buffers.city_name
,   buffers.city_names
,   buffers.state_name
,   buffers.state_names
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
,   hsn.postal_code
,   hsn.city_name
,   hsn.city_names
,   hsn.state_name
,   hsn.state_names
,   hsn.road_name
,   hsn.road_names
,   hsn.tags as tags_network
,   hsn.road_name_way
,   hsn.interpolation
,   hsn.interpolation_tag
,   hsn.way
,   hsn.name
,   first_value(tags_hsn) over(partition by osm_id) as first_tags_hsn
,   unnest(range_hsn) as range_hsn

from hsn 
)


select 
hsn_long.osm_id
,   hsn_long.index_searched_query
,   '{date}' as date
,   '{version}' as version
,   hsn_long.coordinates
,   hsn_long.postal_code
,   hsn_long.city_name
,   hsn_long.city_names
,   hsn_long.state_name
,   hsn_long.state_names
,   hsn_long.road_name
,   hsn_long.road_names
,   hsn_long.tags_network
,   hsn_long.road_name_way
,   hsn_long.interpolation
,   hsn_long.interpolation_tag
,   hsn_long.way
,   hsn_long.name
,   first_tags_hsn as tags
,   min(range_hsn) as min_hsn
,   max(range_hsn) as max_hsn
,   array_agg(distinct range_hsn) as intermediates
from hsn_long
group by 
hsn_long.osm_id
,   hsn_long.index_searched_query
,   hsn_long.coordinates
,   hsn_long.postal_code
,   hsn_long.city_name
,   hsn_long.city_names
,   hsn_long.state_name
,   hsn_long.state_names
,   hsn_long.road_name
,   hsn_long.road_names
,   hsn_long.tags_network
,   hsn_long.road_name_way
,   hsn_long.interpolation
,   hsn_long.interpolation_tag
,   hsn_long.way
,   hsn_long.name
,   first_tags_hsn

"""

adminAreaAa8 = query.replace("{schema_name}", str(schemaname))

addresRanges = adminAreaAa8.replace('{query_coordinates}', query_coordinates)

# pandas DataFrame
AdminOrdr8Area = pd.read_sql(addresRanges, postgres_db_connection())

get_hnr_df(AdminOrdr8Area)

# # Create new geometry column from the "way" column
# AdminOrdr8Area['geometry'] = AdminOrdr8Area['way'].apply(lambda way: loads(way.split(';')[0]))
#
# # Create a GeoPandas DataFrame
# spatial_query_result = gpd.GeoDataFrame(AdminOrdr8Area, geometry='geometry')
#
# # Convert non-compatible columns to string
# non_compatible_types = ['object', 'bool']  # Add more types if needed
# non_string_columns = spatial_query_result.select_dtypes(
#     exclude=['string', 'int', 'float', 'datetime', 'geometry']).columns
#
# for column in non_string_columns:
#     if spatial_query_result[column].dtype.name in non_compatible_types:
#         spatial_query_result[column] = spatial_query_result[column].astype(str)
#
# # Export to GeoPackage
# pathline = r"E:\\Amol\\9_addressRangesPython\\PolygonAddrssRangesline.gpkg"
#
# spatial_query_result.to_file(pathline, driver='GPKG')
# #
# # # create Polygon Geometry
# # gdf = spatial_query_result.copy()
# #
# # # Remove the existing "coordinates" column
# # gdf.drop(columns='geometry', inplace=True)
# #
# # df_no_duplicates_specific = gdf.drop_duplicates(subset=['way'])
# #
# # # Create new geometry column from the "way" column
# # df_no_duplicates_specific['geometry'] = df_no_duplicates_specific['coordinates'].apply(lambda way: loads(way.split(';')[0]))
# #
# # # Export to GeoPackage
# # pathline = r"E:\\Amol\\9_addressRangesPython\\PolygonAddrssRangesline.gpkg"
# #
# # df_no_duplicates_specific.to_file(pathline, driver='GPKG')