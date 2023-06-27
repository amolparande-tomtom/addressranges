import pandas as pd
import psycopg2
from shapely import wkb
import geopandas as gpd
from shapely.wkt import loads
from map_content.utils import utils




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


select 
	hsn_long.osm_id
,   hsn_long.road_name
,   hsn_long.way
,   min(range_hsn) as min_hsn
,   max(range_hsn) as max_hsn	
,   hsn_long.index_searched_query
,   '{date}' as date
,   '{version}' as version
,   hsn_long.coordinates
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
,   hsn_long.coordinates
,   hsn_long.road_name
,   hsn_long.road_names
,   hsn_long.tags_network
,   hsn_long.road_name_way
,   hsn_long.interpolation
,   hsn_long.interpolation_tag
,   hsn_long.way
,   hsn_long.name
,   first_tags_hsn

order by hsn_long.osm_id



"""

adminAreaAa8 = query.replace("{schema_name}", str(schemaname))

addresRanges = adminAreaAa8.replace('{query_coordinates}', query_coordinates)

AdminOrdr8Area = pd.read_sql(addresRanges, postgres_db_connection())





# Create new geometry column from the "way" column
AdminOrdr8Area['geometry'] = AdminOrdr8Area['way'].apply(lambda way: loads(way.split(';')[0]))

# Create a GeoPandas DataFrame
spatial_query_result = gpd.GeoDataFrame(AdminOrdr8Area, geometry='geometry')

# Convert non-compatible columns to string
non_compatible_types = ['object', 'bool']  # Add more types if needed
non_string_columns = spatial_query_result.select_dtypes(
    exclude=['string', 'int', 'float', 'datetime', 'geometry']).columns

for column in non_string_columns:
    if spatial_query_result[column].dtype.name in non_compatible_types:
        spatial_query_result[column] = spatial_query_result[column].astype(str)

# Export to GeoPackage
pathline = r"E:\\Amol\\9_addressRangesPython\\PolygonAddrssRangesline.gpkg"

spatial_query_result.to_file(pathline, driver='GPKG')
#
# # create Polygon Geometry
# gdf = spatial_query_result.copy()
#
# # Remove the existing "coordinates" column
# gdf.drop(columns='geometry', inplace=True)
#
# df_no_duplicates_specific = gdf.drop_duplicates(subset=['way'])
#
# # Create new geometry column from the "way" column
# df_no_duplicates_specific['geometry'] = df_no_duplicates_specific['coordinates'].apply(lambda way: loads(way.split(';')[0]))
#
# # Export to GeoPackage
# pathline = r"E:\\Amol\\9_addressRangesPython\\PolygonAddrssRangesline.gpkg"
#
# df_no_duplicates_specific.to_file(pathline, driver='GPKG')