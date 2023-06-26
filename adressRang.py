import pandas as pd
import psycopg2
import geopandas as gpd # for creating geoDataframe once we have shapely created coordinates
import shapely
from shapely.wkt import loads
from shapely import wkb# for converting lat lon to a POINT coordinate
# from map_content.utils.openmap import apt_openmap_lookup
from map_content.utils import utils
# from map_content.utils.openmap import find_openmap_schema

# conurl = "postgresql://10.137.244.158/ggg?user=ggg&password=ok"

con = psycopg2.connect(
    host="10.137.244.158",
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
            host="10.137.244.158",
            port="5432",
            database="ggg",
            user="ggg",
            password="ok"
        )
        return con
    except Exception as error:
        print("Oops! An exception has occured:", error)
        print("Exception TYPE:", type(error))

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
    country_iso3 = 'nld'
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

def ovAdminAreaOrder8Area(schema):
    adminArea = """SELECT osm_id ,admin_level,boundary,"name",place,country, ST_AsText(way) as geometry
    FROM "{schema_name}".planet_osm_polygon 
    where boundary= 'administrative' and admin_level like '8'"""
    adminAreaAa8 = adminArea.replace("{schema_name}", str(schema))
    AdminOrdr8Area = pd.read_sql(adminAreaAa8, postgres_db_connection())
    # Convert the WKB coordinates to Shapely geometries
    # AdminOrdr8Area['geometry'] = AdminOrdr8Area['way'].apply(wkb.loads)
    return AdminOrdr8Area


def openmap_hnr_lookup(
    coordinates: gpd.GeoSeries, schemas: pd.DataFrame, credentials: dict or None = None
) -> gpd.GeoDataFrame:
    """Performs spatial queries on a list of coordinates to find the HNR
    in OSM format

    :param coordinates: geopandas.GeoSeries made out of coordinates (shapely.Point elements) for the APTs to lookup
    :type coordinates: gpd.GeoSeries
    :param schemas: pandas.DataFrame listing the schemas and database to lookup
    :type schemas: pd.DataFrame
    :param credentials: dictionary containing the credentials for a connection
    :type credentials: dict or None, optional
    :return: points in OM near the coordinates
    :rtype: gpd.GeoDataFrame
    """


    # Create query for addresses to reverse lookup
    query_coordinates = (
        """select index_searched_query
                        ,st_geomfromtext (coordinates, 4326)  coordinates
                         from """
        + utils.convert_coordinates_to_query(coordinates)
        + " as t (index_searched_query, coordinates)"
    )

    # Initialize connection if not passed as parameter
    if credentials is None:
        # _, conn = dbconnection.connect("../sql/database.ini", "osm-vad")
        conn = postgres_db_connection()
    else:
        conn = psycopg2.connect(**credentials)


    # Create DataFrame to get results
    lookup_df = pd.DataFrame()

    # Iterate over schemas and databases to make spatial queries
    for _, row in schemas.iterrows():
        query = """
       with sample as ({query_coordinates})

, tags as (
select distinct skeys(tags) keys
from {schema}.planet_osm_polygon pop 
where admin_level  in ('4', '8')
)

, name_tags as (
select * 
from tags
where (keys like '%name:%' or keys like '%alt%name') and keys not like '%pronunciation%'
)   

, hsn_tags as (
select distinct skeys(tags) keys 
from {schema}.planet_osm_point
where "addr:housenumber" is not null or tags::text like '%addr:housenumber%'

)

, hsn_keys as (
select * from hsn_tags where (keys like '%addr:housenumber%')

)

, buffers as (
select 
    sample.index_searched_query
,   sample.coordinates
,   ST_Transform(ST_Buffer(ST_Transform(ST_SetSRID(sample.coordinates, 4326), 3857), 1050), 4326) buffer
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
                FROM {schema}.planet_osm_line road
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
            From {schema}.planet_osm_polygon polygon

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
                from {schema}.planet_osm_polygon pop 
                where admin_level in ('8')
                and boundary ='administrative'

                ) city 
on ST_Intersects(sample.coordinates, city.way)


left outer join (
    select admin_level , place, name, way,
    array_remove(tags->array((select keys from name_tags)), null) as name_tags_array
    from {schema}.planet_osm_polygon pop 
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

from {schema}.planet_osm_line hnr

join buffers on ST_Intersects(buffers.buffer, hnr.way)

join {schema}.planet_osm_ways ways  on ways.id = hnr.osm_id

where hnr."addr:interpolation" is not null 
)

,   hsn as (
select 
pop.tags as tags_hsn
,   array_remove(array_append(pop.tags -> array((select keys from hsn_keys )), pop."addr:housenumber"), null) as range_hsn
, address_ranges.*

from address_ranges

left join {schema}.planet_osm_point pop 
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
            """.format(
            query_coordinates=query_coordinates,
            date=row.date,
            schema=row.nspname,
            version=row.date,
        )

        # Geopandas Query database
        # spatial_query_result = gpd.GeoDataFrame.from_postgis(
        #     query, conn, geom_col="coordinates"
        # )
        # pandas Query database
        pandasSpatialQueryR = pd.read_sql(query, conn)

        # Convert the WKB coordinates to Shapely geometries
        pandasSpatialQueryR['geometry'] = pandasSpatialQueryR['coordinates'].apply(wkb.loads)

        # Create a GeoPandas DataFrame
        spatial_query_result = gpd.GeoDataFrame(pandasSpatialQueryR, geometry='geometry')

        # Concat responses
        lookup_df = pd.concat([lookup_df, spatial_query_result], ignore_index=True)

    conn.close()

    return gpd.GeoDataFrame(lookup_df)



if __name__ == "__main__":

    # Create test dataframe
    test_addresses = [
        ("11-97 DE ENTREE AMSTERDAM 1101 HE NLD", 52.31177019833552, 4.939634271503648),
        ("Aalsterweg 303, 5644 RL Eindhoven, NL", 51.41176179168882, 5.482757611072691),
        ("Ammunitiehaven 343 2511 XM s Gravenhage NL",52.07742315143409,4.3212179573462075,),
        ("Baarsweg 148, 3192 VB, Hoogvliet Rotterdam, Ne...",51.856975720153564,4.350903401715045,),
        ("Baas Gansendonckstraat 3, 1061CZ Amsterdam, NL",52.37733757641722,4.840407597295104, ),
                    ]

    test_df = pd.DataFrame(test_addresses, columns=["searched_query", "lat", "lon"])
    test_df["coordinates"] = test_df.apply(lambda x: shapely.geometry.Point(x.lon, x.lat), axis=1)
    test_gdf = gpd.GeoDataFrame(test_df, geometry="coordinates")

    schema = find_openmap_schema("nl")
    # OV Admin area Order 8 Area
    schemaname = find_openmap_schema("gbr").nspname[0]
    aa8 = ovAdminAreaOrder8Area(schemaname).head(1).geometry.values[0]

    query_buffers = openmap_hnr_lookup(test_gdf.coordinates, schema )
    print(query_buffers)

    # Convert non-compatible columns to string
    non_compatible_types = ['object', 'bool']  # Add more types if needed
    non_string_columns = query_buffers.select_dtypes(exclude=['string', 'int', 'float', 'datetime', 'geometry']).columns

    for column in non_string_columns:
        if query_buffers[column].dtype.name in non_compatible_types:
            query_buffers[column] = query_buffers[column].astype(str)

    path = r"E:\\Amol\\9_addressRangesPython\\addrssRanges.gpkg"
    query_buffers.to_file(path, driver='GPKG')
    # query_buffers.to_file(path,index=False)
    # Export to geodatabase (Esri File Geodatabase or GDB)

    gdf = query_buffers.copy()

    # Create new geometry column from the "way" column
    gdf['geometry'] = gdf['way'].apply(lambda way: loads(way.split(';')[0]))

    # Remove the existing "coordinates" column
    gdf.drop(columns='coordinates', inplace=True)

    # Export to GeoPackage
    pathline = r"E:\\Amol\\9_addressRangesPython\\addrssRangesline.gpkg"

    gdf.to_file(pathline, driver='GPKG')


