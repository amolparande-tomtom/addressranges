import pandas as pd
import psycopg2
from shapely import wkb
import geopandas as gpd

from adressRang import find_openmap_schema

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

ovAdminAreaOrder8Area(schemaname).head(1).geometry.values[0]

# Create a GeoPandas DataFrame
# spatial_query_result = gpd.GeoDataFrame(AdminOrdr8Area, geometry='geometry')



