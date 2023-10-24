import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extensions import connection
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, expr, udf,col,size,lit,count

from pyspark.sql.types import StructType, StructField, StringType,ArrayType, IntegerType




# Create Spark Session

spark = SparkSession.builder.appName("Address Ranges Count by Country").getOrCreate()

def parse_hnr_tags(tags, tags_network, interpolation):
    # Initialize variables to store parsed information
    constant = None
    interpolation_value = None
    intermediates = None
    street = None

    # Parse tags
    if tags:
        tags_list = [tag.split("=>") for tag in tags.split('",')]
        for tag in tags_list:
            key = tag[0].strip('" ')
            value = tag[1].strip('" ')
            if key == "constant":
                constant = value
            if key == "street":
                street = value

    # Parse tags_network
    if tags_network:
        tags_network_list = [tag.split("=>") for tag in tags_network.split('",')]
        for tag in tags_network_list:
            key = tag[0].strip('" ')
            value = tag[1].strip('" ')
            if key == "interpolation":
                interpolation_value = value
            if key == "intermediate":
                intermediates = [hsn.strip() for hsn in value.split(",")]

    # If interpolation is not provided, use interpolation_value
    if interpolation is None:
        interpolation = interpolation_value

    # Return the parsed information as a tuple
    return (constant, interpolation_value, interpolation, intermediates, street)



# Sample preprocessing logic (replace with your actual preprocessing logic)
def preprocess_hnr_hsn_udf(min_hsn, max_hsn):
    def process_hsn(hsn):
        # Sample preprocessing logic for a single house number
        # Replace this with your own logic to process a single house number
        if hsn is not None:
            hsn = hsn.strip()  # Remove leading and trailing spaces
            hsn = hsn.upper()  # Convert to uppercase
        return hsn

    min_hsn = process_hsn(min_hsn)
    max_hsn = process_hsn(max_hsn)

    return (min_hsn, max_hsn)

# Sample preprocessing logic for get_hnr_df (replace with your actual logic)
def get_hnr_df_udf(interpolation):
    # Sample logic to produce the house number range
    if interpolation == "alphabetic":
        # Your alphabetic interpolation logic here
        hnr_range = "Alphabetic Range"
    else:
        # Your numeric interpolation logic here
        hnr_range = "Numeric Range"

    return hnr_range

# Sample preprocessing logic for get_alphabetic_hnr_df_udf
# Sample preprocessing logic for get_alphabetic_hnr_df_udf
def get_alphabetic_hnr_df_udf(min_hsn, max_hsn):
    # Extract the first character from min_hsn and max_hsn
    first_char = max_hsn[0]

    # Extract numeric part of hsn if present, or assume 1 as the minimum
    min_numeric = int(min_hsn[1:]) if min_hsn[1:].isdigit() else 1
    max_numeric = int(max_hsn[1:]) if max_hsn[1:].isdigit() else 1

    # Sample logic to produce alphabetic variance house number ranges
    hnr_range = [f"{first_char}{i}" for i in range(min_numeric, max_numeric + 1)]

    return hnr_range


# Sample logic for correct_hnr_array (replace with your actual logic)
def correct_hnr_array_udf(arr):
    if isinstance(arr, float):  # Check for None or float
        return []

    if isinstance(arr, list):
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

# Define the get_numeric_hnr_df_udf UDF
def get_numeric_hnr_df_udf(min_hsn, max_hsn, interpolation):
    hnr_array = []

    def safe_int(value):
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    min_hsn_numeric = safe_int(min_hsn)
    max_hsn_numeric = safe_int(max_hsn)

    if min_hsn_numeric is not None and max_hsn_numeric is not None:
        if interpolation == "even":
            for i in range(min_hsn_numeric, max_hsn_numeric + 1, 2):
                hnr_array.append(str(i))
        elif interpolation == "odd":
            for i in range(min_hsn_numeric + 1, max_hsn_numeric + 1, 2):
                hnr_array.append(str(i))
        else:
            for i in range(min_hsn_numeric, max_hsn_numeric + 1):
                hnr_array.append(str(i))

    return hnr_array




def get_country_schema(country: str, conn: connection) -> str:
    """
    :param conn:
    :param country: Country Name in ISO-3  Code
    :return: Schema for Given ISO-3 Country Code
    """
    # Schemas. list in PostgreSQL database.
    schemas_df = pd.read_sql('select * from pg_catalog.pg_namespace', con=conn)
    # conn.close()
    return schemas_df.loc[schemas_df.nspname.str.contains(f'_{country}')].nspname.iloc[0]


def adminAreaList(schema: str, admin_level: str, conn: connection) -> list:
    adminlist = """
    SELECT "name" 
    FROM {schema}.planet_osm_polygon
    where boundary= 'administrative' and admin_level = '{admin_level}'""".format(schema=schema, admin_level=admin_level)
    schemas_df = pd.read_sql(adminlist, con=conn)
    # conn.close()
    AdminNames = [i for i in schemas_df.name]
    return AdminNames

def format_query(schema, admin_level,admin) -> str:
    query = """with sample as(SELECT osm_id as aa8_osm_id ,"name" as index_searched_query,ST_SetSRID(way, 4326) as coordinates
    FROM "{schema}".planet_osm_polygon
    where boundary= 'administrative' and admin_level = '{admin_level}' and "name" = '{admin}'
    )

, tags as (
select distinct skeys(tags) keys
from "{schema}".planet_osm_polygon pop
where admin_level  in ('4', '8')
)

, hnr_way as (
select sample.aa8_osm_id, sample.index_searched_query, pol.* 
from {schema}.planet_osm_line pol 
join sample on ST_Intersects(pol.way, sample.coordinates)
where "addr:interpolation" is not null 

)

, name_tags as (
select *
from tags
where (keys like '%name:%' or keys like '%alt%name') and keys not like '%pronunciation%'
)

, hsn_tags as (
select distinct skeys(tags) keys
from "{schema}".planet_osm_point
where "addr:housenumber" is not null or tags::text like '%addr:housenumber%'
)
, hsn_keys as (
select * from hsn_tags where (keys like '%addr:housenumber%')
)

, address_ranges as (
select 
	hnr_way.index_searched_query
,   hnr_way.aa8_osm_id
,	hnr_way.osm_id
,   ST_astext(hnr_way.way) way
,   hnr_way."addr:interpolation" as interpolation
,   hnr_way.tags
,   hnr_way.tags->'addr:street' as road_name_way
,   hnr_way.tags->'addr:interpolation' as interpolation_tag
,   hnr_way."name"
,   unnest(ways.nodes) nodes

from hnr_way
join "{schema}".planet_osm_ways ways  on ways.id = hnr_way.osm_id

) 

,   hsn as (
select
pop.tags as tags_hsn
,   array_remove(array_append(pop.tags -> array((select keys from hsn_keys )), pop."addr:housenumber"), null) as range_hsn
, address_ranges.*
from address_ranges
left join "{schema}".planet_osm_point pop
on pop.osm_id = address_ranges.nodes
where pop.tags is not null and pop.tags-> 'layer_id' = '15633'
)

,   hsn_long as (
    select
    hsn.osm_id
,   hsn.index_searched_query
,   hsn.aa8_osm_id
--,   hsn.coordinates
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
,addressrangesfinal as (select
    hsn_long.osm_id
,   hsn_long.way
,   min(range_hsn) as min_hsn
,   max(range_hsn) as max_hsn
,   hsn_long.index_searched_query
,   hsn_long.aa8_osm_id
--,   ST_AsText(hsn_long.coordinates) as coordinates
,   hsn_long.tags_network
,   hsn_long.road_name_way
,   hsn_long.interpolation
,   hsn_long.interpolation_tag
,   hsn_long.name
,   first_tags_hsn as tags
,   array_agg(distinct range_hsn) as intermediates
from hsn_long
group by
    hsn_long.osm_id
,   hsn_long.index_searched_query
,   hsn_long.aa8_osm_id
--,   coordinates
,   hsn_long.tags_network
,   hsn_long.road_name_way
,   hsn_long.interpolation
,   hsn_long.interpolation_tag
,   hsn_long.way
,   hsn_long.name
,   first_tags_hsn)


select * from addressrangesfinal
""".format(schema=schema, admin_level=admin_level,admin = admin)

    return query




host = '10.137.173.42'
database = 'ggg'
user = 'ggg'
password = 'ok'
port = 5432

# establish connection
conn = psycopg2.connect(host=host, database=database, user=user, password=password, port=port)

# schema = get_country_schema('fra', conn)

# admin = adminAreaList(schema,'8',conn)

# print(admin)

# Main Code
all_dfs = []
for country in ['aut']:
    country_schema = get_country_schema(country,conn)
    adminOrder = adminAreaList(country_schema,'8',conn)

    for admin in adminOrder:
        adminnew = admin.replace("'", '"')
        formatted_query = format_query(country_schema, '8',adminnew)
        # Define the PGSQL server connection properties
        host = '10.137.173.42'
        database = 'ggg'
        user = 'ggg'
        password = 'ok'
        port = 5432

        # Define the PGSQL server connection properties
        pg_properties = {
            "user": "ggg",
            "password": "ok",
            "driver": "org.postgresql.Driver",
            "url": "jdbc:postgresql://10.137.173.42:5432/ggg"
        }

        # Step 4: Read data from PostgreSQL
        df = spark.read.jdbc(url=pg_properties["url"], table=f"({formatted_query}) as subquery", properties=pg_properties)

        # Add a new column "country" with the value 'FRA' to every row
        df = df.withColumn("country", lit(country))

        # Register the UDF function with PySpark
        udf_parse_hnr_tags = udf(parse_hnr_tags, StructType([
            StructField("constant", StringType(), True),
            StructField("interpolation_value", StringType(), True),
            StructField("interpolation", StringType(), True),
            StructField("intermediates", StringType(), True),
            StructField("street", StringType(), True)
        ]))

        # Apply the UDF to the DataFrame "parse_hnr_tags" function
        df = df.withColumn("parsed_hnr_tags", udf_parse_hnr_tags(df["tags"], df["tags_network"], df["interpolation"]))

        # Register the UDF function with PySpark
        udf_preprocess_hnr_hsn = udf(preprocess_hnr_hsn_udf, StructType([
            StructField("min_hsn_numeric", StringType(), True),
            StructField("max_hsn_numeric", StringType(), True)
        ]))

        # Apply the UDF to the DataFrame "udf_preprocess_hnr_hsn"
        df = df.withColumn("preprocessed_hsn", udf_preprocess_hnr_hsn(df["min_hsn"], df["max_hsn"]))

        # Register the UDF function with PySpark
        udf_get_hnr_df = udf(get_hnr_df_udf, StringType())


        # Apply the UDF to the DataFrame
        df = df.withColumn("hnr_range", udf_get_hnr_df(df["interpolation"]))

        # Register the get_alphabetic_hnr_df_udf UDF with PySpark
        udf_get_alphabetic_hnr_df = udf(get_alphabetic_hnr_df_udf, ArrayType(StringType()))

        # Apply the get_alphabetic_hnr_df_udf UDF to the DataFrame with alphabetic data
        df = df.withColumn("hnr_range", udf_get_alphabetic_hnr_df(df["min_hsn"], df["max_hsn"]))
        # Show the results for alphabetic data
        # df.show(truncate=False)

        # Register the get_numeric_hnr_df_udf UDF with PySpark
        udf_get_numeric_hnr_df = udf(get_numeric_hnr_df_udf, ArrayType(StringType()))

        # Apply the get_numeric_hnr_df_udf UDF to the DataFrame with numeric data
        df = df.withColumn("hnr_array",udf_get_numeric_hnr_df(df["min_hsn"], df["max_hsn"],df["interpolation"]))

        # Register the UDF function with PySpark
        udf_correct_hnr_array = udf(correct_hnr_array_udf, ArrayType(StringType()))

        # # Apply the UDF to the DataFrame
        df = df.withColumn("corrected_hnr_array", udf_correct_hnr_array(df["hnr_array"]))
        # Count the elements in the "corrected_hnr_array" column and create a new column "hnr_array_count"
        df = df.withColumn("hnr_array_count", size(df["corrected_hnr_array"]))
        # Select and keep only the specified columns
        df = df.select("country","interpolation", "hnr_array_count","hnr_array", "intermediates" )

        # Group by "country" and "interpolation," and count "hnr_array_count" for each group
        df = df.groupBy("country", "interpolation").agg(count("hnr_array_count").alias("count"))

        # df.show(truncate=False)
        # break
        all_dfs.append(df)
# # Concatenate all DataFrames in all_dfs into a single DataFrame
# final_df = all_dfs[0].union(*all_dfs[1:])

# Concatenate the DataFrames and store the result in a new DataFrame
concatenated_df = all_dfs[0]
for df in all_dfs[1:]:
    concatenated_df = concatenated_df.union(df)

# Show the concatenated DataFrame
concatenated_df.show()

spark.stop()




