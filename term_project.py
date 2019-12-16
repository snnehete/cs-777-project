#Task - predict trip duration using linear regression

from __future__ import print_function
import sys
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.ml.feature import OneHotEncoderEstimator, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

if __name__ == "__main__":
    if len(sys.argv) != 3:
            print("Usage: term_project <data file> <zone lookup>", file=sys.stderr)
            exit(-1)

    spark = SparkSession.builder \
            .master("local") \
            .getOrCreate()

	#Load the data into a dataframe and drop irrelevant values
    dataset = str(sys.argv[1])
    data = spark.read.csv(dataset, header=True) \
            .drop('VendorID', 'passenger_count', 'RatecodeID', 'store_and_fwd_flag', 'payment_type', 'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge', 'total_amount', 'congestion_surcharge')\
            .dropna() \
            .dropDuplicates()
            
    #data cleaning
    data = data.withColumn('tpep_pickup_datetime', to_timestamp(data.tpep_pickup_datetime, 'yyyy-MM-dd HH:mm:ss')) \
            .withColumn('tpep_dropoff_datetime', to_timestamp(data.tpep_dropoff_datetime, 'yyyy-MM-dd HH:mm:ss')) \
            .withColumn('trip_distance', data.trip_distance.cast("float")) \
            .filter(data.trip_distance > 0.0) \
            .withColumn('PULocationID', data.PULocationID.cast("int")) \
            .filter((data.PULocationID >= 1) & (data.PULocationID < 264)) \
            .withColumn('DOLocationID', data.DOLocationID.cast("int")) \
            .filter((data.DOLocationID >= 1) & (data.DOLocationID < 264))
    
    #compute the target column and remove outliers (trips greater than 6 hours)
    data = data.withColumn('trip_duration', (col("tpep_dropoff_datetime").cast("long") - col("tpep_pickup_datetime").cast("long")))
    data = data.filter(data.trip_duration < 21600)
    
    #convert timestamps to individual values (day, month, hour, etc.)
    data = data.withColumn('pickup_month', month('tpep_pickup_datetime')) \
            .withColumn('pickup_day', dayofmonth('tpep_pickup_datetime')) \
            .withColumn('pickup_day_of_week', dayofweek('tpep_pickup_datetime')) \
            .withColumn('pickup_hour', hour('tpep_pickup_datetime'))
    data = data.drop('tpep_pickup_datetime', 'tpep_dropoff_datetime')
    
    #some exploratory data analysis
    #load the zone lookup tables
    zone_lookup = spark.read.csv(sys.argv[2], header=True)
    zone_lookup = zone_lookup.withColumn('LocationID', zone_lookup.LocationID.cast("int"))    
    #translating the pickup and dropoff ID to zone names
    zoned_trips = data.select('trip_duration', 'PULocationID', 'DOLocationID', 'trip_distance')
    zoned_trips = zoned_trips.join(zone_lookup, zoned_trips.PULocationID == zone_lookup.LocationID, 'inner') \
            .drop('PULocationID', 'LocationID') \
            .withColumnRenamed('Borough', 'pickup_borough') \
            .withColumnRenamed('Zone', 'pickup_zone') \
            .withColumnRenamed('service_zone', 'pickup_service_zone')
    zoned_trips = zoned_trips.join(zone_lookup, zoned_trips.DOLocationID == zone_lookup.LocationID, 'inner') \
            .drop('DOLocationID', 'LocationID') \
            .withColumnRenamed('Borough', 'dropoff_borough') \
            .withColumnRenamed('Zone', 'dropoff_zone') \
            .withColumnRenamed('service_zone', 'dropoff_service_zone') \
            .orderBy(zoned_trips.trip_duration.desc())
    zoned_trips.show(10)
    
    #one-hot encode all categorical values
    encoder = OneHotEncoderEstimator(inputCols=["PULocationID", "DOLocationID", "pickup_month", "pickup_day", "pickup_day_of_week", "pickup_hour"],
                                 outputCols=["pickup_location", "dropoff_location", "month", "day", "day_of_week", "hour"])
    model = encoder.fit(data)
    data = model.transform(data)
    data = data.drop("PULocationID", "DOLocationID", "pickup_month", "pickup_day", "pickup_day_of_week", "pickup_hour")
    
    #construct the feature vector for each trip
    vectorAssembler = VectorAssembler(inputCols = ['trip_distance', 'dropoff_location', 'day_of_week', 'hour', 'pickup_location', 'day', 'month'], outputCol = 'features')
    input_data = vectorAssembler.transform(data)
    input_data = input_data.select(['features', 'trip_duration'])
    
    #split the data into training and test sets
    test_train_split = input_data.randomSplit([0.8, 0.2], 71)
    train = test_train_split[0]
    test = test_train_split[1]
    
    #define and train the linear regression model
    model = LinearRegression(featuresCol='features', labelCol='trip_duration', maxIter=100, regParam=0.07, elasticNetParam=0.8)
    model = model.fit(train)
    trainingSummary = model.summary
    print("Training Data SD: %f" % train.agg(stddev(train.trip_duration)).collect()[0][0])
    print("Training RMSE: %f" % trainingSummary.rootMeanSquaredError)
    
    #predict test values
    predictions = model.transform(test)
    test_result = model.evaluate(test)
    print("Test Data SD: %f" % test.agg(stddev(test.trip_duration)).collect()[0][0])
    print("Test RMSE: %f" % test_result.rootMeanSquaredError)