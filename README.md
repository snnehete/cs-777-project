# Taxi Trip Duration Prediction using Linear Regression

The data subdirectory contains the associated dataset. The docs subdirectory contains the report and supporting files. The file term_project.py is the main file of the project to be run in pyspark.  

# How to run  

First, download the data subdirectory and unzip the [dataset.zip](data/dataset.zip) file into the data subdirectory. Alternatively, the dataset may be downloaded from [Google Cloud](https://storage.cloud.google.com/shrunkhala_nehete_a3/dataset.csv?showFTMessage=false).

Then the project can be executed by submitting the task to spark-submit. It takes two arguments - the location of the dataset and the location of taxi_zone_lookup.csv. In this case, it will be executed as:

```python

spark-submit term_project.py data/dataset.csv data/taxi_zone_lookup.csv 

```
