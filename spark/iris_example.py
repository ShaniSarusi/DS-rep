# $example on$
import os
from os import chdir
from os.path import join, sep
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession

# Set working directory
chdir(join('C:', sep, 'Users', 'zwaks', 'Documents', 'Workspaces', 'GitHub', 'DataScientists'))
#os.environ["PYTHONPATH"] = "C:\spark\spark-2.2.0-bin-hadoop2.7\python\lib\py4j-0.10.3-src.zip;C:\spark\spark-2.2.0-bin-hadoop2.7\python;C:\Program Files\JetBrains\PyCharm Community Edition 2017.1.3\helpers\pydev"

spark = SparkSession\
    .builder\
    .appName("iris") \
    .getOrCreate()
# .config("spark.sql.shuffle.partitions", "3") \

# Load and parse the data file, converting it to a DataFrame.
data_path = join('spark', 'iris.csv')
df = spark.read.format('com.databricks.spark.csv').option("header", "True").load(data_path)

feature_cols = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]
for feature in feature_cols:
    df = df.withColumn(feature, df[feature].cast("double"))


assembler = VectorAssembler(inputCols=["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"], outputCol="features")
df = assembler.transform(df)
df = df.select([c for c in df.columns if c not in feature_cols])


# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol="Species", outputCol="indexedLabel").fit(df)

# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(df)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = df.randomSplit([0.7, 0.3])

# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="features", numTrees=10)

# Convert indexed labels back to original labels.
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel", labels=labelIndexer.labels)

# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, rf, labelConverter])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("predictedLabel", "Species", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

rfModel = model.stages[2]
print(rfModel)  # summary only
# $example off$

spark.stop()