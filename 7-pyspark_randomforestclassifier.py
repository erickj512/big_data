from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
import pandas as pd


dataset = load_iris()
df_dataset = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df_dataset['label'] = pd.Series(dataset.target)
print(df_dataset.head())

sc = SparkContext().getOrCreate()
sqlContext = SQLContext(sc)
data = sqlContext.createDataFrame(df_dataset)
print(data.printSchema())

features = dataset.feature_names
va = VectorAssembler(inputCols=features, outputCol='features')
va_df = va.transform(data)
va_df = va_df.select(['features', 'label'])
va_df.show(5)
(train, test) = va_df.randomSplit([0.8, 0.2])

rfc = RandomForestClassifier(featuresCol="features", labelCol="label")
rfc = rfc.fit(train)

pred = rfc.transform(test)
pred.show(5)
evaluator=MulticlassClassificationEvaluator(predictionCol="prediction")
acc = evaluator.evaluate(pred)
print("Prediction Accuracy: ", acc)

y_pred=pred.select("prediction").collect()
y_orig=pred.select("label").collect()
cm = confusion_matrix(y_orig, y_pred)
print("Confusion Matrix:")
print(cm)

sc.stop()
