# -*- coding: utf-8 -*-

from pyspark.mllib.feature import HashingTF
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark import sql
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local").setAppName("SpamEmailDetection")
sc = SparkContext(conf = conf)
sqlContext = sql.SQLContext(sc)

spam = sc.textFile('emails_spam.txt').map(lambda e: e.split())

nospam = sc.textFile('emails_nospam.txt').map(lambda e: e.split())

tf = HashingTF(numFeatures = 100)

def trian_model(spam,nospam):
    spam_features = tf.transform(spam)
    spam_label = spam_features.map(lambda f: LabeledPoint(1,f))
    nospam_features = tf.transform(nospam)
    nospam_label = nospam_features.map(lambda f: LabeledPoint(0,f))
    train_data = spam_label.union(nospam_label)
    model = LogisticRegressionWithSGD.train(train_data)
    return model

model = trian_model(spam,nospam)

query = sc.textFile('query.txt')

query_words = query.map(lambda x: x.split())

query_predict_output = model.predict(tf.transform(query_words)).zip(query).toDF().show()

# +---+--------------------+
# | _1|                  _2|
# +---+--------------------+
# |  1|this is a year of...|
# |  1|you are the lucky...|
# |  1|Do not miss your ...|
# |  1|Get real money fa...|
# |  0|Dear Spark Learne...|
# |  0|Hi Mom, Apologies...|
# |  0|Wow, hey Fred, ju...|
# |  0|Hi Spark user lis...|
# |  1|Please do not rep...|
# |  0|Hi Mahmoud, Are y...|
# +---+--------------------+

def accuracy_score(m,data):
    predict = m.predict(data.map(lambda x: x.features))
    actual_and_predict = data.map(lambda x: x.label).zip(predict)
    accuracy = actual_and_predict.filter(lambda x: x[0]==x[1]).count()/float(data.count())
    return accuracy

accuracy_score(model, train_data) # overall acuracy is 100%

accuracy_score(model, spam_train) # spam email dection acuracy is 100%

accuracy_score(model, nospam_train) # nospam email dection acuracy is 100%

