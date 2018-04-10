# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 23:11:11 2018

@author: Nikhil Dikshit
"""

import pandas
import numpy
import tensorflow as tf
import matplotlib.pylab as plot
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

# spyder runs script multiple times on the same graph 
# [follow the model building section]
# we are supposedly adding our variable to the graph for each run
tf.reset_default_graph()
# solution 2: we could have used 'tf.get_variable_scope().reuse_variables() in the with the LSTM Unit in the script to reset the graph

# load the required layer-file for processing
def loadData(fromPath):
    dataFrame = pandas.read_csv(fromPath, header = None, sep=",")
    return dataFrame

# pre-processing of the data
def processData(sheet):
    # dropping the duplicates
    count = len(sheet)
    print("\nTotal Instances in the Dataset:", count)
    sheet.drop_duplicates(subset = None, inplace = True)
    newCount = len(sheet)
    print("\nTotal Instances after dropping duplicates:", newCount, "\n")
    
    # encoding 'normal' as '1' and 'abnormal/attack' as '0'
    dataFrameX = sheet.drop(sheet.columns[34], axis = 1, inplace = False)
    dataFrameY = sheet.drop(sheet.columns[0:34], axis = 1, inplace = False)
    dataFrameX[dataFrameX.columns[1:4]] = dataFrameX[dataFrameX.columns[1:4]].stack().rank(method = 'dense').unstack()
    
    print("Total Normal and Attack labels after dropping duplicates:")
    dataFrameY[dataFrameY[34] != 'normal'] = 0
    dataFrameY[dataFrameY[34] == 'normal'] = 1
    print(dataFrameY[34].value_counts())
    print("\n")

    # label-based indexing
    dataFrameY.columns = ["y1"]
    dataFrameY.loc[:,('y2')] = dataFrameY['y1'] == 0
    dataFrameY.loc[:,('y2')] = dataFrameY['y2'].astype(int)
    
    return dataFrameX, dataFrameY

# selection of features
def FeatureSelection(newX, newY):
    labels = numpy.array(newY).astype(int)
    x = numpy.array(newX)
    
    # RandomForestClassifier is used to select features with high importance-scores
    classifier = RandomForestClassifier(random_state = 0)
    classifier.fit(x, labels)
    rankings = classifier.feature_importances_
    
    # understanding the features importances graphically
    listFeatures = numpy.argsort(rankings)[::-1]
    std = numpy.std([tree.feature_importances_ for tree in classifier.estimators_], axis = 0)
    plot.figure(figsize = (10, 5))
    plot.title("Feature Importances (at y-axis) versus Features IDs (at x-axis)")
    plot.bar(range(x.shape[1]), rankings[listFeatures], color = "r", yerr=std[listFeatures], align = "center")
    plot.xticks(range(x.shape[1]), listFeatures)
    plot.xlim([-1, x.shape[1]])
    plot.show()
    
    # ranking the features according to their importance-scores   
    print("\nFeature Ranking -\n")
    for f in range(x.shape[1]):
        print("%d. Feature %d: %f" % (f + 1, listFeatures[f], rankings[listFeatures[f]]))
        
    # reducing features to a desired count (usually selecting the top features)
    # selecting multiple rows at once from the dataframe
    newX = newX.iloc[:, classifier.feature_importances_.argsort()[::-1][:2]]

    # converting Pandas dataframe to a Numpy array
    xFinal = newX.as_matrix()
    yFinal = labels
    return xFinal, yFinal

# breakpoint
# print("Nikhil")

layerFile = "C:/Users/Nikhil Dikshit/Desktop/ISTRAC/Output Files/Instances.txt"
dataFrame = loadData(layerFile)

processedX, processedY = processData(dataFrame)
print("Plotting all the features on graph:")

reducX,reducY = FeatureSelection(processedX, processedY)

# dividing the dataset into 80% training data and 20% testing data
trainingX = reducX[:3600]
trainingY = reducY[:3600]
testingX = reducX[3601:5000]
testingY = reducY[3601:5000]

# breakpoint
# print("\nNikhil")

print("\nTrainX shape is:", trainingX.shape)
print("TrainY shape is:", trainingY.shape)

print("\nTestX shape is:", testingX.shape)
print("TestY shape is:", testingY.shape)

# converting array of input features to a dataframe
semiTrainX = pandas.DataFrame(trainingX)
semiTestX = pandas.DataFrame(testingX)
scaler = MinMaxScaler(feature_range=(0, 1))

# normalization of train-data features
scalerTrainData = scaler.fit(semiTrainX)
trainNorm = scalerTrainData.transform(semiTrainX)
trainNormX = pandas.DataFrame(trainNorm)

# normalization of test-data features
scaler_testdata = scaler.fit(semiTestX)
testNorm = scaler_testdata.transform(semiTestX)
testNormX = pandas.DataFrame(testNorm)

# hyper-parameters
learningRate = 0.005
# number of selected features 
inputFeatures = 2
# inputFeatures = trainNormX.shape[1]
displayStep = 100
trainingCycles = 1000
nClasses = 2
# number of LSTM units in the hidden layer
# set to 1 for the IOT setting (making it light-weight)
hiddenUnits = 1 
# number of time-steps to backpropagate
timeSteps = 40

# taking inputs for the model as [inputsSize,inputFeatures]
with tf.name_scope('input'):
    # batchSize is 'None' because 'inputSize' is not restricted
    x = tf.placeholder(tf.float64, shape = [None, timeSteps, inputFeatures], name = "xInput")
    y = tf.placeholder(tf.float64, shape = [None, nClasses], name = "yInput")

# weights are randomly initialized
with tf.name_scope("weights"):
    W = tf.Variable(tf.random_normal([hiddenUnits, nClasses]), name = "LayerWeights")

# biases are randomly initialized
with tf.name_scope("biases"):
    b = tf.Variable(tf.random_normal([nClasses]), name = "UnitBiases")

''' 
Input is a 3D tensor of size (batchSize, timeSteps, inputFeatures). Prior to building 
the model, we need to reshape the inputs from 2D tensor of size (batchSize, inputFeatures) 
to 3D tensor of size (batchSize, timeSteps, inputFeatures).
'''

# modifying data for the required time-steps
def rnn_data(data, timeSteps, labels = False):
    rnn_df = []
    for i in range(len(data) - timeSteps):
        if labels:
            try:
                rnn_df.append(data.iloc[i + timeSteps].as_matrix())
            except AttributeError:
                rnn_df.append(data.iloc[i + timeSteps])
        else:
            data_ = data.iloc[i: i + timeSteps].as_matrix()
            rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])

    return numpy.array(rnn_df, dtype = numpy.float32)

# modifying train data
trainDataX = pandas.DataFrame(trainNormX)
trainDataY = pandas.DataFrame(trainingY)
newTrainingX = rnn_data(trainDataX, timeSteps, labels = False)
newTrainingY = rnn_data(trainDataY,timeSteps, labels = True)

print("\nNew TrainX shape is:", newTrainingX.shape)
print("New TrainY shape is:", newTrainingY.shape)

# modifying test data
testDataX = pandas.DataFrame(testNormX)
testDataY = pandas.DataFrame(testingY)
newTestingX = rnn_data(testDataX, timeSteps, labels = False)
newTestingY = rnn_data(testDataY, timeSteps, labels = True)

print("\nNew TestX shape is:", newTestingX.shape)
print("New TestY shape is:", newTestingY.shape)
print("\n")

# building the model
# sequencing the inputs according to 'timeSteps'
seqTimeSteps = tf.unstack(x, timeSteps, axis = 1)
cell = tf.contrib.rnn.GRUCell(hiddenUnits)
with tf.variable_scope('LSTMCell'):
    # spyder runs script multiple times on the same graph 
    output = tf.contrib.rnn.static_rnn(cell, seqTimeSteps, dtype = tf.float64)
    # we are supposedly adding our variable to the graph for each run
    # tf.get_variable_scope().reuse_variables()
    # solution 2: we could have used 'tf.reset_default_graph()' in the beginning of the script to reset the graph

# linear activation
result = tf.add(tf.matmul(output[-1], tf.cast(W, tf.float64)), tf.cast(b, tf.float64))

# loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = result))
optimizer = tf.train.AdamOptimizer(learning_rate = learningRate).minimize(loss)

# training the model
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
for i in range (trainingCycles):
    _,c = session.run([optimizer,loss], feed_dict = {x:newTrainingX, y:newTrainingY})
    # displaying logs per epoch step
    if (i) % displayStep == 0:
        print("Loss for the Training Cycle", i, ":", session.run(loss, feed_dict = {x:newTrainingX,y:newTrainingY}))
correctPrediction = tf.equal(tf.argmax(result, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correctPrediction, 'float'))
print('\nAccuracy on test set:', accuracy.eval({x:newTestingX, y:newTestingY}))

# breakpoint
# print("Nikhil")

# evaluation metrics
predClass = session.run(tf.argmax(result, 1), feed_dict = {x:newTestingX, y:newTestingY})
labelsClass = session.run(tf.argmax(y, 1), feed_dict = {x:newTestingX, y:newTestingY})
confusion = tf.contrib.metrics.confusion_matrix(labelsClass, predClass, dtype = tf.int32)
confusionMatrix = session.run(confusion, feed_dict={x:newTestingX, y:newTestingY})
print ("\nConfusion Matrix:\n", confusionMatrix)

# plotting the confusion matrix
labels = ['Normal', 'Attack']
fig = plot.figure()  
ax = fig.add_subplot(111)
cax = ax.matshow(confusionMatrix)
plot.title('\nConfusion Matrix of the Classifier:\n')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plot.xlabel('Predicted')
plot.ylabel('True')
plot.show()

# metrics calculations
TP = confusion[0, 0]
FN = confusion[0, 1]
FP = confusion[1, 0]
TN = confusion[1, 1]

# accuracy
accuracyByCM = (TP+TN)/(TP+FP+TN+FN)
print ("\nAccuracy:", session.run(accuracyByCM, feed_dict = {x:newTestingX, y:newTestingY}))

# recall
recall = TP/(TP+FN)
print ("\nRecall:", session.run(recall, feed_dict = {x:newTestingX, y:newTestingY}))

# precision
precision = TP/(TP+FP)
print ("\nPrecision:", session.run(precision, feed_dict = {x:newTestingX, y:newTestingY}))

# f score
fScore = 2*((precision*recall)/(precision+recall))
print ("\nF Score:", session.run(fScore, {x:newTestingX, y:newTestingY}))

# false alarm rate
FAR = FP/(FP+TN)
print ("\nFalse Alarm Rate:", session.run(FAR, feed_dict = {x:newTestingX, y:newTestingY}))

# specificity
specificity = TN/(TN+FP)
print ("\nSpecificity:", session.run(specificity, feed_dict = {x:newTestingX, y:newTestingY}))

# efficiency
efficiency = recall/FAR
print("\nEfficiency: ",session.run(efficiency, feed_dict = {x:newTestingX, y:newTestingY}))

# breakpoint
# print("Nikhil")