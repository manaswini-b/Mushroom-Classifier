# Example of Naive Bayes implemented from Scratch in Python
import csv
import random
import math

POSSIBLE_VALUES_OF_ATTRIBUTES = [
	['e','p'],
    ['b', 'c', 'x', 'f', 'k', 's'],
    ['f', 'g', 'y', 's'],
    ['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y'],
    ['t', 'f'],
    ['a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's'],
    ['a', 'd', 'f', 'n'],
    ['c', 'w', 'd'],
    ['b', 'n'],
    ['k', 'n', 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y'],
    ['e', 't'],
    ['b', 'c', 'u', 'e', 'z', 'r', '?'],
    ['f', 'y', 'k', 's'],
    ['f', 'y', 'k', 's'],
    ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'],
    ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'],
    ['p', 'u'],
    ['n', 'o', 'w', 'y'],
    ['n', 'o', 't'],
    ['c', 'e', 'f', 'l', 'n', 'p', 's', 'z'],
    ['k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y'],
    ['a', 'c', 'n', 's', 'v', 'y'],
    ['g', 'l', 'm', 'p', 'u', 'w', 'd']
]

def loadCsv(filename):
	lines = csv.reader(open(filename))
	dataset = list(lines)
	for i in range(len(dataset)):
		for x in range(len(dataset[i])):
			#print(x)
			#print(POSSIBLE_VALUES_OF_ATTRIBUTES[x])
			dataset[i][x] = [a for a,b in enumerate(POSSIBLE_VALUES_OF_ATTRIBUTES[x]) if b == dataset[i][x]][0]+1
	return dataset

def splitDataset(dataset, splitRatio):
	trainSize = splitRatio
	trainSet = []
	index = 0
	copy = list(dataset)
	while len(trainSet) < trainSize:
		trainSet.append(copy.pop(index))
		index += 1
	return [trainSet, copy]

def separateByClass(dataset):
	separated = {}
	print(dataset)
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[0] not in separated):
			separated[vector[0]] = []
		separated[vector[0]].append(vector)
	return separated

def mean(numbers):
	return sum(numbers)/float(len(numbers))

def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[0]
	return summaries

def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.items():
		summaries[classValue] = summarize(instances)
	return summaries

def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.items():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			print(x,mean,stdev)
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities
			
def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i][0] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def main():
	filename = 'mushroom_data.data'
	splitRatio = 4000
	dataset = loadCsv(filename)
	trainingSet, testSet = splitDataset(dataset, splitRatio)
	#print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingSet), len(testSet))
	# prepare model
	summaries = summarizeByClass(trainingSet)
	# test model
	testSet_except_output = testSet
	for row in testSet_except_output:
		del row[1]
	predictions = getPredictions(summaries, testSet_except_output)
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: {0}%').format(accuracy)

main()