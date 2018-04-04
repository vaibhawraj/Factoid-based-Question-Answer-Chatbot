# ScriptName : StanfordDataset.py
# Description : Loads and provide necessary methods to access and explore test
#				dataset
#
#
# Importing libraries
import json

class StanfordDataset:
	def __init__(self):
		trainingF = open('dataset/testingData.json','r')
		trainingData = trainingF.readline()
		trainingF.close()

		self.trainingDataJson = json.loads(trainingData)

		self.titles = []
		for i in range(0,len(self.trainingDataJson['data'])):
		    self.titles.append(self.trainingDataJson['data'][i]['title'])

	# Get Dataset topic by name
	# Input:
	#		topicName(str)	: Name of topic
	# Output:
	#		devData(dict)	: JSON of data on that topic
	def getTopic(self,topicName): 
		devTitle = topicName
		for index in range(0,len(self.titles)):
		    if devTitle == self.titles[index]:
		        break
		devData = self.trainingDataJson['data'][index]
		return devData

	# Get All listed question
	# Input:
	#		topicName(str)	: Name of topic
	# Output:
	#		questions(list)	: List of Questions
	def getAllQuestions(self,topicName):
		devData = self.getTopic(topicName)
		questions = []
		for index in range(0,len(devData['paragraphs'])):
		    p = devData['paragraphs'][index]
		    for qs in range(0,len(p['qas'])):
		        questions.append(p['qas'][qs]['question'])
		return questions

	# Get paragraphs for that topic
	# Input:
	#		topicName(str)		: Name of topic
	# Output:
	#		paragraphs(list)	: List of paragraphs
	def getParagraph(self,topicName):
		devData = self.getTopic(topicName)
		paragraphs = []
		for index in range(0,len(devData['paragraphs'])):
			paragraphs.append(devData['paragraphs'][index]['context'])
		return paragraphs