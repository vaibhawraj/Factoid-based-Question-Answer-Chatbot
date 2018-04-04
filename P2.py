# Author : Vaibhaw Raj
# Created on : Sun, Feb 11 2018
# Description : Entry point for Simple Question Answer Chatbot
# Program Argument :
#		datasetName = "Name of dataset text file" eg. "Beyonce.txt"
# Usage :
#		$ python3 P2.py dataset/IPod

print("Bot> Please wait, while I am loading my dependencies")
from DocumentRetrievalModel import DocumentRetrievalModel as DRM
from ProcessedQuestion import ProcessedQuestion as PQ
import re
import sys

if len(sys.argv) == 1:
	print("Bot> I need some reference to answer your question")
	print("Bot> Please! Rerun me using following syntax")
	print("\t\t$ python3 P2.py <datasetName>")
	print("Bot> You can find dataset name in \"dataset\" folder")
	print("Bot> Thanks! Bye")
	exit()

datasetName = sys.argv[1]
# Loading Dataset
try:
	datasetFile = open(datasetName,"r")
except FileNotFoundError:
	print("Bot> Oops! I am unable to locate \"" + datasetName + "\"")
	exit()

# Retrieving paragraphs : Assumption is that each paragraph in dataset is
# separated by new line character
paragraphs = []
for para in datasetFile.readlines():
	if(len(para.strip()) > 0):
		paragraphs.append(para.strip())

# Processing Paragraphs
drm = DRM(paragraphs,True,True)

print("Bot> Hey! I am ready. Ask me factoid based questions only :P")
print("Bot> You can say me Bye anytime you want")

# Greet Pattern
greetPattern = re.compile("^\ *((hi+)|((good\ )?morning|evening|afternoon)|(he((llo)|y+)))\ *$",re.IGNORECASE)

isActive = True
while isActive:
	userQuery = input("You> ")
	if(not len(userQuery)>0):
		print("Bot> You need to ask something")

	elif greetPattern.findall(userQuery):
		response = "Hello!"
	elif userQuery.strip().lower() == "bye":
		response = "Bye Bye!"
		isActive = False
	else:
		# Proocess Question
		pq = PQ(userQuery,True,False,True)

		# Get Response From Bot
		response =drm.query(pq)
	print("Bot>",response)