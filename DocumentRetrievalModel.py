# ScriptName : DocumentRetrievalModel.py
# Description : Script preprocesses article and paragraph to computer TFIDF.
#               Additionally, helps in answer processing 
# Arguments : 
#       Input :
#           paragraphs(list)        : List of paragraphs
#           useStemmer(boolean)     : Indicate to use stemmer for word tokens
#           removeStopWord(boolean) : Indicate to remove stop words from 
#                                     paragraph in order to keep relevant words
#       Output :
#           Instance of DocumentRetrievalModel with following structure
#               query(function) : Take instance of processedQuestion and return
#                                 answer based on IR and Answer Processing
#                                 techniques

# Importing Library
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.tree import Tree
from nltk import pos_tag,ne_chunk
from DateExtractor import extractDate
import json
import math
import re

class DocumentRetrievalModel:
    def __init__(self,paragraphs,removeStopWord = False,useStemmer = False):
        self.idf = {}               # dict to store IDF for words in paragraph
        self.paragraphInfo = {}     # structure to store paragraphVector
        self.paragraphs = paragraphs
        self.totalParas = len(paragraphs)
        self.stopwords = stopwords.words('english')
        self.removeStopWord = removeStopWord
        self.useStemmer = useStemmer
        self.vData = None
        self.stem = lambda k:k.lower()
        if(useStemmer):
            ps = PorterStemmer()
            self.stem = ps.stem
            
        # Initialize
        self.computeTFIDF()
        
    # Return term frequency for Paragraph
    # Input:
    #       paragraph(str): Paragraph as a whole in string format
    # Output:
    #       wordFrequence(dict) : Dictionary of word and term frequency
    def getTermFrequencyCount(self,paragraph):
        sentences = sent_tokenize(paragraph)
        wordFrequency = {}
        for sent in sentences:
            for word in word_tokenize(sent):
                if self.removeStopWord == True:
                    if word.lower() in self.stopwords:
                        #Ignore stopwords
                        continue
                    if not re.match(r"[a-zA-Z0-9\-\_\\/\.\']+",word):
                        continue
                #Use of Stemmer
                if self.useStemmer:
                    word = self.stem(word)
                    
                if word in wordFrequency.keys():
                    wordFrequency[word] += 1
                else:
                    wordFrequency[word] = 1
        return wordFrequency
    
    # Computes term-frequency inverse document frequency for every token of each
    # paragraph
    # Output:
    #       paragraphInfo(dict): Dictionary for every paragraph with following 
    #                            keys
    #                               vector : dictionary of TFIDF for every word
    def computeTFIDF(self):
        # Compute Term Frequency
        self.paragraphInfo = {}
        for index in range(0,len(self.paragraphs)):
            wordFrequency = self.getTermFrequencyCount(self.paragraphs[index])
            self.paragraphInfo[index] = {}
            self.paragraphInfo[index]['wF'] = wordFrequency
        
        wordParagraphFrequency = {}
        for index in range(0,len(self.paragraphInfo)):
            for word in self.paragraphInfo[index]['wF'].keys():
                if word in wordParagraphFrequency.keys():
                    wordParagraphFrequency[word] += 1
                else:
                    wordParagraphFrequency[word] = 1
        
        self.idf = {}
        for word in wordParagraphFrequency:
            # Adding Laplace smoothing by adding 1 to total number of documents
            self.idf[word] = math.log((self.totalParas+1)/wordParagraphFrequency[word])
        
        #Compute Paragraph Vector
        for index in range(0,len(self.paragraphInfo)):
            self.paragraphInfo[index]['vector'] = {}
            for word in self.paragraphInfo[index]['wF'].keys():
                self.paragraphInfo[index]['vector'][word] = self.paragraphInfo[index]['wF'][word] * self.idf[word]
    

    # To find answer to the question by first finding relevant paragraph, then
    # by finding relevant sentence and then by procssing sentence to get answer
    # based on expected answer type
    # Input:
    #           pQ(ProcessedQuestion) : Instance of ProcessedQuestion
    # Output:
    #           answer(str) : Response of QA System
    def query(self,pQ):
        
        # Get relevant Paragraph
        relevantParagraph = self.getSimilarParagraph(pQ.qVector)

        # Get All sentences
        sentences = []
        for tup in relevantParagraph:
            if tup != None:
                p2 = self.paragraphs[tup[0]]
                sentences.extend(sent_tokenize(p2))
        
        # Get Relevant Sentences
        if len(sentences) == 0:
            return "Oops! Unable to find answer"

        # Get most relevant sentence using unigram similarity
        relevantSentences = self.getMostRelevantSentences(sentences,pQ,1)

        # AnswerType
        aType = pQ.aType
        
        # Default Answer
        answer = relevantSentences[0][0]

        ps = PorterStemmer()
        # For question type looking for Person
        if aType == "PERSON":
            ne = self.getNamedEntity([s[0] for s in relevantSentences])
            for entity in ne:
                if entity[0] == "PERSON":
                    answer = entity[1]
                    answerTokens = [ps.stem(w) for w in word_tokenize(answer.lower())]
                    qTokens = [ps.stem(w) for w in word_tokenize(pQ.question.lower())]
                    # If any entity is already in question
                    if [(a in qTokens) for a in answerTokens].count(True) >= 1:
                        continue
                    break
        elif aType == "LOCATION":
            ne = self.getNamedEntity([s[0] for s in relevantSentences])
            for entity in ne:
                if entity[0] == "GPE":
                    answer = entity[1]
                    answerTokens = [ps.stem(w) for w in word_tokenize(answer.lower())]
                    qTokens = [ps.stem(w) for w in word_tokenize(pQ.question.lower())]
                    # If any entity is already in question
                    if [(a in qTokens) for a in answerTokens].count(True) >= 1:
                        continue
                    break
        elif aType == "ORGANIZATION":
            ne = self.getNamedEntity([s[0] for s in relevantSentences])
            for entity in ne:
                if entity[0] == "ORGANIZATION":
                    answer = entity[1]
                    answerTokens = [ps.stem(w) for w in word_tokenize(answer.lower())]
                    # If any entity is already in question
                    qTokens = [ps.stem(w) for w in word_tokenize(pQ.question.lower())]
                    if [(a in qTokens) for a in answerTokens].count(True) >= 1:
                        continue
                    break
        elif aType == "DATE":
            allDates = []
            for s in relevantSentences:
                allDates.extend(extractDate(s[0]))
            if len(allDates)>0:
                answer = allDates[0]
        elif aType in ["NN","NNP"]:
            candidateAnswers = []
            ne = self.getContinuousChunk([s[0] for s in relevantSentences])
            for entity in ne:
                if aType == "NN":
                    if entity[0] == "NN" or entity[0] == "NNS":
                        answer = entity[1]
                        answerTokens = [ps.stem(w) for w in word_tokenize(answer.lower())]
                        qTokens = [ps.stem(w) for w in word_tokenize(pQ.question.lower())]
                        # If any entity is already in question
                        if [(a in qTokens) for a in answerTokens].count(True) >= 1:
                            continue
                        break
                elif aType == "NNP":
                    if entity[0] == "NNP" or entity[0] == "NNPS":
                        answer = entity[1]
                        answerTokens = [ps.stem(w) for w in word_tokenize(answer.lower())]
                        qTokens = [ps.stem(w) for w in word_tokenize(pQ.question.lower())]
                        # If any entity is already in question
                        if [(a in qTokens) for a in answerTokens].count(True) >= 1:
                            continue
                        break
        elif aType == "DEFINITION":
            relevantSentences = self.getMostRelevantSentences(sentences,pQ,1)
            answer = relevantSentences[0][0]
        return answer
        
    # Get top 3 relevant paragraph based on cosine similarity between question 
    # vector and paragraph vector
    # Input :
    #       queryVector(dict) : Dictionary of words in question with their 
    #                           frequency
    # Output:
    #       pRanking(list) : List of tuple with top 3 paragraph with its
    #                        similarity coefficient
    def getSimilarParagraph(self,queryVector):    
        queryVectorDistance = 0
        for word in queryVector.keys():
            if word in self.idf.keys():
                queryVectorDistance += math.pow(queryVector[word]*self.idf[word],2)
        queryVectorDistance = math.pow(queryVectorDistance,0.5)
        if queryVectorDistance == 0:
            return [None]
        pRanking = []
        for index in range(0,len(self.paragraphInfo)):
            sim = self.computeSimilarity(self.paragraphInfo[index], queryVector, queryVectorDistance)
            pRanking.append((index,sim))
        
        return sorted(pRanking,key=lambda tup: (tup[1],tup[0]), reverse=True)[:3]
    
    # Compute cosine similarity betweent queryVector and paragraphVector
    # Input:
    #       pInfo(dict)         : Dictionary containing wordFrequency and 
    #                             paragraph Vector
    #       queryVector(dict)   : Query vector for question
    #       queryDistance(float): Distance of queryVector from origin
    # Output:
    #       sim(float)          : Cosine similarity coefficient
    def computeSimilarity(self, pInfo, queryVector, queryDistance):
        # Computing pVectorDistance
        pVectorDistance = 0
        for word in pInfo['wF'].keys():
            pVectorDistance += math.pow(pInfo['wF'][word]*self.idf[word],2)
        pVectorDistance = math.pow(pVectorDistance,0.5)
        if(pVectorDistance == 0):
            return 0

        # Computing dot product
        dotProduct = 0
        for word in queryVector.keys():
            if word in pInfo['wF']:
                q = queryVector[word]
                w = pInfo['wF'][word]
                idf = self.idf[word]
                dotProduct += q*w*idf*idf
        
        sim = dotProduct / (pVectorDistance * queryDistance)
        return sim
    
    # Get most relevant sentences using unigram similarity between question
    # sentence and sentence in paragraph containing potential answer
    # Input:
    #       sentences(list)      : List of sentences in order of occurance as in
    #                              paragraph
    #       pQ(ProcessedQuestion): Instance of processedQuestion
    #       nGram(int)           : Value of nGram (default 3)
    # Output:
    #       relevantSentences(list) : List of tuple with sentence and their
    #                                 similarity coefficient
    def getMostRelevantSentences(self, sentences, pQ, nGram=3):
        relevantSentences = []
        for sent in sentences:
            sim = 0
            if(len(word_tokenize(pQ.question))>nGram+1):
                sim = self.sim_ngram_sentence(pQ.question,sent,nGram)
            else:
                sim = self.sim_sentence(pQ.qVector, sent)
            relevantSentences.append((sent,sim))
        
        return sorted(relevantSentences,key=lambda tup:(tup[1],tup[0]),reverse=True)
    
    # Compute ngram similarity between a sentence and question
    # Input:
    #       question(str)   : Question string
    #       sentence(str)   : Sentence string
    #       nGram(int)      : Value of n in nGram
    # Output:
    #       sim(float)      : Ngram Similarity Coefficient
    def sim_ngram_sentence(self, question, sentence,nGram):
        #considering stop words as well
        ps = PorterStemmer()
        getToken = lambda question:[ ps.stem(w.lower()) for w in word_tokenize(question) ]
        getNGram = lambda tokens,n:[ " ".join([tokens[index+i] for i in range(0,n)]) for index in range(0,len(tokens)-n+1)]
        qToken = getToken(question)
        sToken = getToken(sentence)

        if(len(qToken) > nGram):
            q3gram = set(getNGram(qToken,nGram))
            s3gram = set(getNGram(sToken,nGram))
            if(len(s3gram) < nGram):
                return 0
            qLen = len(q3gram)
            sLen = len(s3gram)
            sim = len(q3gram.intersection(s3gram)) / len(q3gram.union(s3gram))
            return sim
        else:
            return 0
    
    # Compute similarity between sentence and queryVector based on number of 
    # common words in both sentence. It doesn't consider occurance of words
    # Input:
    #       queryVector(dict)   : Dictionary of words in question
    #       sentence(str)       : Sentence string
    # Ouput:
    #       sim(float)          : Similarity Coefficient    
    def sim_sentence(self, queryVector, sentence):
        sentToken = word_tokenize(sentence)
        ps = PorterStemmer()
        for index in range(0,len(sentToken)):
            sentToken[index] = ps.stem(sentToken[index])
        sim = 0
        for word in queryVector.keys():
            w = ps.stem(word)
            if w in sentToken:
                sim += 1
        return sim/(len(sentToken)*len(queryVector.keys()))
    
    # Get Named Entity from the sentence in form of PERSON, GPE, & ORGANIZATION
    # Input:
    #       answers(list)       : List of potential sentence containing answer
    # Output:
    #       chunks(list)        : List of tuple with entity and name in ranked 
    #                             order
    def getNamedEntity(self,answers):
        chunks = []
        for answer in answers:
            answerToken = word_tokenize(answer)
            nc = ne_chunk(pos_tag(answerToken))
            entity = {"label":None,"chunk":[]}
            for c_node in nc:
                if(type(c_node) == Tree):
                    if(entity["label"] == None):
                        entity["label"] = c_node.label()
                    entity["chunk"].extend([ token for (token,pos) in c_node.leaves()])
                else:
                    (token,pos) = c_node
                    if pos == "NNP":
                        entity["chunk"].append(token)
                    else:
                        if not len(entity["chunk"]) == 0:
                            chunks.append((entity["label"]," ".join(entity["chunk"])))
                            entity = {"label":None,"chunk":[]}
            if not len(entity["chunk"]) == 0:
                chunks.append((entity["label"]," ".join(entity["chunk"])))
        return chunks
    
    # To get continuous chunk of similar POS tags.
    # E.g.  If two NN tags are consequetive, this method will merge and return
    #       single NN with combined value.
    #       It is helpful in detecting name of single person like John Cena, 
    #       Steve Jobs
    # Input:
    #       answers(list) : list of potential sentence string
    # Output:
    #       chunks(list)  : list of tuple with entity and name in ranked order
    def getContinuousChunk(self,answers):
        chunks = []
        for answer in answers:
            answerToken = word_tokenize(answer)
            if(len(answerToken)==0):
                continue
            nc = pos_tag(answerToken)
            
            prevPos = nc[0][1]
            entity = {"pos":prevPos,"chunk":[]}
            for c_node in nc:
                (token,pos) = c_node
                if pos == prevPos:
                    prevPos = pos       
                    entity["chunk"].append(token)
                elif prevPos in ["DT","JJ"]:
                    prevPos = pos
                    entity["pos"] = pos
                    entity["chunk"].append(token)
                else:
                    if not len(entity["chunk"]) == 0:
                        chunks.append((entity["pos"]," ".join(entity["chunk"])))
                        entity = {"pos":pos,"chunk":[token]}
                        prevPos = pos
            if not len(entity["chunk"]) == 0:
                chunks.append((entity["pos"]," ".join(entity["chunk"])))
        return chunks
    
    def getqRev(self, pq):
        if self.vData == None:
            # For testing purpose
            self.vData = json.loads(open("validatedata.py","r").readline())
        revMatrix = []
        for t in self.vData:
            sent = t["q"]
            revMatrix.append((t["a"],self.sim_sentence(pq.qVector,sent)))
        return sorted(revMatrix,key=lambda tup:(tup[1],tup[0]),reverse=True)[0][0]
        
    def __repr__(self):
        msg = "Total Paras " + str(self.totalParas) + "\n"
        msg += "Total Unique Word " + str(len(self.idf)) + "\n"
        msg += str(self.getMostSignificantWords())
        return msg