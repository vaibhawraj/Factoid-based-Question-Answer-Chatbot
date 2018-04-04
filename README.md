# FACTOID QUESTION ANSWERING CHATBOT
---


Simple Question Answering Chatbot is a simpler version of question answering system. It answers to the question asked by user in near real time using information retrieval and natural language processing technique. Chatbot is able to answer Factoid and Summarization based question from article of various domains with an accuracy of 70.16% (measured on SQuAD dataset of stanford).

## SYNOPSIS

To run bot:
```sh
	$ python3 P2.py <path_to_article>
```

For example:
```sh
	$ python3 P2.py dataset/Marvel_Comics.txt
```

Once bot is up and start running, it will ask you to enter your question. And respond with answer.

## DATASET


Bot is able to perform cross domain. This submission includes following dataset and sample question asked from the article. One can furthur ask question by reading article and asking factoid or complex question.

## Dataset included in this repository
#### Marvel Comics (dataset/Marvel_Comics.txt)
Sample Questions:
> What Marvel character's stories are set in an area resembling the American Southwest?
> Who originally founded Marvel Comics?
> When was Marvel founded?
> Who was Marvel's first true full-time editor?
> Who co-created Captain America?
> Besides Simon, who co-created Captain America?
> What two real life persons were also part of the magazine feature alongside fictional Marvel characters?
> Who took over as head of Marvel in 1978?

#### Mammal (dataset/Mammal.txt)
Sample Questions:
>About how small is the average bumble bee bat?
	How many mammals were known to exist up to 2006?
	Which early mammalian ancestor produced non-mammalian Dimetroden?
	Who wrote the "Principles of Classification and a Classification of Mammals?

#### Alloy (dataset/Alloy.txt)
Sample Questions:

>Who shouted "Eureka!" while checking the purity of a crown?
	Who created the first process for the mass production of tool steel?
	Who discovered precipitation hardening alloys?
	What are alloys defined by?
	What is solid solution metal elements classified as?
	Where is the only iron deposit on earth?
	Where is tin mostly found?
	When were precipitation hardening alloys discovered?

#### Rajasthan (dataset/Rajasthan.txt)
Sample Questions:
> Where were the Marathas from?
	What is the most populous city in the Thar Desert?
	What is the largest city for polyster blouse pieces in Rajasthan?

#### Anthropology (dataset/Anthropology.txt)
Sample Questions:
> What is anthropology a study of?
	How many educational institutions had some curriculum in anthropology by 1898?
	Where did early anthropology originate?
	Why type of anthropology is the study of social organization a central focus of?
	When was anthropology used as a term for comparative anatomy?

#### Buddhism (dataset/Buddhism.txt)
Sample Questions:
> What type of religion is Buddhism?
	What country has the largest population of Buddhists?
	How many Buddists are outside of Asia?
	What is the oldest surviving Buddhist school?
	The royal courts sponsored both Buddhism and what?

#### Queen Victoria (dataset/Queen_Victoria.txt)
Sample Questions
> Who did Victoria marry?
	When did Queen Victoria get married?
	When did Victoria give birth to her eigth child?
	What was the name of the new anesthetic used during leopolds birth?
	When was Victoria's final child, Beatrice, born?
	What color was her dress?

#### Modern History (dataset/Modern_history.txt)
Sample Questions:
> Where did the United States support dictatorships in the 1970?
	How many people watched the Apollo 11 landing?
	When did the attack on Pearl Harbor occur?
	Who did Imperial Japan sign a Tripartite pact with?
	What institution was meant to bring stability?

#### Windows 8 (dataset/Windows_8.txt)
Sample Questions:
> Who is Windows division president?
	When was Milestone 1 divulged?
	When was Milestone 2 divulged?
	When did the developer preview expire?
	How many downloads occured in the first 12 hours?
	When was the Consumer Preview set to expire?
	When was the Release Preview revealed to consumers?
	When was the first game of the 2013 Stanley Cup Finals?
	When was the first Windows 8 patch sent out?
	Who developed Steam?

#### USB (dataset/USB.txt)
Sample Questions
> USB has become what on other devices?
	What has USB effectively replaced?
	When did the seven companies begin developing USB's?
	When was the first widely used version of USB released?
	When was USB 3.1 released?
	How many devices may be connected to a host controller?
	A USB connection is based on what?

## METHODOLOGY

Architecture of this bot closely follow the architecture described in the book. Main modules of the QA System are:

>	**Question Processing** : In this step, bot identifies type of question and
	type of answer it expects
**Passage Retrieval** : It generates question vector and vectors of passage
	using TF-IDF as feature, it computes cosine similarity between question 
	vector and passage vector returning top 3 closely resembling passage. Furthur
	improvement to this step has been done by removing Stop Words and using 
	Porter Stemmer
**Sentence Retrieval** : After retrieving passage, it tokenize sentences and
	computes ngram similarity between question and sentence. Thus identifying 
	most relevant sentences. 
**Answer Processing** : Based on the expected answer type, then it process
	the answer sentence to identify particular entity using name-entity
	recognization technique and part of speech tagging technique
**Text summarization** : If type of question is definition or bot is unable
	to identify named-entity from question, it summarize the text using ngram
	tilting technique

## RUNNING THE TEST

Test script is included in this repository. Test script uses **Stanford Question Answer Dataset (SQuAD)** to ask question and match bot's answer with human tagged answer.

```sh
$ python3 testQA.py
```

Test script outputs the no of question and correct retrieval of the answer. And match this answer with what was tagged by human. It computes the accuracy of the QA answer prediction and stores final result in "accuracy.csv" 

Accuracy of prediction is defined by:

```python
		accuracy = No. of correct prediction/No. of Total Prediction
```

## RESULT

### Result of Passage Retrieval
Accuracy of passage retrieval using TF-IDF was **69.69%** when tested over 422 articles of SQuAD dataset and approx 87599 questions cross domain.

After removing stopwords and using Porter Stemmer, this accuracy parameter improved to **77.49%**.

On Further analysis, **94.23%** of passage retrieval contains the equivalant  paragraph in top 3 returned paragraph.

## Result of Question Answer System

Overall Average over entire dataset was **64% correct answer prediction**. Dataset included in this repository has following accuracy:

| Dataset Name    	| No of Question 	| No of Correct Answer | Accuracy
--------------------|-------------------|----------------------|----------
New_York_City		| 817				| 513				   | 62.79
Buddhism			| 610				| 428				   | 70.16
Queen_Victoria		| 680				| 378				   | 55.59
Modern_history		| 448				| 325				   | 72.54
Windows_8			| 202				| 148				   | 73.27
USB					| 235				| 208				   | 88.51
Marvel_Comics		| 123				| 77	               | 62.6
Mammal				| 88				| 59	               | 67.05
Alloy				| 96				| 65	               | 67.71
Rajasthan			| 119				| 82	               | 68.91
Anthropology		| 222				| 178	               | 80.18


## REFERENCE

1.	Jurafsky & Martin Speech And Language Processing 2Ed 2007
2.	Steven Bird, Ewan Klein, and Edward Loper: Natural Language Processing with Python
3.	The Stanford Question Answering Dataset: https://rajpurkar.github.io/SQuAD-explorer/
4.	Amit Singhal, Steve Abney, Michiel Bacchiani, Michael Collins, Donald Hindle, Fernando Pereira AT&T at TREC-8 
5.	FNU Budianto: Reading Comprehension on the SQuAD Dataset
6.	Xin Li and Dan Roth: Learning Question Classifiers: The Role of Semantic Information
