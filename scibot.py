'''

Data Science Research Bot (a.k.a SciBot)

Brian Byrne (bbyrne5)
Kevin Trinh (ktrinh1)

'''

import sys
import os
import itertools
from fim import apriori, fpgrowth

# Task 1: Data preprocessing ===============================================

def dataPreprocessing():

	microsoftFolder = 'data/microsoft/'

	# Open Index

	indexFile = open(microsoftFolder + 'index.txt', 'r')

	index = {}

	for line in indexFile:
		l = line.split('\t')

		folder 		= l[0]
		filename	= l[1]
		pid			= l[2]
		title		= l[3]

		i  = {}
		i['folder']		= folder + '/'
		i['filename']	= filename + '.txt' # PDFID
		i['title']		= title

		index[pid]		= i

	indexFile.close()

	# Open papers file

	'''
	Note: Some Chinese characters appear in some of the
	paper titles and they will display weird in the terminal.

	'''

	papersFile = open(microsoftFolder + 'Papers.txt', 'r')

	papers = {}

	for line in papersFile:
		l = line.split('\t')

		pid 		= l[0]
		title_case	= l[1]
		title 		= l[2]
		year		= l[3]
		conf		= l[7]
		cid			= l[9]

		paper = {}
		paper['title_case'] = title_case # case sensitive title
		paper['title']		= title
		paper['year']		= year
		paper['conf']		= conf
		paper['cid']		= cid # conference id

		papers[pid]			= paper

	papersFile.close()

	# Open paper keywords file

	paperKeywordsFile = open(microsoftFolder + 'PaperKeywords.txt', 'r')

	paperKeywords = {}

	for line in paperKeywordsFile:
		l = line.split('\t')

		pid 		= l[0]
		keyword 	= l[1]

		if pid not in paperKeywords:
			paperKeywords[pid] = []
	
		paperKeywords[pid].append(keyword)

	paperKeywordsFile.close()

	# Open affiliations file

	affiliationsFile = open(microsoftFolder + 'PaperAuthorAffiliations.txt', 'r')

	affiliations = {}

	for line in affiliationsFile:
		l = line.rstrip().split('\t')

		pid = l[0]
		aid	= l[1]	# author id
		fid	= l[2]	# affiliation id
		aff = l[4] 	# normalized affiliation name
		sid	= l[5]	# authoer sequence number

		if pid not in affiliations:
			affiliations[pid] = []

		affiliation = {}
		affiliation['aid'] = aid
		affiliation['fid'] = fid
		affiliation['aff'] = aff
		affiliation['sid'] = sid

		affiliations[pid].append(affiliation)

	affiliationsFile.close()

	# Open authors file

	authorsFile	= open(microsoftFolder + 'Authors.txt', 'r')

	authors = {}

	for line in authorsFile:
		l = line.rstrip().split('\t')

		aid = l[0]
		aut = l[1]

		authors[aid] = aut

	authorsFile.close()

	# Consolidate data into papers object

	for key, value in papers.items():
		if key in index:
			papers[key]['folder'] 		= index[key]['folder']
			papers[key]['filename']		= index[key]['filename']

		if key in paperKeywords:
			papers[key]['keywords']		= paperKeywords[key]

		if key in affiliations:
			papers[key]['affiliations']	= affiliations[key]

			'''
			for affiliation in papers[key]['affiliations']:

				if affiliation['aid'] in authors:

					affiliation['aut'] = authors[affiliation['aid']]
			'''
	
	return papers, authors

# Task 2: Entity mining ====================================================
# Candidate generation and quality assessment

# Taken from ResponseBotDemo.html by Prof. Meng Jiang
def easy_tokenizer(text):

	ret = []
	for x in [',', '.', '--', '!', '?', ';', '(', ')', '/', '"', '_']:
		text = text.replace(x, ' '+x+' ')

	for word in text.split(' '):
		if word == '':
			continue

		ret.append(word.lower())

	return ret

def entityMining(papers):

	textFolder = 'data/text/'
	support = 9
	entities = {}

	# Create stopwords list
	stopwordsFile = open('stopwords.txt', 'r')
	stopwords = set()

	for line in stopwordsFile:
		word = line.strip('\r\n').lower()
		stopwords.add(word)

	stopwordsFile.close()

	# Add words to paper object that pass criteria
	for key, value in papers.items():
		if 'folder' in papers[key] and 'filename' in papers[key]:

			# Get candidates
			candidates = []
			dataFile = open(textFolder + papers[key]['folder'] + papers[key]['filename'])

			for line in dataFile:
				text = line.strip('\r\n')
				words = easy_tokenizer(text)
				candidates.append(words)

			dataFile.close()

			# Compute words dict
			wordDict = {}

			for words in candidates:
				for word in words:
					if word in stopwords or len(word) == 1 or word.isdigit():
						continue
					if word not in wordDict:
						wordDict[word] = 0
					wordDict[word] += 1

			# Compute bigrams
			bigrams = {}
			L = 0

			for words in candidates:
				n = len(words)
				L += n
				for i in range(0, n-1):
					if words[i] in wordDict and words[i+1] in wordDict:
						bigram = words[i] + '_' + words[i+1]
						if bigram not in bigrams:
							# bigram's count, first word's count, second word's count, significance score
							bigrams[bigram] = [0, wordDict[words[i]], wordDict[words[i+1]], 0.0]
						bigrams[bigram][0] += 1

			# Readjust bigrams scores
			for bigram in bigrams:
				bigrams[bigram][3] = (1.0 * bigrams[bigram][0] -  \
					1.0 * bigrams[bigram][1] * bigrams[bigram][2]/L) / \
					((1.0 * bigrams[bigram][0])**0.5)

			# Compute transactions
			bigramDict = {}

			for bigram in bigrams:
				if bigrams[bigram][0] > 1:
					first, second = bigram.split('_')
					if first not in bigramDict:
						bigramDict[first] = set()
					bigramDict[first].add(second)

			# Compute quality entities
			transactions = []
			for words in candidates:
				transaction = set() # set of words/bigrams
				n = len(words)
				i = 0
				while i < n:
					if words[i] in bigramDict and i+1 < n and words[i+1] in bigramDict[words[i]]:
						transaction.add(words[i] + '_' + words[i+1])
						i += 2
						continue
					if words[i] in stopwords or len(words[i]) == 1 or words[i].isdigit():
						i += 1
						continue
					transaction.add(words[i])
					i += 1
				transactions.append(list(transaction))

			patterns = fpgrowth(transactions, supp=-support)

			# Create entity list
			entity = []
			for pattern, support in sorted(patterns, key=lambda x:-x[1]):
				entity.append(pattern)

			entities[key] = entity

	return entities
	
# Task 3: Entity Typing ====================================================

def entityExtraction(papers):

	textFolder = 'data/text/'
	phrasesPerPaper = {}

	for key, value in papers.items():
		phrases = {}
		if 'folder' in papers[key] and 'filename' in papers[key]:
			dataFile = open(textFolder + papers[key]['folder'] + papers[key]['filename'])
			for line in dataFile:
				array = line.strip('\r\n').split(' ')
				n = len(array)
				if n < 5:
					continue
				for i in range(0, n-2):
					if array[i] == '(' and array[i+2] == ')':
						abbreviation = array[i+1]
						l = len(abbreviation)
						if l > 1 and abbreviation.isalpha():
							if i >= l and abbreviation.isupper():
								isValid = True
								for j in range(0, l):
									if not abbreviation[l-1-j].lower() == array[i-1-j][0].lower():
										isValid = False
									if isValid:
										phrase = ''
										for j in range(0, l):
											phrase = array[i-1-j] + ' ' + phrase
										phrase = phrase[0:-1].lower()
										if phrase not in phrases:
											phrases[phrase] = 0
										phrases[phrase] += 1
							if i >= l-1 and abbreviation[-1] == 's' and array[i-1][-1] == 's' and abbreviation[0:-1].isupper():
								isValid = True
								for j in range(1, l):
									if not abbreviation[l-1-j].lower() == array[i-j][0].lower():
										isValid = False
									if isValid:
										phrase = ''
										for j in range(1, l):
											phrase = array[i-j] + ' ' + phrase
										phrase = phrase[0:-1].lower()
										if phrase not in phrases:
											phrases[phrase] = 0
										phrases[phrase] += 1
		phrasesPerPaper[key] = phrases

	return phrasesPerPaper

def entityTyping(papers):
	
	textFolder = 'data/text/'

	phrasesPerPaper = entityExtraction(papers)

	nType 		= 4
	s_method 	= 'method algorithm model approach framework process scheme implementation procedure strategy architecture'
	s_problem 	= 'problem technique process system application task evaluation tool paradigm benchmark software'
	s_dataset 	= 'data dataset database'
	s_metric 	= 'value score measure metric function parameter'
	types 		= ['METHOD', 'PROBLEM', 'DATASET', 'METRIC']
	wordsets 	= [set(s_method.split(' ')), set(s_problem.split(' ')), set(s_dataset.split(' ')), set(s_metric.split(' '))]

	index, nIndex = [{}], 1

	for key, value in phrasesPerPaper.items():
		temp = ''
		for phrase, count in phrasesPerPaper[key].items():
			array = phrase.strip('\r\n').split('\t')
			phrase = array[0]
			words = phrase.split(' ')
			n = len(words)
			if n > nIndex:
				for i in range(nIndex, n):
					index.append({})
				nIndex = n
			temp = index[n-1]
			if n > 1:
				for i in range(0, n-1):
					word = words[i]
					if word not in temp:
						temp[word] = {}
					temp = temp[word]
				word = words[n-1]
			else:
				word = words[0]
			temp[word] = phrase

	nContext = 5
	phrases = {}
	for key, value in papers.items():
		if 'folder' in papers[key] and 'filename' in papers[key]:
			dataFile = open(textFolder + papers[key]['folder'] + papers[key]['filename'])
			for line in dataFile:
				words = line.strip('\r\n').split(' ')
				wordsLower = line.strip('\r\n').lower().split(' ')
				l = len(words)
				i = 0
				while i < l:
					isValid = False
					for j in range(min(nIndex, l-i), 0, -1):
						temp = index[j-1]
						k = 0
						while k < j and i+k < l:
							tempword = wordsLower[i+k]
							if tempword not in temp:
								break
							temp = temp[tempword]
							k += 1
						if k == j:
							phrase = temp
							if phrase not in phrases:
								phrases[phrase] = [0,[[0 for t in range(0, nType)] for c in range(0, nContext)]]
							phrases[phrase][0] += 1
							for c in range(0, nContext):
								if i-1-c >= 0:
									trigger = wordsLower[i-1-c]
									for t in range(0, nType):
										if trigger in wordsets[t]:
											phrases[phrase][1][c][t] += 1
								if i+k+c < l:
									trigger = wordsLower[i+k+c]
									for t in range(0, nType):
										if trigger in wordsets[t]:
											phrases[phrase][1][c][t] += 1
							isValid = True
							break
					if isValid:
						i += j
						continue
					i += 1

	s = 'ENTITY\tCOUNT'
	for c in range(0, nContext):
		s += '\tWINDOWSIZE' + str(c+1)
	print(s)
	for [phrase, [count, ctmatrix]] in sorted(phrases.items(), key=lambda x:-x[1][0]):
		s = phrase + '\t' + str(count)
		for c in range(0, nContext):
			maxv, maxt = -1, -1
			for t in range(0, nType):
				v = ctmatrix[c][t]
				if v > maxv:
					maxv = v
					maxt = t
			if maxv == 0:
				s += '\tN/A'
			else:
				s += '\t' + types[maxt]
				for t in range(0, nType):
					v = ctmatrix[c][t]
					if v == 0:
						continue
					s += ' ' + types[t][0] + types[1][-1] + ':' + str(v)
		print(s)

# Task 4: Collaboration Discovery ==========================================

def collaborationDiscovery(papers):

	support = 9
	frequentCollaborators = []

	allAuthorsPerPaper = []
	for key, value in papers.items():
		if 'affiliations' in papers[key]:
			authorsPerPaper = set()
			for affiliation in papers[key]['affiliations']:
				authorsPerPaper.add(authors[affiliation['aid']])
			allAuthorsPerPaper.append(authorsPerPaper)

	patterns = fpgrowth(allAuthorsPerPaper, supp=-support)

	for (pattern,support) in sorted(patterns,key=lambda x:-x[1]):
		if len(pattern) > 1:
			frequentCollaborators.append((pattern, support))

	return frequentCollaborators

# Task 5: Problem-method association mining ================================

def associationMining(papers):

	textFolder = 'data/text/'
	support = 9
	confidence = 10
	rules = {}

	# Create stopwords list
	stopwordsFile = open('stopwords.txt', 'r')
	stopwords = set()

	for line in stopwordsFile:
		word = line.strip('\r\n').lower()
		stopwords.add(word)

	stopwordsFile.close()

	transactions = []

	for key, value in papers.items():
		if 'folder' in papers[key] and 'filename' in papers[key]:

			# Get candidates
			candidates = []
			dataFile = open(textFolder + papers[key]['folder'] + papers[key]['filename'])

			for line in dataFile:
				text = line.strip('\r\n')
				words = easy_tokenizer(text)
				candidates.append(words)

			dataFile.close()

			# Compute words dict
			wordDict = {}

			for words in candidates:
				for word in words:
					if word in stopwords or len(word) == 1 or word.isdigit():
						continue
					if word not in wordDict:
						wordDict[word] = 0
					wordDict[word] += 1

			# Compute bigrams
			bigrams = {}
			L = 0

			for words in candidates:
				n = len(words)
				L += n
				for i in range(0, n-1):
					if words[i] in wordDict and words[i+1] in wordDict:
						bigram = words[i] + '_' + words[i+1]
						if bigram not in bigrams:
							# bigram's count, first word's count, second word's count, significance score
							bigrams[bigram] = [0, wordDict[words[i]], wordDict[words[i+1]], 0.0]
						bigrams[bigram][0] += 1

			# Readjust bigrams scores
			for bigram in bigrams:
				bigrams[bigram][3] = (1.0 * bigrams[bigram][0] -  \
					1.0 * bigrams[bigram][1] * bigrams[bigram][2]/L) / \
					((1.0 * bigrams[bigram][0])**0.5)

			# Compute transactions
			bigramDict = {}

			for bigram in bigrams:
				if bigrams[bigram][0] > 1:
					first, second = bigram.split('_')
					if first not in bigramDict:
						bigramDict[first] = set()
					bigramDict[first].add(second)

			# Compute quality entities
			transactions = []
			for words in candidates:
				transaction = set() # set of words/bigrams
				n = len(words)
				i = 0
				while i < n:
					if words[i] in bigramDict and i+1 < n and words[i+1] in bigramDict[words[i]]:
						transaction.add(words[i] + '_' + words[i+1])
						i += 2
						continue
					if words[i] in stopwords or len(words[i]) == 1 or words[i].isdigit():
						i += 1
						continue
					transaction.add(words[i])
					i += 1
				transactions.append(list(transaction))

	rules = apriori(transactions, target='r', supp=support, conf=confidence, report='sc')
	
	print '--------- One-to-Many Assocation Rules ------------'
	for left, right, support, confidence in sorted(rules, key=lambda x:x[0]):
		print left, '-->', right, support, confidence
	print 'Number of rules: ', len(rules)

# Task 6: Problem/method/author-to-conference classification ===============

def pma2cClassification(papers):

	pass

# Task 7: Paper clustering =================================================

def paperClustering(paper):

	pass

# Main Execution ===========================================================

if __name__ == '__main__':

	papers, authors = dataPreprocessing()

	# entities = entityMining(papers)
	# frequentCollaborators = collaborationDiscovery(papers)

	# entityTyping(papers)

	associationMining(papers)


 	'''
	i = 10
	for frequentCollaborator in frequentCollaborators:
		print(frequentCollaborator)
		i -= 1
		if i <= 0:
			break
	'''

	'''for key, value in papers.items():
		if 'words' in papers[key]:
			print(key)
			
			for key2, value2 in value.items():
				print(key2)
				print(value2)

			break

			
			for word, support in papers[key]['words'].items():

				print(word + ' ' + str(support))	
	'''