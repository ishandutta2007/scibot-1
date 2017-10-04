'''

Data Science Research Bot (a.k.a SciBot)

Brian Byrne (bbyrne5)
Kevin Trinh (ktrinh1)

'''

import sys
import os

def initializeStructure():

	textFolder		= 'data/text/'
	microsoftFolder = 'data/microsoft/'

	# Open Index ----------------------------------------

	indexFile = open(microsoftFolder + 'index.txt', 'r')

	index = {}

	for line in indexFile:
		
		l = line.split('\t')

		folderName 	= l[0]
		filename	= l[1]
		pid			= l[2]
		title		= l[3]

		i  = {}
		i['folderName']	= folderName
		i['filename']	= filename
		i['title']		= title

		index[pid]		= i

	indexFile.close()

	# Open Papers File ---------------------------------

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
		paper['title_case'] = title_case
		paper['title']		= title
		paper['year']		= year
		paper['conf']		= conf
		paper['cid']		= cid

		papers[pid]			= paper

	papersFile.close()

	# Open Paper Keywords File ----------------------------

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

	# Open Affiliations File -------------------------------

	affiliationsFile = open(microsoftFolder + 'PaperAuthorAffiliations.txt', 'r')

	affiliations = {}

	for line in affiliationsFile:

		l = line.rstrip().split('\t')

		pid 		= l[0]
		aid			= l[1]
		fid			= l[2]
		aff 		= l[4]
		sid			= l[5]

		if pid not in affiliations:
			affiliations[pid] = []

		affiliation = {}
		affiliation['aid'] = aid
		affiliation['fid'] = fid
		affiliation['aff'] = aff
		affiliation['sid'] = sid

		affiliations[pid].append(affiliation)

	affiliationsFile.close()

	# Open Authors File --------------------------------------

	authorsFile	= open(microsoftFolder + 'Authors.txt', 'r')

	authors = {}

	for line in authorsFile:

		l = line.rstrip().split('\t')

		aid 	= l[0]
		aut 	= l[1]

		authors[aid] = aut

	authorsFile.close()

	# Consolidate Data into Papers structure ----------------

	for key, value in papers.items():

		if key in index:

			papers[key]['folderName'] 	= index[key]['folderName']
			papers[key]['filename']		= index[key]['filename']

		if key in paperKeywords:

			papers[key]['keywords']		= paperKeywords[key]

		if key in affiliations:

			papers[key]['affiliations']	= affiliations[key]

	return papers, authors

if __name__ == '__main__':

	
	papers, authors = initializeStructure()

	for key, value in authors.items():

		print(key)
		print(value)

