'''

Data Science Research Bot (a.k.a SciBot)

Brian Byrne (bbyrne5)
Kevin Trinh (ktrinh1)

'''

import sys
import os

if __name__ == '__main__':
                
		textFolder		= 'data/text/'
		microsoftFolder = 'data/microsoft/'

		# Open Index

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
			paper['title_case']
			paper['title']
			paper['year']
			paper['conf']
			paper['cid']








