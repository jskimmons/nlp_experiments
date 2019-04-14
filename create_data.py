# first, want to create a long text file

import json
import os
import re

json_li = []

outer_dir = "data"
directories = ["general", "dev", "ai", "marketresearch", "random", "sales-gamification"]
for directory in directories:
	path = os.path.join(outer_dir, directory)
	for filename in os.listdir(path):
		if filename.endswith(".json"):
			fp = open(os.path.join(path, filename))
			json_li.append(json.load(fp))

data_fp = open('data/general_text.txt', 'w+')

for json_obj in json_li:
	for message in json_obj:
		message['text'] = re.sub(r'<(.)*>', '', message['text'], flags=re.MULTILINE)
		try:
			data_fp.write(message['text'])
			data_fp.write("\n")
		except:
			# print("Something went wrong reading this line: {}".format(message["text"]))
			continue

