import fasttext
import glob
import csv

model = fasttext.load_model("/workspace/datasets/fasttext/title_model_100.bin")
# opening the file in read mode
my_file = open("/workspace/datasets/fasttext/top_words.txt", "r")
  
# reading the file
data = my_file.read()
  
# replacing end splitting the text 
# when newline ('\n') is seen.
data_into_list = data.split("\n")
my_file.close()
nearest_neighbors = {}
for i in data_into_list:
    nearest_neighbors[i] = model.get_nearest_neighbors(i)
csv_file = []
for key, value in nearest_neighbors.items():
    for i in value:
        if i[0] > 0.75:
            key +=','+i[1]
    csv_file.append(key)
with open("/workspace/datasets/fasttext/synonyms.csv", 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(csv_file)