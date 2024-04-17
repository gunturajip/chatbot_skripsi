stopword_list = []
with open("id.stopwords.02.01.2016.txt", "r") as file:
    stopword_list = [line.strip() for line in file]

print(stopword_list)