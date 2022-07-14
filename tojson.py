#A program to convert CSV file to json format
#the CSV file should be in a 2 row form
import json
import csv
x=0
with open("C:/faq.csv","r") as f:#read the csv file from the system
    r= csv.reader(f) #reader
    next(r) #to eliminate the first read
    data={"intents":[]} #creating a disctionary name intents
    for row in r:#iterating through the dictionary to find Q&A
        x=x+1
        data["intents"].append({"tag":x,"patterns": row[0],"responses":row[1]})#appending the data into the dictionary as patterns and responses

with open("FAQIntents.json","w") as f:#creating the file and dumpinf all the data in the dictionary into json file
    json.dump(data,f,indent=2)

print("File Created")