# ChatBot

A NLP Chatbot which can be encorporated into a websight for seemless interaction with the user.

# Installation

To install this project follow these steps after cloning this repo in your local machine.

Install all the dependencies by

```bash
pip install nltk torch torchvision Flask
```

Then install the nltk package by

```bashpython
import nltk
nltk.download('punkt')
exit()
```

Then running the training file by doing

```bash
python train.py
```

# Optional 

After that, you can try and check if the training has been done correctly by running

```bash
python chat.py
```


If you can chat with Bob in your terminal now then it seems that all these steps have been done correctly. Now moving forward with the front end 

# FrontEnd

To fire up the front end and to chat with Bob, run this command.

```bash
python webApp.py
```

You should see your browser window with a tiny chat icon in lower right corner, if you click on it, it should show you the chat window.

Enjoy. :)

PS: To personalise the experience with Bob the Bot, you just have to make changes in newIntent.json file
