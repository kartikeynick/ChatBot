# Bob the Bot (Chat)

A NLP Chatbot which can be incorporated into a website for seamless interaction with the user.

# Installation

To install this project follow these steps after cloning this repo in your local machine.

Install all the dependencies by

```bash
pip install nltk torch torchvision Flask
```

Then install the nltk package by

```bash
python
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

Now it should show a link for the local host in the terminal, by "Running on (link)" click on the link or copy paste the link in your browser and it will take you to the browser window. Now you should be able to see a tiny chat icon in lower right corner of your browser window, if you click on it, it should show you the chat window where you can chat with Bob.

Enjoy. :)

PS: To personalize the experience with Bob the Bot, you just have to make changes in newIntent.json file


