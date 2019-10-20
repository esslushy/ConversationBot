# Conversation Bot

## Description

The goal of this independent study is to make a more realistic conversation bot by teaching it to understand the entirety of a conversation and not just the previous line fed into it. By doing this, I am hoping to achieve the goal of tricking a human into believing they are speaking with another human and not a machine.  
  
This repository will cover the creation of a working chatbot, possible attempts to improve that same chatbot, and multiple styles of training to see which version is the most effective. The end goal will be a model able to learn new vocabulary words from context as well as learn while it is talking to humans. I plan to experiment with multiple models of training such as having it talk to itself and using a discriminator to tell if the conversation is realistic, have it attempt to just produce results that are more realistic by again using a discriminator to give it a realism rating, and attempting to let it talk to humans over long periods of time to see how it develops. I may do one type of training for each model or set it up so that some models use 2 or all 3 types of training to see how effective it is.

## Set Up

1. If desired, create a python environment to encompass the libraries if you don't want to change your global python settings. If you don't know what this is, or are comfortable with the libraries being globally installed proceed onto the next step

2. Install the necessary libraries outlined in the `requirements.txt` file using the command `pip install -r requirements.txt`

3. Run any of the capitalized files and they should work automatically. Files labeled `settings.py` contain the settings for the folder which they reside in. Modifying these files will modify the model. Make sure you know the effect before attempting to modify these files.
