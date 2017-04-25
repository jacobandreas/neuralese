#!/usr/bin/env python2

import json
import os

messages = 0
dialogues = 0
empty = 0
nocoop = 0

for filename in os.listdir("."):
    with open(filename) as f:
        try:
            data = json.load(f)
        except ValueError as e:
            #print "couldn't load", filename
            continue

    dialogues += 1
    success1 = False
    success2 = False
    for step in data:
        if isinstance(step, dict):
            continue
        if step[0]["message"] != "":
            success1 = True
            messages += 1
        if step[1]["message"] != "":
            success2 = True
            messages += 1

    if not (success1 or success2):
        empty += 1
    if not (success1 and success2):
        nocoop += 1

print "dialogues:", dialogues
print "messages: ", messages
print "empty:    ", empty
print "no coop:  ", nocoop
