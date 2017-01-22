#!/usr/bin/env python2

import json
import os

for filename in os.listdir("."):
    with open(filename) as f:
        try:
            data = json.load(f)
        except ValueError as e:
            #print "couldn't load", filename
            continue

    success1 = False
    success2 = False
    for step in data:
        if isinstance(step, dict):
            continue
        if step[0]["message"] != "":
            success1 = True
        if step[1]["message"] != "":
            success2 = True

    if not (success1 or success2):
        print filename
