from collections import namedtuple
Batch = namedtuple(
        "Batch", 
        ["target", "distractor", "left", "right", "sentence", "label"])
