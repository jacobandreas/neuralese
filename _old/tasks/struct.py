from collections import namedtuple

Batch = namedtuple(
        "Batch", 
        ["target", "distractor", "left", "right", "sentence", "label"])

Transition = namedtuple("Transition", ["s1", "m1", "a", "s2", "m2", "r", "term"])
