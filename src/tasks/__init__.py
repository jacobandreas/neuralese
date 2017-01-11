from abstract import AbstractRefTask
from cards import CardsTask
from color import ColorRefTask
from echo import EchoTask
from lock import LockTask
import util

load = util.class_loader("tasks", lambda c: c.task.name)
