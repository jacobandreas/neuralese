from abstract import AbstractRefTask
from birds import BirdsRefTask
from color import ColorRefTask
from drive import DriveTask
import util

load = util.class_loader("tasks", lambda c: c.task.name)
