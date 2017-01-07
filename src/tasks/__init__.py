from color import ColorRefTask
from lock import LockTask
from echo import EchoTask
import util

load = util.class_loader("tasks", lambda c: c.task.name)
