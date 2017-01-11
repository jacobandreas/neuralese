from recurrent_q import RecurrentQModel
from trivial import TrivialModel
import util

load = util.class_loader("models", lambda c: c.model.name)
