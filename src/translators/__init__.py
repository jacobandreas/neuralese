from disc_belief import DiscBeliefTranslator
from gen_belief import GenBeliefTranslator
import util

load = util.class_loader("translators", lambda c: c.translator.name)
