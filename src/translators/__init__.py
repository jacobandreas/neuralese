from belief import BeliefTranslator
import util

load = util.class_loader("translators", lambda c: c.translator.name)
