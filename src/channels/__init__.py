from gaussian import GaussianChannel
import util

load = util.class_loader("channels", lambda c: c.channel.name)
