"""MaleCNS visual → 6-leg button trainer.

Learns a small set of synaptic *gains* on existing connectome edges so that
a labeled screen patch maps onto six VNC motor pools. Sparsity and signs stay
frozen. This is not a pixel MLP and does not train the mushroom body.
"""

__version__ = "0.1.0"

BUTTONS = ("LF", "RF", "LM", "RM", "LH", "RH")
