__version__ = "0.5.0"

#: Prototype grade. 0.5 rather than 0.1 because both loops, the bridge, the data
#: layer and the mixing rules are implemented and tested -- but the gold-standard
#: evidence for the central handoff does not exist, so no version here can be 1.0
#: until a measurement closes it. See README, 'What would make this 1.0'.
__all__ = ['observability', 'models', 'taint', 'propagate', 'state', 'pbpk',
           'calibrate', 'cell', 'corona', 'delta', 'worldmodel',
           'phases', 'patient', 'multimodal', 'critic', 'datasets', 'agents']
