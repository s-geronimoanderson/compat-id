#!/usr/bin/env python

import numpy as np
from scipy.sparse import lil_matrix

class Network:
    """An abstract network class."""
    def __init__(self, augmented=False, size=0):
        """Size is the process count."""
        #if augmented:
        #    import AChax.ACGMatrix as ACGMatrix
        #self.network = lil_matrix((size, size))
        self.network = np.zeros((size, size))
        self.size = size

    def __str__(self):
        """Just pass-through for now."""
        return self.network.__str__()

    def send(self, source, destination, payload):
        """Simulate sending the payload from the source to the destination."""
        self.network[source, destination] += payload

    def as_numpy_array(self):
        return self.network

    def tolil(self, copy=False):
        result = self.network
        if copy:
            result = self.network.tolil(copy=True)
        return result

    def broadcast(self, root=0, scale=0):
        """Broadcast from the given root to all other processes."""
        sources = [root]
        destinations = [d for d in range(self.size) if d != root]
        return self.many_to_many(destinations=destinations,
                                 payload=scale,
                                 sources=sources)
    
    def reduce(self, root=0, scale=0):
        """Send from all processes (except the root) to the root."""
        sources = [d for d in range(self.size) if d != root]
        destinations = [root]
        return self.many_to_many(destinations=destinations,
                                 payload=scale,
                                 sources=sources)
    
    def many_to_many(self, destinations=None, payload=0, sources=None):
        """Send from the given sources to the given destinations."""
        if destinations is None:
            destinations = []
        if sources is None:
            sources = []
        for source in sources:
            if self.is_valid_process(source):
                for destination in destinations:
                    if self.is_valid_process(destination):
                        self.send(
                            source=source,
                            destination=destination,
                            payload=payload)
                    else:
                        raise IndexError
            else:
                raise IndexError
        return self

    def is_valid_process(self, process):
        return (0 <= process and process <= self.size)


# Run-as-script idiom.

if __name__ == "__main__":
    pass


# End.
