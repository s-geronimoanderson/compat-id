#!/usr/bin/env python

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import lil_matrix

class Network:
    """An abstract network class."""
    def __init__(self, augmented=False, size=0):
        """Size is the process count."""
        self.size = size
        if augmented:
            self.augmented = True
            from AChax.ACGMatrix import ACGMatrix
            self.network = ACGMatrix(nranks=size,
                                     # TODO: Do this without density.
                                     dense=True)
        else:
            self.augmented = False
            #self.network = lil_matrix((size, size))
            self.network = np.zeros((size, size))

    def __str__(self):
        """Just pass-through for now."""
        return self.network.__str__()

    def send(self, source, destination, payload):
        """Simulate sending the payload from the source to the destination."""
        if self.augmented:
            self.network.add_transfer_edge(
                (source, destination),
                payload)
        else:
            self.network[source, destination] += payload

    def get_matrix(self):
        if self.augmented:
            matrix = self.network.get_matrix()
        else:
            matrix = self.network
        return matrix

    def broadcast(self, root=0, scale=0):
        """Broadcast from the given root to all other processes."""
        if self.augmented:
            from AChax.Broadcast import Broadcast
            actor = Broadcast()
            self.network = actor.generate(
                nprocs=self.size,
                params={'root': root, 'scale': scale})
            result = self
        else:
            sources = [root]
            destinations = [d for d in range(self.size) if d != root]
            result = self.many_to_many(destinations=destinations,
                                       payload=scale,
                                       sources=sources)
        return result
    
    def reduce(self, root=0, scale=0):
        """Send from all processes (except the root) to the root."""
        if self.augmented:
            from AChax.Reduce import Reduce
            actor = Reduce()
            self.network = actor.generate(
                nprocs=self.size,
                params={'root': root, 'scale': scale})
            result = self
        else:
            sources = [d for d in range(self.size) if d != root]
            destinations = [root]
            result = self.many_to_many(destinations=destinations,
                                       payload=scale,
                                       sources=sources)
        return result

    def load(self, file_name):
        """Load from file_name."""
        if self.augmented:
            self.network.load(file_name)
        else:
            from scipy import io
            self.network = io.mmread(file_name)
        return self

    def save(self, file_name, comment=None):
        """Write to file_name."""
        if self.augmented:
            self.network.save(file_name)
        else:
            from scipy import io
            io.mmwrite(file_name, self.get_matrix(), comment=comment)
        return

    def many_to_many(self, destinations=None, payload=0, sources=None):
        """Send from the given sources to the given destinations."""
        if self.augmented:
            pass
        else:
            # TODO: Tidy up this logic (too nested)
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
