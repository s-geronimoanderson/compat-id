#!/usr/bin/env python

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import lil_matrix

class Network:
    """An abstract network class."""
    def __init__(self,
            augmented=False,
            communicator_count=0,
            process_count=0):
        """Size is the process count plus the communicator count."""
        self.communicator_count = communicator_count
        self.process_count = process_count
        self.size = size = communicator_count + process_count
        if augmented:
            # TODO: Can I take out this case, now that self.network is an
            # nparray again?
            self.augmented = True
            from AChax.ACGMatrix import ACGMatrix
            self.network = ACGMatrix(nranks=size,
                                     # TODO: Do this without density?
                                     dense=True).get_matrix()
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
        if False:
        #if self.augmented:
            matrix = self.network.get_matrix()
        else:
            matrix = self.network
        return matrix

    def broadcast(self, root, scale):
        """Broadcast from the given root to all other processes."""
        if self.augmented:
            from AChax.Broadcast import Broadcast as Actor
            params = {'root': root, 'scale': scale}

            # TODO: Generalize the below?
            actor = Actor()
            network = actor.generate(
                nprocs=self.process_count,
                params=params).get_matrix()
            self.network += np.imag(network)
            result = self
        else:
            sources = [root]
            destinations = [d for d in range(self.size) if d != root]
            result = self.many_to_many_old(destinations=destinations,
                                           payload=scale,
                                           sources=sources)
        return result

    def many_to_many(self, scale):
        """A general many-to-many pattern."""
        if self.augmented:
            from AChax.ManyToMany import ManyToMany as Actor
            params = {'scale': scale}

            # The below may be able to be generalized.
            actor = Actor()
            network = actor.generate(
                nprocs=self.process_count,
                params=params).get_matrix()

            coo = network.tocoo()
            resized_network = np.zeros((self.size, self.size))
            for row, col, value in zip(coo.row, coo.col, coo.data):
                resized_network[row][col] += value.imag

            self.network += resized_network
            result = self
        else:
            raise NotImplementedError
        return result

    def nn2d(self, dimensions, periodic, scale):
        """Two-dimensional, five-point nearest-neighbor."""
        if self.augmented:
            from AChax.NN2D05 import NN2D05 as Actor
            params = {'dims': dimensions,
                      'scale': scale,
                      'periodic': [periodic]*len(dimensions)}

            actor = Actor()
            network = actor.generate(
                nprocs=self.process_count,
                params=params).get_matrix()

            coo = network.tocoo()
            resized_network = np.zeros((self.size, self.size))
            for row, col, value in zip(coo.row, coo.col, coo.data):
                resized_network[row][col] += value.imag

            self.network += resized_network
            result = self
        else:
            raise NotImplementedError
        return result


    def nn3d(self, dimensions, periodic, scale):
        """Three-dimensional, seven-point nearest-neighbor."""
        # TODO: This is actually identical to nn2d, except for the import.
        if self.augmented:
            from AChax.NN3D07 import NN3D07 as Actor
            params = {'dims': dimensions,
                      'scale': scale,
                      'periodic': [periodic]*len(dimensions)}

            actor = Actor()
            network = actor.generate(
                nprocs=self.process_count,
                params=params).get_matrix()

            coo = network.tocoo()
            resized_network = np.zeros((self.size, self.size))
            for row, col, value in zip(coo.row, coo.col, coo.data):
                resized_network[row][col] += value.imag

            self.network += resized_network
            result = self
        else:
            raise NotImplementedError
        return result


    def reduce(self, root, scale):
        """Send from all processes (except the root) to the root."""
        if self.augmented:
            from AChax.Reduce import Reduce as Actor
            params = {'root': root, 'scale': scale}


            actor = Actor()
            network = actor.generate(
                nprocs=self.process_count,
                params=params).get_matrix()

            self.network += np.imag(network)
            result = self
        else:
            sources = [d for d in range(self.size) if d != root]
            destinations = [root]
            result = self.many_to_many_old(destinations=destinations,
                                           payload=scale,
                                           sources=sources)
        return result

    def load(self, file_name, dense=False):
        """Load from file_name."""
        from scipy import io
        if dense:
            self.network = io.mmread(file_name).todense()
        else:
            self.network = io.mmread(file_name)
        return self

    def save(self, file_name, comment=None):
        """Write to file_name."""
        #if self.augmented:
        if False:
            self.network.save(file_name)
        else:
            from scipy import io
            io.mmwrite(file_name, self.get_matrix(), comment=comment)
        return

    def many_to_many_old(self, destinations=None, payload=0, sources=None):
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
