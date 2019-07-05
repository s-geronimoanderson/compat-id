#!/usr/bin/env python

import numpy as np
import random

from matrix import Matrix

class Network:
    """An abstract network class."""
    def __init__(self, size=0):
        """Size is the process count."""
        self.network = np.zeros((size, size))
        self.reshape = self.network.reshape
        self.size = size

    def __str__(self):
        """Just pass-through for now."""
        return self.network.__str__()

    def send(self, source, destination, payload):
        """Simulate sending the payload from the source to the destination."""
        self.network[source, destination] += payload

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


# Communication patterns.

from enum import IntEnum

class Pattern(IntEnum):
    BROADCAST = 1
    REDUCTION = 2
    ONE_TO_MANY = 3
    MANY_TO_ONE = 4
    MANY_TO_MANY = 5


# Generator functions.

import hilbert


def generate_labeled_matrices(
        classify_root=False,
        classify_scale=False,
        patterns=None,
        process_count=28,
        sample_count=10000):
    """Return a matrix list and its corresponding label list."""
    if patterns is None:
        patterns = [Pattern.BROADCAST, Pattern.REDUCTION]

    matrices = []
    labels = []

    for sample_index in range(sample_count):
        current = Network(size=process_count)

        chosen_patterns = []
        for _ in len(patterns):
            chosen_patterns.append(random.choice(patterns))
        chosen_patterns.sort()

        coordinates = []
        for pattern in chosen_patterns:
            root = random.randrange(process_count)
            if classify_root:
                coordinates.append(root)
            scale = 2**random.randrange(scale_bit_min, scale_bit_max)

            if pattern == Pattern.BROADCAST:
                current = current.broadcast(root=root, scale=scale)
            elif pattern == Pattern.REDUCTION:
                current = current.reduce(root=root, scale=scale)

        # Classify.
        if patterns[0] != patterns[1]:
            coordinates.append(1)
        elif patterns[0] == Pattern.BROADCAST:
            coordinates.append(0)
        else:
            # Necessarily, patterns[0] == patterns[1] == Pattern.REDUCTION.
            coordinates.append(2)
        classification = hilbert.tuple_to_scaler(process_count, coordinates)

        # Done.
        matrices.append(current)
        labels.append(classification)
    return {'matrices': matrices, 'labels': labels}


def load_matrices(
        canonical_pattern_order=False,
        classify_root=False,
        classify_scale=False,
        stacked_curve_label=False,
        operation_count=1,
        patterns=None,
        process_count=64,
        random_root=True,
        random_scale=True,
        root=0,
        sample_count=1024,
        scale=512,
        scale_bit_min=4,
        scale_bit_max=14,
        enumerated_label=False):
    """Return a dictionary containing test data and a target vector."""
    feature_count = process_count**2
    data = np.zeros((sample_count, feature_count))
    target = []

    simple_label_mapping = {}
    simple_label_index = 0

    range_operation_count = range(operation_count)

    if patterns is None:
        patterns = [Pattern.BROADCAST, Pattern.REDUCTION]

    for sample_index in range(sample_count):
        current = Network(size=process_count)
        classification = ""

        chosen_patterns = []
        for _ in range_operation_count:
            chosen_patterns.append(random.choice(patterns))

        if canonical_pattern_order:
            chosen_patterns.sort()

        coordinates = []

        for pattern in chosen_patterns:
            if random_root:
                root = random.randrange(process_count)
                coordinates.append(root)
            if random_scale:
                scale = 2**random.randrange(scale_bit_min, scale_bit_max)

            if pattern == Pattern.BROADCAST:
                current = current.broadcast(root=root, scale=scale)
            elif pattern == Pattern.REDUCTION:
                current = current.reduce(root=root, scale=scale)

            if classification == "":
                classification = "{}(".format(pattern)
            else:
                classification = "{} + {}(".format(classification, pattern)

            parameterization = ""
            if classify_root:
                if parameterization != "":
                   parameterization = "{}, ".format(parameterization)
                parameterization = "{}root: {}".format(parameterization, root)
            if classify_scale:
                if parameterization != "":
                   parameterization = "{}, ".format(parameterization)
                parameterization = "{}scale: {}".format(parameterization,
                                                        scale)
            classification = "{}{})".format(classification,
                                            parameterization)

        # Reshape.
        data[sample_index, :] = current.reshape(feature_count)

        # Classify.
        if enumerated_label:
            classification = simple_label_mapping.get(
                classification,
                simple_label_index)
            if classification == simple_label_index:
                simple_label_index += 1
        if stacked_curve_label:
            if patterns[0] != patterns[1]:
                coordinates.append(1)
            elif patterns[0] == Pattern.BROADCAST:
                coordinates.append(0)
            else:
                # patterns[0] == patterns[1] == Pattern.REDUCTION.
                coordinates.append(2)
            classification = hilbert.xyz2d(process_count, coordinates)
        target.append(classification)
    return {'data': data, 'target': target}

def load_data(process_count=28,
              training_count=6,
              testing_count=1):
    """Keras-style data loader."""
    train_data = generate_labeled_matrices(
        process_count=process_count,
        sample_count=training_count)
    train_images = train_data['matrices']
    train_labels = train_data['labels']

    test_data = generate_labeled_matrices(
        process_count=process_count,
        sample_count=testing_count)
    test_images = test_data['matrices']
    test_labels = test_data['labels']
    return (train_images, train_labels), (test_images, test_labels)


def load_bcast_vs_reduce(
        classify_root=False,
        classify_scale=False,
        sample_count=1000,
        random_root=True,
        random_scale=False,
        root=0,
        scale=512,
        scale_bit_min=4,
        scale_bit_max=14,
        size=64):
    data = []
    target = []

    # Generate broadcast matrices with random roots.
    for i in range(sample_count // 2):
        if random_root:
            root = random.choice(range(size))
        if random_scale:
            scale = 2**random.choice(range(scale_bit_min, scale_bit_max))

        current = Network(size=size).broadcast(root=root, scale=scale)

        data.append(current.to_coo_matrix())
        classification = "pattern=broadcast"
        if classify_root:
            classification = "{},root={}".format(classification, root)
        if classify_scale:
            classification = "{},scale={}".format(classification, scale)
        target.append(classification)

    # Generate reduction matrices with random roots.
    for i in range(sample_count // 2):
        if random_root:
            root = random.choice(range(size))
        if random_scale:
            scale = 2**random.choice(range(scale_bit_min, scale_bit_max))
        current = Network(size=size).reduce(root=root, scale=scale)
        data.append(current.to_coo_matrix())
        classification = "pattern=reduce"
        if classify_root:
            classification = "{},root={}".format(classification, root)
        if classify_scale:
            classification = "{},scale={}".format(classification, scale)
        target.append(classification)

    result = {'data': data, 'target': target}
    return result


# Run-as-script idiom.

if __name__ == "__main__":
    pass
