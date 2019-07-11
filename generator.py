#!/usr/bin/env python

import numpy as np
import os
import random

from enum import IntEnum
from network import Network
from scipy import io


# Communication patterns.

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
        scale_bit_min=4,
        scale_bit_max=9,
        sample_count=10000):
    """Return a matrix list and its corresponding label list."""
    labels = np.empty(shape=sample_count, dtype=np.int64)
    matrices = np.empty(shape=(sample_count, process_count, process_count))
    if patterns is None:
        patterns = [Pattern.BROADCAST, Pattern.REDUCTION]

    for sample_index in range(sample_count):
        current = Network(augmented=False, size=process_count)

        chosen_patterns = []
        for _ in range(len(patterns)):
            chosen_patterns.append(random.choice(patterns))
        chosen_patterns.sort()

        coordinates = []
        for pattern in chosen_patterns:
            root = random.randrange(process_count)
            if classify_root:
                coordinates.append(root)
            scale = 2**random.randrange(scale_bit_min, scale_bit_max)

            if scale > 512:
                print('+', end='')

            if pattern == Pattern.BROADCAST:
                current = current.broadcast(root=root, scale=scale)
            elif pattern == Pattern.REDUCTION:
                current = current.reduce(root=root, scale=scale)

        # Classify.
        if chosen_patterns[0] is not chosen_patterns[1]:
            # Here, the patterns are different (one of each).
            coordinates.append(1)
        elif chosen_patterns[0] is Pattern.BROADCAST:
            # Here, patterns[0] is patterns[1] is Pattern.BROADCAST.
            coordinates.append(0)
        else:
            # Here, patterns[0] is patterns[1] is Pattern.REDUCTION.
            coordinates.append(2)
        classification = hilbert.tuple_to_scalar(process_count, coordinates)

        # Done.
        labels[sample_index] = classification
        matrices[sample_index] = current.as_numpy_array()
        """
        # Write out.
        fqfn = os.path.join(output_directory, f'{sample_index}-{classification}.mtx')
        io.mmwrite(
            fqfn,
            current.tolil(),
            comment=f'{classification}')
        """
    return labels, matrices


def load_matrices(
        classify_root=False,
        classify_scale=False,
        sample_count=1,
        process_count=2):
    """
    matrices = np.empty(
        shape=(sample_count, process_count, process_count),
        dtype=object)
    labels = np.empty(shape=sample_count, dtype=np.int64)

    if not os.listdir(matrix_directory):
        print("Matrix directory is empty.")
        #os.makedirs(matrix_directory)
        generate_labeled_matrices(
            classify_root=classify_root,
            classify_scale=classify_scale,
            process_count=process_count,
            output_directory=matrix_directory,
            sample_count=training_count)

    for f in os.listdir(matrix_directory):
        # Get file contents as matrix.
        fqfn = os.path.join(matrix_directory, f)
        matrix = mmread(fqfn)

        # Get index and classification from file name.
        f_sans_ext = os.path.splitext(f)[0]
        sample_index = int(f_sans_ext.split('-')[0])
        classification = int(f_sans_ext.split('-')[1])
        print(f, f_sans_ext, sample_index, classification)
        matrices[sample_index] = matrix
        labels[sample_index] = classification
    """

    # Eventually, this function will load from disk.
    return generate_labeled_matrices(
        classify_root=classify_root,
        classify_scale=classify_scale,
        process_count=process_count,
        sample_count=sample_count)


def load_data(
        classify_root=False,
        classify_scale=False,
        process_count=28,
        scale_bit_min=4,
        scale_bit_max=9,
        testing_count=50,
        training_count=150):
    """Load matrices and class names."""
    train_labels, train_matrices = load_matrices(
        process_count=process_count,
        sample_count=training_count)

    test_labels, test_matrices = load_matrices(
        process_count=process_count,
        sample_count=testing_count)

    class_names = hilbert.generate_class_names(
        process_count=process_count)
    return (train_matrices, train_labels), (test_matrices, test_labels), class_names


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
