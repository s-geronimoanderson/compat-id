#!/usr/bin/env python

import glob
import numpy as np
import os
import random
import sys

from enum import IntEnum
from itertools import combinations_with_replacement
from network import Network
from scipy import io

import hilbert


# Communication patterns.

class Pattern(IntEnum):
    BROADCAST = 0
    MANY_TO_MANY = 1
    NN2D05 = 2
    REDUCTION = 3

# Generator functions.

def load_label_mapping(patterns):
    """Return a mapping for the given patterns."""
    combos = combinations_with_replacement(sorted(patterns), len(patterns))
    mapping, names = {}, []
    for label, combo in enumerate(combos):
        int_combo = tuple([int(p) for p in combo])
        human_label = ''.join([str(p.name)[0] for p in combo])
        mapping[int_combo] = label
        names.insert(label, human_label)
    return {'mapping': mapping, 'names': names}


def generate_labeled_matrices(
        augmented=False,
        classify_root=False,
        classify_scale=False,
        communicator_count=0,
        compressed=False,
        individual_matrix_market=False,
        output_directory=None,
        patterns=None,
        process_count=28,
        scale_bit_min=4,
        scale_bit_max=9,
        sample_count=10000):
    """Return a matrix list and its corresponding label list."""
    print(f'Generating {sample_count} {process_count}-process matrices: ', end='')
    sys.stdout.flush()

    size = communicator_count + process_count
    factors = [f for f in range(2, process_count) if process_count % f == 0]
    powers = [p for p in range(2, sample_count) if sample_count % p == 0]
    pacifier = {key: f'-{key}' for key in powers}
    d = load_label_mapping(patterns)
    coordinate_to_classification, names = d['mapping'], d['names']

    def pacify(x):
        print(pacifier.get(x, ''), end='')
        sys.stdout.flush()

    if compressed:
        labels = np.empty(shape=sample_count, dtype=np.int64)
        matrices = np.empty(shape=(sample_count, size, size))

    if patterns is None:
        patterns = [Pattern.BROADCAST, Pattern.REDUCTION]

    for sample_index in range(sample_count):
        pacify(sample_index)

        current = Network(
            augmented=augmented,
            communicator_count=communicator_count,
            process_count=process_count)

        # TODO: Use itertools and random to make this simpler.
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

            if pattern == Pattern.BROADCAST:
                current.broadcast(root=root, scale=scale)
            elif pattern == Pattern.REDUCTION:
                current.reduce(root=root, scale=scale)
            elif pattern == Pattern.MANY_TO_MANY:
                current.reduce(root=root, scale=scale)
            elif pattern == Pattern.NN2D05:
                # 2D means something between (2 × n-2) and (n-2 x 2).
                height = random.choice(factors)
                current.nn2d(
                    dimensions=(height, process_count//height),
                    periodic=False,
                    scale=scale)

        # Classify.
        coordinate = tuple([int(p) for p in chosen_patterns])
        classification = coordinate_to_classification[coordinate]

        # Done.
        if compressed:
            labels[sample_index] = classification
            matrices[sample_index] = current.get_matrix()

        if individual_matrix_market:
            file_spec = f'{sample_index}-*.mtx'
            file_matches = glob.glob(os.path.join(output_directory, file_spec))
            if file_matches:
                print("File matches: ", file_matches)
            else:
                # Write out.
                file_name = f'{sample_index}-{classification}'
                current.save(
                    os.path.join(output_directory, f'{file_name}.mtx'),
                    comment=f'{file_name}')

    pacify(sample_count)
    print(" generated!")

    # Save aggregated labels and matrices, compressed.
    if compressed:
        np.savez_compressed(
            './tmp/matrices_and_labels.npz',
            labels=labels,
            matrices=matrices,
            names=names)
    print("saved.")
    return #labels, matrices


def load_matrices(
        input_directory,
        augmented=False,
        communicator_count=0,
        compressed=True,
        process_count=512,
        sample_count=512):
    """
    if not os.listdir(matrix_directory):
        print("Matrix directory is empty.")
        #os.makedirs(matrix_directory)
        generate_labeled_matrices(
            classify_root=classify_root,
            classify_scale=classify_scale,
            process_count=process_count,
            output_directory=matrix_directory,
            sample_count=training_count)
    """
    size = communicator_count + process_count

    labels = np.empty(shape=sample_count, dtype=np.int64)
    matrices = np.empty(shape=(sample_count, size, size))

    current_count = 0

    print(f'Loading {sample_count} {process_count}-process matrices: ', end='')
    for index, f in enumerate(os.listdir(input_directory)):
        # Get index and classification from file name.
        f_name, f_name_ext = os.path.splitext(f)
        if f_name_ext == '.mtx':
            current_count += 1
            if current_count > sample_count:
                break
            print('.', end='')
            creation_index, label = [int(x) for x in f_name.split('-')]

            # Get file contents as matrix.
            fqfn = os.path.join(input_directory, f)
            current = Network(
                augmented=augmented,
                communicator_count=communicator_count,
                process_count=process_count)
            current.load(fqfn)

            matrix = current.get_matrix()

            matrices[index] = matrix
            labels[index] = label

            if label > 10:
                print(f'Whoa! Index {index} has label {label} > 10')
    print(" loaded!")

    print("flattening for fun")
    row_length = size**2
    matrices_as_rows = np.empty(shape=(sample_count, row_length))
    for index in range(sample_count):
        matrix_as_row = np.reshape(matrices[index], row_length)
        matrices_as_rows[index] = matrix_as_row

    print("labeling for fun")
    labeled_matrices_as_rows = np.concatenate((matrices_as_rows, labels))

    #print("writing out for fun")

    return labels, matrices


def load_data(
        augmented=False,
        classify_root=False,
        classify_scale=False,
        communicator_count=0,
        compressed=False,
        process_count=28,
        scale_bit_min=4,
        scale_bit_max=9,
        testing_count=50,
        training_count=150):
    """Load matrices and class names."""
    sample_count = testing_count + training_count

    if compressed:
        d = np.load('./tmp/matrices_and_labels.npz')
        labels, matrices, names = d['labels'], d['matrices'], d['names']
    else:
        labels, matrices = load_matrices(
            './matrices',
            augmented=augmented,
            communicator_count=communicator_count,
            process_count=process_count,
            sample_count=sample_count)
        names = hilbert.generate_class_names(process_count=process_count)

    training_labels = labels[:training_count]
    testing_labels = labels[training_count:sample_count]

    training_matrices = matrices[:training_count]
    testing_matrices = matrices[training_count:sample_count]

    return ((training_matrices, training_labels),
            (testing_matrices, testing_labels),
            names)


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
    generate_labeled_matrices(
        augmented=True,
        communicator_count=1,
        compressed=True,
        individual_matrix_market=False,
        output_directory='./matrices',
        patterns=[
            Pattern.BROADCAST,
            Pattern.NN2D05,
            Pattern.REDUCTION],
        process_count=2**5,
        sample_count=2**15)

# Fin.
