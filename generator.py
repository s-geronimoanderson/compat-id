#!/usr/bin/env python

import glob
import numpy as np
import os
import random

from enum import IntEnum
from network import Network
from scipy import io

import hilbert


# Communication patterns.

class Pattern(IntEnum):
    BROADCAST = 0
    NN2D05 = 1
    REDUCTION = 2

# Generator functions.

def generate_labeled_matrices(
        augmented=False,
        classify_root=False,
        classify_scale=False,
        communicator_count=0,
        output_directory=None,
        patterns=None,
        process_count=28,
        scale_bit_min=4,
        scale_bit_max=9,
        sample_count=10000):
    """Return a matrix list and its corresponding label list."""
    size = communicator_count + process_count
    factors = [f for f in range(2, process_count) if process_count % f == 0]

    #labels = np.empty(shape=sample_count, dtype=np.int64)
    #matrices = np.empty(shape=(sample_count, process_count, process_count))
    if patterns is None:
        patterns = [Pattern.BROADCAST, Pattern.REDUCTION]

    for sample_index in range(sample_count):
        current = Network(
            augmented=augmented,
            communicator_count=communicator_count,
            process_count=process_count)

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
            elif pattern == Pattern.NN2D05:
                # 2D means something between (2 × n-2) and (n-2 x 2).
                height = random.choice(factors)
                current.nn2d(
                    dimensions=(height, process_count//height),
                    periodic=False,
                    scale=scale)

        # Classify.
        if False: # Old version
            if chosen_patterns[0] is not chosen_patterns[1]:
                # Here, the patterns are different (one of each).
                coordinates.append(1)
            elif chosen_patterns[0] is Pattern.BROADCAST:
                coordinates.append(0)
            else:
                coordinates.append(2)
            classification = hilbert.tuple_to_scalar(size, coordinates)
        else:
            coordinate = tuple([int(p) for p in chosen_patterns])
            coordinate_to_classification = {
                (0,0,0): 0, # bbb
                (0,0,1): 1, # bbn
                (0,0,2): 2, # bbr
                (0,1,1): 3, # bnn
                (0,1,2): 4, # bnr
                (0,2,2): 5, # brr
                (1,1,1): 6, # nnn
                (1,1,2): 7, # nnr
                (1,2,2): 8, # nrr
                (2,2,2): 9} # rrr
            classification = coordinate_to_classification[coordinate]

        # Done.
        if output_directory is None:
            #labels[sample_index] = classification
            #matrices[sample_index] = current.get_matrix()
            pass
        else:
            file_spec = f'{sample_index}-*.mtx'
            file_matches = glob.glob(os.path.join(output_directory, file_spec))
            if file_matches:
                print("File matches: ", file_matches)
            else:
                # Write out.
                file_name = f'{sample_index}-{classification}'
                current.save(
                    os.path.join(output_directory,
                                 f'{file_name}.mtx'),
                    comment=f'{file_name}')
    return #labels, matrices


def load_matrices(
        input_directory,
        augmented=False,
        communicator_count=0,
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

    print(f'Loading {process_count}-process matrices: ', end='')
    for index, f in enumerate(os.listdir(input_directory)):
        current_count += 1
        if current_count > sample_count:
            break
        # Get index and classification from file name.
        f_name, f_name_ext = os.path.splitext(f)
        if f_name_ext == '.mtx':
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

    print(" loaded!")
    return labels, matrices


def load_data(
        augmented=False,
        classify_root=False,
        classify_scale=False,
        communicator_count=0,
        process_count=28,
        scale_bit_min=4,
        scale_bit_max=9,
        testing_count=50,
        training_count=150):
    """Load matrices and class names."""
    sample_count = testing_count + training_count
    """
    labels, matrices = generate_labeled_matrices(
        augmented=augmented,
        classify_root=classify_root,
        classify_scale=classify_scale,
        output_directory='./matrices',
        process_count=process_count,
        sample_count=sample_count,
        scale_bit_min=scale_bit_min,
        scale_bit_max=scale_bit_max)
    """

    labels, matrices = load_matrices(
        './matrices',
        augmented=augmented,
        communicator_count=communicator_count,
        process_count=process_count,
        sample_count=sample_count)

    training_labels = labels[:training_count]
    testing_labels = labels[training_count:]

    training_matrices = matrices[:training_count]
    testing_matrices = matrices[training_count:]

    class_names = hilbert.generate_class_names(process_count=process_count)
    return ((training_matrices, training_labels),
            (testing_matrices, testing_labels),
            class_names)


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
        output_directory='./matrices',
        patterns=[Pattern.BROADCAST, Pattern.NN2D05, Pattern.REDUCTION],
        process_count=2**9,
        sample_count=2**10)

# Fin.
