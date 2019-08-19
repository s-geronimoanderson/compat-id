#!/usr/bin/env python

import glob
import math
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
    NN3D07 = 3
    REDUCTION = 4
    SWEEP3D07CORNER = 5
    NULL = 9


abbreviation = {
    Pattern.BROADCAST: 'b',
    Pattern.MANY_TO_MANY: 'm',
    Pattern.NN2D05: '2',
    Pattern.NN3D07: '3',
    Pattern.REDUCTION: 'r',
    Pattern.SWEEP3D07CORNER: 's',
    Pattern.NULL: ''}


# Generator functions.

def load_label_mapping(patterns, cardinality_min=None, cardinality_max=None):
    """Return a mapping for the given patterns with the given cardinality."""
    combos, mapping, names = [], {}, []
    pattern_count = len(patterns)
    if cardinality_min is None:
        cardinality_min = pattern_count
    if cardinality_max is None:
        cardinality_max = pattern_count
    for cardinality in range(cardinality_min, 1 + cardinality_max):
        combos += combinations_with_replacement(sorted(patterns), cardinality)
    for label, combo in enumerate(combos):
        int_combo = tuple([int(p) for p in combo])
        human_label = ''.join([abbreviation[p] for p in combo])
        mapping[int_combo] = label
        names.insert(label, human_label)
    return {'mapping': mapping, 'names': names}


def pacify(x, pacifier=None):
    if pacifier is None:
        pacifier = {}
    print(pacifier.get(x, ''), end='')
    sys.stdout.flush()


def generate_labeled_matrices(
        augmented=False,
        classify_root=False,
        classify_scale=False,
        communicator_count=0,
        compressed=False,
        individual_matrix_market=False,
        musketeer_mode=False,
        output_directory=None,
        output_file_name=None,
        patterns=None,
        pattern_count_min=None,
        pattern_count_max=None,
        process_count=28,
        scale_bit_min=4,
        scale_bit_max=9,
        sample_count=10000,
        variable_scale=False):
    """Return a matrix list and its corresponding label list."""
    print(f'Generating {process_count}-task matrices ({sample_count} samples): ', end='')
    sys.stdout.flush()

    size = communicator_count + process_count
    factors = [f for f in range(2, process_count) if process_count % f == 0]

    powers = [p for p in range(2, sample_count) if sample_count % p == 0]
    pacifier = {key: f'-{key}' for key in powers}

    labels = np.empty(shape=sample_count, dtype=np.int64)
    matrices = np.empty(shape=(sample_count, size, size), dtype=np.complex128)

    if output_directory is None:
        output_directory = './tmp'
    if output_file_name is None:
        output_file_name = 'labels_and_matrices_and_names.npz'

    if patterns is None:
        patterns = [Pattern.BROADCAST, Pattern.REDUCTION]

    if pattern_count_max is None:
        pattern_count_max = len(patterns)
    if pattern_count_min is None:
        pattern_count_min = pattern_count_max
    pattern_counts = range(pattern_count_min, 1 + pattern_count_max)

    d = load_label_mapping(
        patterns,
        cardinality_min=pattern_count_min,
        cardinality_max=pattern_count_max)
    coordinate_to_classification, names = d['mapping'], d['names']

    for sample_index in range(sample_count):
        pacify(sample_index, pacifier)

        current = Network(
            augmented=augmented,
            communicator_count=communicator_count,
            process_count=process_count)

        # TODO: Use itertools and random to make this simpler.
        chosen_patterns = []
        for _ in range(random.choice(pattern_counts)):
            chosen_patterns.append(random.choice(patterns))
        chosen_patterns.sort()

        coordinates = []
        for pattern in chosen_patterns:
            root = random.randrange(process_count)
            if classify_root:
                coordinates.append(root)
            scale = 2**random.randrange(scale_bit_min, scale_bit_max)

            if pattern == Pattern.BROADCAST:
                current.broadcast(root=root, scale=scale, variable_scale=variable_scale)
            elif pattern == Pattern.REDUCTION:
                current.reduce(root=root, scale=scale, variable_scale=variable_scale)
            elif pattern == Pattern.MANY_TO_MANY:
                current.many_to_many(scale=scale, variable_scale=variable_scale)
            elif pattern == Pattern.NN2D05:
                dimensions = random_dimensions(process_count, 2)
                periodicity = random_periodicity(2, musketeer_mode=musketeer_mode)
                current.nn2d(
                    dimensions=dimensions,
                    periodicity=periodicity,
                    scale=scale,
                    variable_scale=variable_scale)
            elif pattern == Pattern.NN3D07:
                dimensions = random_dimensions(process_count, 3)
                periodicity = random_periodicity(3, musketeer_mode=musketeer_mode)
                current.nn3d(
                    dimensions=dimensions,
                    periodicity=periodicity,
                    scale=scale,
                    variable_scale=variable_scale)
            elif pattern == Pattern.SWEEP3D07CORNER:
                corner = random_corner(3)
                dimensions = random_dimensions(process_count, 3)
                current.sweep3d(
                    corner=corner,
                    dimensions=dimensions,
                    scale=scale,
                    variable_scale=variable_scale)

        # Classify.
        coordinate = tuple([int(p) for p in chosen_patterns])
        classification = coordinate_to_classification[coordinate]

        # Done.
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

    print(f'-{sample_count} generated!')

    # Save aggregated labels and matrices, compressed.
    if compressed:
        np.savez_compressed(
            output_file_name,
            labels=labels,
            matrices=matrices,
            names=names)
    print("saved.")
    return #{'labels': labels, 'matrices': matrices}


def load_matrices(
        input_directory,
        augmented=False,
        communicator_count=0,
        compressed=True,
        extended=False,
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

    if extended:
        extended_size = 2 * size
        samples = np.empty(shape=(sample_count, extended_size, extended_size))

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
    print(" loaded!")
    return {'labels': labels, 'matrices': matrices, 'names': names}


def load_data(
        augmented=False,
        classify_root=False,
        classify_scale=False,
        communicator_count=0,
        compressed=False,
        extended=False,
        input_file_name=None,
        process_count=28,
        scale_bit_min=4,
        scale_bit_max=9,
        testing_count=50,
        training_count=150):
    """Load matrices and class names."""
    sample_count = testing_count + training_count

    if input_file_name is None:
        input_file_name = './samples/labels_and_matrices_and_names.npz'

    if compressed:
        data = np.load(input_file_name)
    else:
        data = load_matrices(
            './matrices',
            augmented=augmented,
            communicator_count=communicator_count,
            extended=extended,
            process_count=process_count,
            sample_count=sample_count)

    labels, matrices, names = data['labels'], data['matrices'], data['names']

    if extended:
        powers = [p for p in range(2, sample_count) if sample_count % p == 0]
        pacifier = {key: f'-{key}' for key in powers}
        print(f'Preparing {process_count}-task matrices ({sample_count} samples): ', end='')
        size = process_count + communicator_count
        extended_size = 2 * size
        samples = np.empty(shape=(sample_count, extended_size, extended_size))
        for sample_index, matrix in enumerate(matrices[:sample_count]):
            sample = np.zeros(shape=(extended_size, extended_size))
    
            # Features.
            northwest_matrix = np.zeros(shape=(size, size))
            northeast_matrix = np.zeros(shape=(size, size))
            southwest_matrix = np.zeros(shape=(size, size))
            southeast_matrix = np.zeros(shape=(size, size))

            # Transfer count.
            northwest_matrix += matrix.real
            northwest_matrix_max = np.amax(northwest_matrix)
            if northwest_matrix_max > 0:
                northwest_matrix /= northwest_matrix_max

            # Transfer volume.
            northeast_matrix += matrix.imag
            northeast_matrix_max = np.amax(northeast_matrix)
            if northeast_matrix_max > 0:
                northeast_matrix /= northeast_matrix_max

            # 2-norms.
            #import numpy.linalg as LA
            #for row_index, row in enumerate(matrix):
            #    northeast_matrix[row_index, 0] = LA.norm(row)

            # 2D DCT-II
            # https://stackoverflow.com/a/15983991
            from scipy.fftpack import dct, fft

            # TODO: Try concatenating the matrices and taking a single dct.
            # TODO: Try taking the dct before scaling the matrices.

            # Currently count submatrix 2D DCT-II.
            if True: # Scaled magnitude-only to [0, 1].
                southwest_matrix = np.abs(dct(dct(northwest_matrix, axis=0), axis=1))
                southwest_matrix_max = np.amax(southwest_matrix)
                if southwest_matrix_max != 0:
                    southwest_matrix /= southwest_matrix_max
            if False: # Scaled to [-1, 1].
                southwest_matrix = dct(dct(northwest_matrix, axis=0), axis=1)
                southwest_matrix /= np.amax(np.abs(southwest_matrix))
            if False: # Scaled to [-0.5, 0.5] and shifted to [0, 1].
                southwest_matrix = dct(dct(northwest_matrix, axis=0), axis=1)
                southwest_matrix /= np.amax(np.abs(southwest_matrix))
                southwest_matrix /= 2
                southwest_matrix += 0.5
            if False: # Scale by max - min, shift to [0, 1].
                southwest_matrix = dct(dct(northwest_matrix, axis=0), axis=1)
                southwest_matrix_min = np.amin(southwest_matrix)
                southwest_matrix_max = np.amax(southwest_matrix)
                southwest_matrix_range = southwest_matrix_max - southwest_matrix_min
                southwest_matrix -= southwest_matrix_min
                if southwest_matrix_range != 0:
                    southwest_matrix /= southwest_matrix_range

            # Currently scale submatrix 2D DCT-II.
            if True: # Scaled magnitude-only to [0, 1].
                southeast_matrix = np.abs(dct(dct(northeast_matrix, axis=0), axis=1))
                southeast_matrix_max = np.amax(southeast_matrix)
                if southeast_matrix_max != 0:
                    southeast_matrix /= southeast_matrix_max
            if False: # Scaled to [-1, 1].
                southeast_matrix = dct(dct(northeast_matrix, axis=0), axis=1)
                southeast_matrix /= np.amax(np.abs(southeast_matrix))
            if False: # Scaled to [-0.5, 0.5] and shifted to [0, 1].
                southeast_matrix = dct(dct(northeast_matrix, axis=0), axis=1)
                southeast_matrix /= np.amax(np.abs(southeast_matrix))
                southeast_matrix /= 2
                southeast_matrix += 0.5
            if False: # Scale by max - min, shift to [0, 1].
                southeast_matrix = dct(dct(northeast_matrix, axis=0), axis=1)
                southeast_matrix_min = np.amin(southeast_matrix)
                southeast_matrix_max = np.amax(southeast_matrix)
                southeast_matrix_range = southeast_matrix_max - southeast_matrix_min
                southeast_matrix -= southeast_matrix_min
                if southeast_matrix_range != 0:
                    southeast_matrix /= southeast_matrix_range

            # Wavelet transform.
            from scipy import signal

            #southeast_matrix = signal.cwt(matrix, signal.ricker, np.arange(1, 31))
            #southeast_matrix /= np.amax(southeast_matrix)

            # Add extended features to sample.
            sample[:size, :size] += northwest_matrix
            sample[:size, size:] += northeast_matrix
            sample[size:, :size] += southwest_matrix
            sample[size:, size:] += southeast_matrix
    
            samples[sample_index] = sample
            pacify(sample_index, pacifier)
        matrices = samples
        print(f'-{sample_count} prepared!')
    training_labels = labels[:training_count]
    testing_labels = labels[training_count:sample_count]

    training_matrices = matrices[:training_count]
    testing_matrices = matrices[training_count:sample_count]

    return ((training_matrices, training_labels),
            (testing_matrices, testing_labels),
            names)


def random_corner(cardinality):
    '''Return a tuple representing a random corner in a space with the given cardinality.'''
    corner = [random.choice([0, 1]) for _ in range(cardinality)]
    return tuple(corner)


def random_periodicity(cardinality, musketeer_mode=False):
    '''Return a list representing random periodicity in a space with the given cardinality.'''
    periodicity = [random.choice([True, False]) for _ in range(cardinality)]
    if musketeer_mode:
        # Set all dimensions to the same random value.
        periodicity = [periodicity[0]] * cardinality
    #print(f'Chose periodicity {periodicity}')
    return periodicity


def random_dimensions(process_count, cardinality):
    '''Return a tuple with random dimensions satisfying the process count.'''
    def f(process_count, cardinality, dimensions):
        '''Return recursively a random dimension list satisfying the count.'''
        if cardinality == 0:
            return dimensions
        elif cardinality == 1:
            return dimensions + [process_count]
        else:
            max_power = int(math.log(process_count, 2))
            available_factors = [2**c for c in range(1, max_power + 1 - cardinality )]
            dimension = random.choice(available_factors)
            rest = process_count // dimension
            return f(rest, (cardinality - 1), dimensions + [dimension])
    # Call helper. 
    return tuple(f(process_count, cardinality, []))


# Run-as-script idiom.

if __name__ == "__main__":
    generate_labeled_matrices(
        augmented=True,
        communicator_count=1,
        compressed=True,
        musketeer_mode=True,
        output_file_name='./tmp/bmr-t5-s14.npz',
        #output_file_name='./tmp/bmrs23-t6-s14-variable.npz',
        patterns=[
            Pattern.BROADCAST,
            Pattern.MANY_TO_MANY,
            #Pattern.NN2D05,
            #Pattern.NN3D07,
            Pattern.REDUCTION],
            #Pattern.SWEEP3D07CORNER],
        #pattern_count_min=2,
        #pattern_count_max=2,
        process_count=2**5,
        sample_count=2**14,
        variable_scale=False)
        #variable_scale=True)

# Fin.
