#!/usr/bin/env python

from scipy.sparse import coo_matrix

class Matrix:
    """A sparse matrix class (indexed from zero). Replace with NumPy arrays."""
    def __init__(self, matrix=None, size=None):
        """Size is a tuple (m,n) representing m rows and n columns."""
        if matrix is None:
            self.data = {}
            if size is None:
                self.column_count = 0
                self.row_count = 0
                self.size = (self.row_count, self.column_count)
            else:
                self.row_count, self.column_count = size[:2]
                self.size = size
        else:
            """Initialize to be a clone of the given matrix."""
            self.column_count = matrix.column_count
            self.data = matrix.data
            self.row_count = matrix.row_count
            self.size = matrix.size

    def get(self, subscript):
        """Return the matrix element indexed by the given (valid) subscript."""
        row, column = subscript[:2]
        if self.__is_valid(subscript):
            # Get the value if it's present, else a zero.
            result = self.data.get(subscript, 0)
        else:
            raise IndexError
        return result

    def set(self, subscript, value):
        """Set the matrix element indexed by the given (valid) subscript."""
        if value != 0 and self.__is_valid(subscript):
            self.data[subscript] = value
        else:
            raise IndexError

    def __is_valid(self, subscript):
        """Return whether the given subscript is within the matrix's bounds."""
        return ((0,0) <= subscript and subscript < self.size)

    def __is_valid_row(self, row_number):
        """Return whether the given row is within the matrix's bounds."""
        return self.__is_valid((row_number, 0))

    def __str__(self):
        """Return a NumPy-like matrix representation."""
        result = ""
        for row in range(self.size):
            current = []
            for column in range(self.size):
                subscript = (row, column)
                current.append(self.get(subscript))
            if result == "":
                result = "[{}".format(current)
            else:
                result = "{0}\n {1}".format(result, current)
        return "{}]".format(result)

    def extend_columns(self, matrix):
        raise NotImplementedError

    def extend_rows(self, matrix):
        """Extend the current matrix with the given matrix."""
        row_count, column_count = matrix.size[:2]
        if column_count != self.column_count:
            raise ValueError
        self.row_count += row_count
        self.size = (self.row_count, self.column_count)
        base_row_count = self.row_count
        for key, value in matrix.data.items():
            row, column = key[:2]
            self.set((base_row_count + row, column), value)
        return self

    def replace_row(self, row_number, vector):
        """Replace the specified row with the given vector."""
        if not self.__is_valid_row(row_number):
            raise ValueError
        row_count, column_count = vector.size[:2]
        if row_count != 1 and column_count != 1:
            raise ValueError
        # Eliminate current row entries.
        for col in [col for (row, col) in self.data.items()
                    if row == row_number]:
            self.data.pop(row_number, col)
        # Update row with vector elements.
        if row_count == 1:
            new_row = vector.transpose()
        else:
            new_row = vector
        for key, value in new_row.data.items():
            row, _ = key[:2]
            self.set((row_number, row), value)
        return self

    def submatrix(self, row_set, column_set):
        """Return a submatrix with the given rows and columns."""
        submatrix = Matrix(len(row_set), len(column_set))
        raise NotImplementedError

    def to_vec(self):
        """Return an m*n length vector comprising all the matrix's columns."""
        column_count = self.column_count
        vector = Matrix(size=(self.row_count * column_count, 1))
        for key, value in self.data.items():
            row, column = key[:2]
            subscript = (column * column_count + row, 0)
            vector.set(subscript, value)
        return vector

    def to_ijv(self):
        """Return the matrix in ijv (triplet) array format."""
        row_indices = []
        column_indices = []
        nonzero_elements = []
        k = 0
        for key, value in self.data.items():
            if value == 0:
                continue
            row, col = key[:2]
            row_indices.append(row)
            column_indices.append(col)
            nonzero_elements.append(value)
            k += 1
        return row_indices, column_indices, nonzero_elements

    def to_coo_matrix(self):
        """Return the matrix in COOrdinate format."""
        row_indices, column_indices, nonzero_elements = self.to_ijv()
        return coo_matrix((nonzero_elements, (row_indices, column_indices)),
                          shape=(self.size, self.size))

    def transpose(self):
        """Transpose the matrix."""
        m, n = self.size[:2]
        transposed_size = (n, m)
        transposed_matrix = {}
        for key, value in matrix.data.items():
            i, j = key[:2]
            transposed_key = (j, i)
            transposed_matrix[transposed_key] = value
        self.matrix = transposed_matrix
        self.size = transposed_size
