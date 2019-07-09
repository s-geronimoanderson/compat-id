#!/usr/bin/env python

def go():
    """The main event."""
    show_names = True
    generate_class_names(
        process_count=2**4,
        classify_root=True,
        show_names=True)

def generate_class_names(
        process_count,
        classify_root=False,
        classify_scale=False,
        show_names=False):
    """Return a list containing class names according to the Hilbert curve."""
    pattern_name_combinations = [
        "2b {}:{}",
        "b+r {}:{}",
        "2r {}:{}"]
    class_names = [0 for _ in range(process_count**2 * 3)]
    for index in range(3):
        for first_root in range(process_count):
            for second_root in range(process_count):
                coordinates = (first_root, second_root, index)
                classification = tuple_to_scalar(
                    process_count,
                    coordinates)
                class_name = pattern_name_combinations[index].format(
                    first_root,
                    second_root)
                class_names[classification] = class_name
                if show_names:
                    print(classification, coordinates, class_name)
    return class_names

def tuple_to_scalar(grid_edge_size, coordinates):
    """Return the appropriate conversion based on coordinates' cardinality."""
    cardinality = len(coordinates)
    if cardinality == 1:
        return coordinates[0]
    elif cardinality == 2:
        return xy2d(grid_edge_size, coordinates)
    elif cardinality == 3:
        return xyz2d(grid_edge_size, coordinates)
    return result

def test_xyz(min_size=1, max_size=None):
    """Exhaustively test xyz2d and d2xyz for size n."""
    if min_size < 1:
        raise ValueError
    if max_size is None:
        max_size = min_size
    n = min_size
    while n <= max_size:
        print("Testing for n = {}...".format(n))
        for i in range(n * n):
            x, y, z = d2xyz(n, i)
            if i != xyz2d(n, (x, y, z)):
                return False
        n *= 2
    return True

def xyz2d(n, t):
    """Convert t = (x, y, z) to d.

    Based on C code example from Wikipedia:
    https://en.wikipedia.org/wiki/Hilbert_curve

    Simply treat a given z as an (x, y) layer connected to layers z-1 and z+1.
    """
    x, y, z = t
    d = xy2d(n, (x,y)) + z * n * n
    return d


def d2xyz(n, d):
    """Convert d to (x, y, z).

    Based on C code example from Wikipedia:
    https://en.wikipedia.org/wiki/Hilbert_curve

    Simply treat a given z as an (x, y) layer connected to layers z-1 and z+1.
    """
    max = n * n
    z = 0

    while d >= max:
        z += 1
        d -= max

    x, y = d2xy(n, d)
    return x, y, z

def xy2d(n, t):
    """Convert t = (x, y) to d.

    Ported from C code example; Source:
    https://en.wikipedia.org/wiki/Hilbert_curve
    """
    x, y = t
    d = 0
    # for (s=n/2; s>0; s/=2)
    s = n // 2
    while s > 0:
        rx = (x & s) > 0
        ry = (y & s) > 0
        d += s * s * ((3 * rx) ^ ry)
        x, y = rot(n, x, y, rx, ry)
        s //= 2
    return d


def d2xy(n, d):
    """Convert d to (x, y).

    Ported from C code example; Source:
    https://en.wikipedia.org/wiki/Hilbert_curve
    """
    t = d
    x = y = 0

    # for (s=1; s<n; s*=2)
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        x, y = rot(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


def rot(n, x, y, rx, ry):
    """Rotate/flip a quadrant (in place) appropriately.

    Ported from C code example; Source:
    https://en.wikipedia.org/wiki/Hilbert_curve
    """
    if ry == 0:
        if rx == 1:
            x = (n - 1) - x;
            y = (n - 1) - y;

        # Swap x and y:
        t = x;
        x = y;
        y = t;

    return x, y

if __name__ == "__main__":
    go()

# EOF
