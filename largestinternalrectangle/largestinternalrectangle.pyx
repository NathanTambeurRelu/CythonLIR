from cython import boundscheck, wraparound, initializedcheck, cdivision
from libc.math cimport INFINITY
from libc.stdint cimport uint8_t, uint32_t, int32_t
import numpy as np
cimport numpy as np
from cython import boundscheck, wraparound, initializedcheck
from libc.stdint cimport uint8_t, uint32_t
from cython.parallel cimport prange
import numpy as np
cimport numpy as np

@boundscheck(False)
@wraparound(False)
@initializedcheck(False)
def cy_get_horizontal_adjacency(uint8_t[:, ::1] cells):
    cdef int nrows = cells.shape[0]
    cdef int ncols = cells.shape[1]
    
    # Allocate adjacency matrix in C memory
    cdef uint32_t[:, ::1] adjacency_horizontal = np.zeros((nrows, ncols), dtype=np.uint32)
    
    cdef int x, y, span

    # Use prange for multithreading (each row is independent)
    for y in range(nrows):
        span = 0
        for x in range(ncols - 1, -1, -1):  # Faster than reversed(range())
            if cells[y, x] > 0:
                span += 1
            else:
                span = 0
            adjacency_horizontal[y, x] = span

    return np.array(adjacency_horizontal, copy=False)


@boundscheck(False)
@wraparound(False)
@initializedcheck(False)
def cy_get_vertical_adjacency(uint8_t[:, ::1] cells):
    cdef int nrows = cells.shape[0]
    cdef int ncols = cells.shape[1]

    # Allocate adjacency matrix in C memory
    cdef uint32_t[:, ::1] adjacency_vertical = np.zeros((nrows, ncols), dtype=np.uint32)
    
    cdef int x, y, span

    # Parallelize over columns instead of rows (better memory access pattern)
    for x in range(ncols):
        span = 0
        for y in range(nrows - 1, -1, -1):  # Faster than reversed(range())
            if cells[y, x] > 0:
                span += 1
            else:
                span = 0
            adjacency_vertical[y, x] = span

    return np.array(adjacency_vertical, copy=False)
@boundscheck(False)
@wraparound(False)
@initializedcheck(False)
def largest_interior_rectangle(uint8_t[:, ::1] grid):
    # Get adjacency matrices using the optimized Cython functions
    cdef uint32_t[:, ::1] h_adjacency = cy_get_horizontal_adjacency(grid)
    cdef uint32_t[:, ::1] v_adjacency = cy_get_vertical_adjacency(grid)
    
    # Get span map and find the largest rectangle
    cdef uint32_t[:, :, ::1] s_map = span_map(grid, h_adjacency, v_adjacency)
    return biggest_span_in_span_map(s_map)

@boundscheck(False)
@wraparound(False)
@initializedcheck(False)
def predict_vector_size(uint32_t[:] array):
    cdef int i, n = array.shape[0]
    cdef uint32_t val
    
    for i in range(n):
        val = array[i]
        if val == 0:
            return i
    return n

@boundscheck(False)
@wraparound(False)
@initializedcheck(False)
def h_vector(uint32_t[:, ::1] h_adjacency, int x, int y):
    cdef int vector_size = predict_vector_size(h_adjacency[y:, x])
    cdef int i, j = 0
    cdef uint32_t h = <uint32_t>INFINITY
    cdef uint32_t current
    # Preallocate maximum possible size
    cdef np.ndarray[uint32_t, ndim=1] result = np.zeros(vector_size, dtype=np.uint32)
    cdef uint32_t prev = 0
    
    for i in range(vector_size):
        h = min(h_adjacency[y + i, x], h)
        current = h
        if current != prev:  # Only store unique values
            result[j] = current
            j += 1
            prev = current
    
    # Return only the used portion, reversed
    return result[:j][::-1]

@boundscheck(False)
@wraparound(False)
@initializedcheck(False)
def v_vector(uint32_t[:, ::1] v_adjacency, int x, int y):
    cdef int vector_size = predict_vector_size(v_adjacency[y, x:])
    cdef int i, j = 0
    cdef uint32_t v = <uint32_t>INFINITY
    cdef uint32_t current
    # Preallocate maximum possible size
    cdef np.ndarray[uint32_t, ndim=1] result = np.zeros(vector_size, dtype=np.uint32)
    cdef uint32_t prev = 0
    
    for i in range(vector_size):
        v = min(v_adjacency[y, x + i], v)
        current = v
        if current != prev:  # Only store unique values
            result[j] = current
            j += 1
            prev = current
    
    # Return only the used portion, reversed
    return result[:j][::-1]

@boundscheck(False)
@wraparound(False)
@initializedcheck(False)
def spans(uint32_t[:] h_vector, uint32_t[:] v_vector):
    cdef int h_len = h_vector.shape[0]
    cdef int v_len = v_vector.shape[0]
    cdef np.ndarray[uint32_t, ndim=2] result = np.zeros((h_len, 2), dtype=np.uint32)
    cdef int i
    
    for i in range(h_len):
        result[i, 0] = h_vector[i]
        result[i, 1] = v_vector[v_len - i - 1] if i < v_len else v_vector[0]
    
    return result

@boundscheck(False)
@wraparound(False)
@initializedcheck(False)
@cdivision(True)
def biggest_span(np.ndarray[uint32_t, ndim=2] spans_array):
    cdef np.ndarray[uint32_t, ndim=1] result = np.array([0, 0], dtype=np.uint32)
    
    if spans_array.shape[0] == 0:
        return result
    
    # Calculate areas and find maximum
    cdef np.ndarray[uint32_t, ndim=1] areas = spans_array[:, 0] * spans_array[:, 1]
    cdef int biggest_span_index = np.argmax(areas)
    
    return spans_array[biggest_span_index]

@boundscheck(False)
@wraparound(False)
@initializedcheck(False)
def span_map(uint8_t[:, ::1] grid, uint32_t[:, ::1] h_adjacency, uint32_t[:, ::1] v_adjacency):
    cdef int nrows = grid.shape[0]
    cdef int ncols = grid.shape[1]
    
    # Allocate output span map
    cdef uint32_t[:, :, ::1] span_map = np.zeros((nrows, ncols, 2), dtype=np.uint32)
    
    cdef int x, y, i, j, h_vec_size, v_vec_size
    cdef uint32_t h, v, prev_h, prev_v
    cdef uint32_t max_area, best_h, best_v, area
    
    # Maximum possible size of span vectors
    cdef int max_size = min(nrows, ncols)
    
    # Preallocate memory buffers
    cdef uint32_t[::1] h_vector = np.zeros(max_size, dtype=np.uint32)
    cdef uint32_t[::1] v_vector = np.zeros(max_size, dtype=np.uint32)

    # Parallel loop over rows
    for y in range(nrows):
        for x in range(ncols):
            if grid[y, x] > 0:
                # Compute horizontal span vector
                h_vec_size = 0
                h = <uint32_t>INFINITY
                prev_h = 0
                i = y
                while i < nrows:
                    h = min(h, h_adjacency[i, x])
                    if h != prev_h:
                        h_vector[h_vec_size] = h
                        prev_h = h
                        h_vec_size += 1
                    if h == 0:
                        break
                    i += 1

                # Compute vertical span vector
                v_vec_size = 0
                v = <uint32_t>INFINITY
                prev_v = 0
                i = x
                while i < ncols:
                    v = min(v, v_adjacency[y, i])
                    if v != prev_v:
                        v_vector[v_vec_size] = v
                        prev_v = v
                        v_vec_size += 1
                    if v == 0:
                        break
                    i += 1

                # Compute largest span (area-wise)
                max_area = 0
                best_h = 0
                best_v = 0
                
                j = 0
                while j < h_vec_size:
                    if j < v_vec_size:
                        area = h_vector[j] * v_vector[v_vec_size - j - 1]
                    else:
                        area = h_vector[j] * v_vector[0]

                    if area > max_area:
                        max_area = area
                        best_h = h_vector[j]
                        best_v = v_vector[v_vec_size - j - 1] if j < v_vec_size else v_vector[0]
                    
                    j += 1

                # Store in span_map
                span_map[y, x, 0] = best_h
                span_map[y, x, 1] = best_v

    return np.array(span_map, copy=False)


@boundscheck(False)
@wraparound(False)
@initializedcheck(False)
def biggest_span_in_span_map(uint32_t[:, :, ::1] span_map):
    cdef int nrows = span_map.shape[0]
    cdef int ncols = span_map.shape[1]
    cdef np.ndarray[uint32_t, ndim=2] areas_np = np.zeros((nrows, ncols), dtype=np.uint32)
    cdef uint32_t[:, ::1] areas = areas_np
    
    cdef int y, x
    
    # Calculate areas
    for y in range(nrows):
        for x in range(ncols):
            areas[y, x] = span_map[y, x, 0] * span_map[y, x, 1]
    
    # Find the largest rectangle
    cdef np.ndarray y_indices, x_indices
    y_indices, x_indices = np.where(areas_np == np.max(areas_np))
    
    cdef int y_max = y_indices[0]
    cdef int x_max = x_indices[0]
    
    return np.array([x_max, y_max, span_map[y_max, x_max, 0], span_map[y_max, x_max, 1]], dtype=np.uint32)
