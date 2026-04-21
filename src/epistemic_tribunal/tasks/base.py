"""Base task interface and grid-analysis utilities shared across tasks.

Grid convention
---------------
A grid is a ``list[list[int]]`` where each cell holds a colour index (0–9).
Row 0 is the top row; column 0 is the left-most column.
"""

from __future__ import annotations

from collections import Counter


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

Grid = list[list[int]]


# ---------------------------------------------------------------------------
# Basic grid utilities
# ---------------------------------------------------------------------------


def grid_shape(grid: Grid) -> tuple[int, int]:
    """Return (rows, cols) of *grid*."""
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    return rows, cols


def unique_colours(grid: Grid) -> set[int]:
    """Return the set of distinct colour values present in *grid*."""
    return {cell for row in grid for cell in row}


def colour_counts(grid: Grid) -> dict[int, int]:
    """Return a Counter of colour values in *grid*."""
    return dict(Counter(cell for row in grid for cell in row))


def object_count(grid: Grid, background: int = 0) -> int:
    """Return the number of distinct connected components (non-background).

    Uses 4-connectivity flood-fill.
    """
    rows, cols = grid_shape(grid)
    visited: list[list[bool]] = [[False] * cols for _ in range(rows)]
    count = 0

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != background and not visited[r][c]:
                _flood_fill(grid, visited, r, c, rows, cols, background)
                count += 1
    return count


def _flood_fill(
    grid: Grid,
    visited: list[list[bool]],
    r: int,
    c: int,
    rows: int,
    cols: int,
    background: int,
) -> None:
    """Iterative 4-connectivity flood fill starting at (r, c)."""
    stack = [(r, c)]
    while stack:
        cr, cc = stack.pop()
        if cr < 0 or cr >= rows or cc < 0 or cc >= cols:
            continue
        if visited[cr][cc] or grid[cr][cc] == background:
            continue
        visited[cr][cc] = True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            stack.append((cr + dr, cc + dc))


def connected_components(grid: Grid, background: int = 0) -> list[list[tuple[int, int]]]:
    """Return a list of connected components, each as a list of (row, col) cells."""
    rows, cols = grid_shape(grid)
    visited: list[list[bool]] = [[False] * cols for _ in range(rows)]
    components: list[list[tuple[int, int]]] = []

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != background and not visited[r][c]:
                component: list[tuple[int, int]] = []
                _flood_fill_collect(grid, visited, r, c, rows, cols, background, component)
                components.append(component)
    return components


def _flood_fill_collect(
    grid: Grid,
    visited: list[list[bool]],
    r: int,
    c: int,
    rows: int,
    cols: int,
    background: int,
    out: list[tuple[int, int]],
) -> None:
    stack = [(r, c)]
    while stack:
        cr, cc = stack.pop()
        if cr < 0 or cr >= rows or cc < 0 or cc >= cols:
            continue
        if visited[cr][cc] or grid[cr][cc] == background:
            continue
        visited[cr][cc] = True
        out.append((cr, cc))
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            stack.append((cr + dr, cc + dc))


def bounding_box(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    """Return (min_row, min_col, max_row, max_col) for a list of cells."""
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def is_horizontally_symmetric(grid: Grid) -> bool:
    """Return True if the grid is symmetric about its vertical midline."""
    for row in grid:
        if row != row[::-1]:
            return False
    return True


def is_vertically_symmetric(grid: Grid) -> bool:
    """Return True if the grid is symmetric about its horizontal midline."""
    rows, _ = grid_shape(grid)
    half = rows // 2
    for i in range(half):
        if grid[i] != grid[rows - 1 - i]:
            return False
    return True


def has_any_symmetry(grid: Grid) -> bool:
    """Return True if the grid has horizontal or vertical reflection symmetry."""
    return is_horizontally_symmetric(grid) or is_vertically_symmetric(grid)


def crop_to_content(grid: Grid, background: int = 0) -> Grid:
    """Return the smallest sub-grid containing all non-background cells."""
    rows, cols = grid_shape(grid)
    non_bg = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] != background]
    if not non_bg:
        return [[background]]
    min_r, min_c, max_r, max_c = bounding_box(non_bg)
    return [row[min_c : max_c + 1] for row in grid[min_r : max_r + 1]]


def grids_equal(a: Grid, b: Grid) -> bool:
    """Return True if two grids are identical."""
    if grid_shape(a) != grid_shape(b):
        return False
    return all(a[r][c] == b[r][c] for r in range(len(a)) for c in range(len(a[r])))


def grid_similarity(a: Grid, b: Grid) -> float:
    """Return fraction of cells that match between two grids (0.0–1.0).

    If shapes differ, resize to the smaller shape for comparison.
    """
    rows_a, cols_a = grid_shape(a)
    rows_b, cols_b = grid_shape(b)
    rows = min(rows_a, rows_b)
    cols = min(cols_a, cols_b)
    total = rows * cols
    if total == 0:
        return 0.0
    matches = sum(a[r][c] == b[r][c] for r in range(rows) for c in range(cols))
    return matches / total


# ---------------------------------------------------------------------------
# Transformation primitives
# ---------------------------------------------------------------------------


def rotate_90(grid: Grid) -> Grid:
    """Rotate the grid 90 degrees clockwise."""
    rows, cols = grid_shape(grid)
    new_grid = [[0] * rows for _ in range(cols)]
    for r in range(rows):
        for c in range(cols):
            new_grid[c][rows - 1 - r] = grid[r][c]
    return new_grid


def flip_h(grid: Grid) -> Grid:
    """Flip the grid horizontally (left to right)."""
    return [row[::-1] for row in grid]


def flip_v(grid: Grid) -> Grid:
    """Flip the grid vertically (top to bottom)."""
    return grid[::-1]
