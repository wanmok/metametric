from collections.abc import Iterator

import numpy as np
from jaxtyping import Float


def iterative_max_matching(cost: Float[np.ndarray, "nx ny"]) -> Iterator[tuple[float, list[tuple[int, int, float]]]]:
    """Maximum-weight bipartite matching via the Hungarian algorithm."""
    w = np.asarray(cost)
    nx, ny = w.shape

    # Ensure nx <= ny by transposing if necessary; map back later.
    transposed = False
    if nx > ny:
        w = w.T
        nx, ny = w.shape
        transposed = True

    # 0-based dual potentials
    u = np.zeros(nx)
    v = np.zeros(ny)

    # pred[j] = i means column j is matched to row i; -1 if unmatched
    pred = np.full(ny, -1, dtype=int)

    for i in range(nx):
        # Alternating tree rooted at dummy column alt_tree_root
        alt_tree_root = -1
        j0 = alt_tree_root
        minv = np.full(ny, np.inf)
        used = np.zeros(ny, dtype=bool)
        # way[j] = predecessor column of j in alternating tree; alt_tree_root (-1) for root
        way = np.full(ny, -2, dtype=int)  # -2 = uninitialized; will hold predecessors or -1

        # Build the alternating tree
        while True:
            # Determine current row at the frontier
            i0 = i if j0 == alt_tree_root else pred[j0]

            # Vectorized search for best next column j1 by reduced cost
            cur = -w[i0, :] - u[i0] - v  # maximize -> minimize by negating
            not_used = ~used
            improved = not_used & (cur < minv)
            if improved.any():
                minv[improved] = cur[improved]
                way[improved] = j0
            masked_minv = np.where(not_used, minv, np.inf)
            j1 = int(masked_minv.argmin())
            delta = masked_minv[j1]

            # Update dual potentials
            if used.any():
                v[used] -= delta
            used_rows_mask = used & (pred != -1)
            if used_rows_mask.any():
                u[pred[used_rows_mask]] += delta
            minv[~used] -= delta
            u[i] += delta  # Account for the dummy root

            j0 = j1
            if pred[j0] == -1:  # Advance to j1; if unmatched, we will augment
                break
            used[j0] = True  # Otherwise, include j0 in the tree and continue

        # Augment along the alternating path back to the root
        while True:
            j_prev = way[j0]
            i_prev = i if j_prev == alt_tree_root else pred[j_prev]
            pred[j0] = i_prev
            j0 = j_prev
            if j0 == alt_tree_root:
                break

        # Build result in original orientation
        total = 0.0
        matches = []
        for j in range(ny):
            i = pred[j].item()
            if i != -1:
                ii, jj = (j, i) if transposed else (i, j)
                s = cost[ii, jj].item()
                matches.append((ii, jj, s))
                total += s
        yield total, matches
