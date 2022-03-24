
from collections import deque

def binary_space_partitioning(space, np: int):
    # np: the number of partitions
    # partitions list initialized with a single partition
    P = deque([(0, space[:])])
    N = len(space)
    di = 0

    while np > 1:
        di, p = P.popleft()

        # two new partitions
        p1 = p[:]
        p2 = p[:]

        l, u = p[di] # the lower and upper bound
        m = (u+l)/2 # middle

        p1[di] = (l,m)
        p2[di] = (m,u)

        di = (di + 1) % N
        P.append((di, p1))
        P.append((di, p2))

        np -= 1

    P = [p for _, p in P]
    return P


if __name__ == "__main__":
    D = 2
    space = [[0, 1] for i in range(D)]
    P = binary_space_partitioning(space, np=5)
    for p in P:
        print(p)