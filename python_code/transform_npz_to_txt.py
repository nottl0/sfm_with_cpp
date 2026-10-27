import numpy as np

d = np.load("../scan65/cameras.npz")

with open("../scan65/cameras.txt", "w") as f:
    for k in d.files:
        values = " ".join(f"{x:.17g}" for x in d[k].ravel())
        f.write(f"{k} {values}\n")