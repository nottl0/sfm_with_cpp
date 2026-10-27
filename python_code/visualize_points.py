import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../check.csv")

# Center the point cloud at its mean
x = df.x - df.x.mean()
y = df.y - df.y.mean()
z = df.z - df.z.mean()

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(x, y, z, s=1)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Center the axes limits around zero with equal scale
max_range = max(
    x.max() - x.min(),
    y.max() - y.min(),
    z.max() - z.min()
) / 2

ax.set_xlim(-max_range, max_range)
ax.set_ylim(-max_range, max_range)
ax.set_zlim(-max_range, max_range)
ax.set_box_aspect([1, 1, 1])

plt.show()