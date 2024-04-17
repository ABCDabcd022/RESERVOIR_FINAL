import matplotlib.pyplot as plt
import numpy as np

# Generate some sample data
x = np.linspace(0, 20, 201)
y = np.sin(x)

# Create the plot
fig, ax = plt.subplots()
ax.plot(x, y)

# Calculate the x-coordinate of the point 10 values before the end
x_end = x[-1]
x_start = x_end - 10
x_line = (x_start + x_end) / 2

# Add the vertical line
ax.axvline(x=x_line, color='red')

# Show the plot
plt.show()