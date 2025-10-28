import numpy as np
from matplotlib.backend_bases import MouseButton

class LineDrawer:
    def __init__(self, line):
        self.line = line[0]
        self.xs = self.line.get_xdata()
        self.ys = self.line.get_ydata()
        self.cid = self.line.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        print('click', event)
        if event.button is MouseButton.LEFT and event.inaxes == self.line.axes:
        # Capture pixel coordinates relative to the image
            self.xs = np.append(self.xs, int(event.xdata))
            self.ys = np.append(self.ys, int(event.ydata))
            # Draw a cross at the clicked location (X marker)
            self.line.axes.scatter(self.xs, self.ys, color="red", s=100, marker="x", label="Click")
            self.line.set_data(self.xs, self.ys)
            self.line.figure.canvas.draw()

def extremites(full_Sequence):
    deb = 0
    fin = len(full_Sequence) - 1
    while full_Sequence[deb] != 1:
        deb += 1
    while full_Sequence[fin] != 1:
        fin -= 1
    return deb, fin+1


