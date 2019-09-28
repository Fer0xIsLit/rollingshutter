from matplotlib import pyplot as plt, style, animation
import numpy as np

def transition(a, b, t):
    if t >= 1:
        return b
    elif t <= 0:
        return a
    alpha = 6 * (b - a)
    return alpha * (0.5*(t**2) - 1/3 * (t**3)) + a

lims = (-2, 2)

blades = 3
omega = 1.2
phi = 2.1
v = 1
y0 = 1

mag = 5

green = [i/255 for i in (47, 207, 80,)]

class Line:
    def __init__(self, prop=False, new=True):
        self.new = new
        self.finished = False
        if prop:
            return
        self.obj, = ax.plot([], [], color=green)
    
    def set_data(self, x, y):
        self.new = False
        self.obj.set_data(x, y)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111)

ax.set_xlim(*lims)
ax.set_ylim(*lims)

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.axis('off')

ax.set_facecolor('k')
fig.set_facecolor('k')

line_alpha = 0.7
lines = [ax.plot([], [], color='w', alpha=line_alpha, zorder=21)[0] for i in range(blades)]
retena, = ax.plot([], [], color='w')

points = [ax.scatter([], [], color='w', zorder=20) for i in range(blades)]

offsets = np.array([i*np.pi/blades for i in range(blades)])

contact_lines = [[Line(new=False),] for i in range(blades)]

RANGE = (-1.5, 1.5)

a, b = omega/v * y0 + phi, omega/v
bg_lines = []
bg_alpha = 0.1
buffer = 0.01

for i in range(blades):
    n_lims = [int(np.floor((-a-offsets[i] + b*lims[0])/np.pi)), int(np.ceil((-a-offsets[i] + b*lims[1])/np.pi))]
    n_lims[-1] += 2

    for m in np.arange(*n_lims):
        Y = np.linspace(buffer + (a+offsets[i]+m*np.pi)/b, (a+offsets[i]+(m+1)*np.pi)/b - buffer, 64)
        X = Y / np.tan(a-b*Y + offsets[i])
        bg_lines.append(ax.plot(X, Y, color=green, alpha=bg_alpha)[0])

t = -0.4
dt = 0.007
dT = 0.001
T_start = [t]*blades
T_end = None



def animate(frame):
    global t, T_start, T_end

    y = y0 - v * t

    if y < lims[0]:
        if not T_end:
            T_end = t
        progress = transition(0, 1, (t-T_end)/0.245)
        for i in range(blades):
            lines[i].set_alpha(line_alpha*(1-progress))
        for i in range(len(bg_lines)):
            bg_lines[i].set_alpha(bg_alpha*(1-progress))
  
    theta = omega * t + phi
    thetas = theta + offsets



    for i in range(blades):
        lines[i].set_data([RANGE[0] * np.cos(thetas[i]), RANGE[1] * np.cos(thetas[i])], [RANGE[0] * np.sin(thetas[i]), RANGE[1] * np.sin(thetas[i])])
    retena.set_data([-mag, mag], [y, y])


    k = y/np.sin(thetas)
    withins = (k >= RANGE[0]) & (k <= RANGE[1])
    for i in range(blades):
        if not withins[i]:
            if not contact_lines[i][-1].new:
                T_start[i] = None
                points[i].set_alpha(0)
                contact_lines[i].append(Line())
        else:
            if not T_start[i]:
                T_start[i] = t
                points[i].set_alpha(1)

            p_x = y / np.tan(thetas[i])
            
            points[i].set_offsets(np.c_[p_x, y])

            T = np.arange(T_start[i], t, dT)
            X = (y0-v*T) / np.tan(omega*T+phi+offsets[i])
            Y = (y0-v*T)
            contact_lines[i][-1].set_data(X, Y)
    
    

    t += dt


ani = animation.FuncAnimation(fig, animate, interval=20, frames=np.arange(618))

plt.show()
