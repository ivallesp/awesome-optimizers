from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from autograd import grad

def plot_contourf(cost_f, figsize=[10, 10], _show=True):
    fig = plt.figure(figsize=figsize)
    ax=fig.gca()
    x = np.arange(cost_f.xmin, cost_f.xmax, 0.1)
    y = np.arange(cost_f.ymin, cost_f.ymax, 0.1)
    X, Y = np.meshgrid(x, y)
    zs = np.array([cost_f.eval(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    Gx, Gy = np.gradient(Z) # gradients with respect to x and y
    G = (Gx**2.0+Gy**2.0)**.5  # gradient magnitude
    N = G/G.max()  # normalize 0..1
    ax.contourf(X, Y, Z, cmap=plt.cm.get_cmap(plt.cm.afmhot), levels=np.linspace(zs.min(), zs.max(), 1000))
    plt.text(cost_f.x_optimum, cost_f.y_optimum,"x", color="b", size=20)
    if _show:
        plt.show()
    return fig, ax


def plot_trajectories(trajectories_dict, cost_f, figsize=[10,10],
                     filepath="test.gif", frames=10):
    
    fig, ax = plot_contourf(cost_f=cost_f, figsize=[10, 10], _show=False)
    
    global dots; dots=[]
    def update(frame_number):
        global dots
        ax=fig.gca()
        for sc in dots: sc.remove()
        dots=[]
       
        for name, (x, y, c) in trajectories_dict.items():
            ax.plot(x[:frame_number], y[:frame_number], color=c, zorder=1, linewidth=2)
            k=ax.scatter(x[frame_number], y[frame_number], color=c, zorder=1, s=50)
            dots.append(k)

        plt.legend(trajectories_dict.keys())

    animation = FuncAnimation(fig, update, interval=1, frames=frames);
    animation.save(filepath, dpi=80, writer = "imagemagick");
    
    
def plot_cost_function_3d(cost_f, grain=0.01, figsize=[10,6]):
    x_grid = np.arange(cost_f.xmin, cost_f.xmax, grain)  # Controls the X region covered by the mesh grid
    y_grid = np.arange(cost_f.ymin, cost_f.ymax, grain)  # Controls the Y region covered by the mesh grid

    X, Y = np.meshgrid(x_grid, y_grid)
    Z = np.array([cost_f.eval(x_grid,y_grid) for x_grid,y_grid in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)
    Gx, Gy = np.gradient(Z)  # gradients with respect to x and y

    fig = plt.figure(figsize=figsize)
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm, antialiased=False, shade=False, 
                    rstride=5, cstride=1, linewidth=0, alpha = 1)
    ax.patch.set_facecolor('white')
    ax.view_init(elev=30., azim=70)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.xaxis.labelpad = ax.yaxis.labelpad = ax.zaxis.labelpad = 15
    plt.show()
    
def plot_evolution_charts(cost_f, errors, distance, xs, ys):
    plt.figure(figsize=[18,6])
    plt.plot(errors)
    plt.title("Error (Z axis) evolution over time. Minimum error obtained in {0} iterations: {1}".format(len(errors), min(errors)))
    plt.xlabel("time (iterations)")
    plt.ylabel("error")
    plt.grid()
    plt.show()

    plt.figure(figsize=[18,6])
    plt.semilogy(np.abs(np.array(errors) - cost_f.z_optimum))
    plt.title("Log-error (Z axis) evolution over time. Minimum error obtained in {0} iterations: {1}".format(len(errors), min(errors)))
    plt.xlabel("time (iterations)")
    plt.ylabel("log(error)")
    plt.grid()
    plt.show()

    plt.figure(figsize=[18,6])
    plt.plot(distance)
    plt.ylim([0, max(distance)])
    plt.title("Distance to minimum evolution over time. Minimum distance obtained in {0} iterations: {1}".format(len(errors), min(errors)))
    plt.xlabel("time (iterations)")
    plt.ylabel("distance")
    plt.grid()
    plt.show()
    

def plot_cinematics_charts(xs, ys):
    plt.figure(figsize=[18,6])
    plt.subplot(131)
    plt.plot(xs)
    plt.title("X parameter evolution")
    plt.xlabel("iterations")
    plt.ylabel("x")
    plt.grid()
    plt.subplot(132)
    plt.plot(ys)
    plt.title("Y parameter evolution")
    plt.xlabel("iterations")
    plt.ylabel("y")
    plt.grid()
    plt.subplot(133)
    plt.plot(xs, ys)
    plt.title("x/y evolution")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.show()

    vel_xs=np.diff(xs)
    vel_ys=np.diff(ys)
    plt.figure(figsize=[18,6])
    plt.subplot(131)
    plt.plot(np.abs(vel_xs))
    plt.title("X parameter velocity (momentum; $v_x$)")
    plt.xlabel("iterations")
    plt.ylabel("vx")
    plt.ylim(0,1.05*max(vel_xs))
    plt.grid()

    plt.subplot(132)
    plt.plot(np.abs(vel_ys))
    plt.title("Y parameter velocity (momentum; $v_y$)")
    plt.xlabel("iterations")
    plt.ylabel("vy")
    plt.ylim(0,1.05*max(vel_ys))
    plt.grid()

    plt.subplot(133)
    plt.plot(np.sqrt(np.array(vel_xs)**2 + np.array(vel_ys)**2))
    plt.title("Absolute velocity ($\sqrt{v_x^2 + v_y^2}$)")
    plt.xlabel("iterations")
    plt.ylabel("v")
    plt.ylim(0,1.05*max(np.sqrt(np.array(vel_xs)**2 + np.array(vel_ys)**2)))
    plt.grid()
    plt.show()