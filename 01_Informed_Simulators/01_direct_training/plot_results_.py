from matplotlib import pyplot as plt
import os

def get_plot_path(path):
    directory_of_this_file = '/'.join(path.split('/')[:-1])
    file_name_no_ext = os.path.basename(path).split('/')[-1].split('.')[0]
    plot_path = os.path.join(directory_of_this_file, file_name_no_ext)
    return plot_path

def plot_results(t, z, z_ref, path):
    fig = plt.figure()
    x_ax, v_ax = fig.subplots(2,1)
    x_ax.set_title('Position')
    x_ax.plot(t, z_ref[:,0], label='ref')
    v_ax.plot(t, z_ref[:,1], label='ref')
    v_ax.set_title('Velocity')
    x_ax.plot(t, z[:,0], label='sol')
    v_ax.plot(t, z[:,1], label='sol')
    x_ax.legend()
    v_ax.legend()
    fig.tight_layout()
    fig.savefig(f'{path}.png')
    plt.show()

def plot_results_advanced():
    pass

