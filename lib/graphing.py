from lib.setup import *

#creates bar graph with desired specifications
def graph_bar(heights, x_labels, title, y_title, save):
    x = [i for i in range(len(heights))]
    fig, ax = plt.subplots()
    plt.bar(x, heights)
    ax.set_ylabel(y_title)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_title(title)
    fig.tight_layout()
    fig.set_size_inches(12, 12)
    fig.savefig(os.path.join("charts",save) + '.png')
    plt.show()

def graph_matrix(data, x_labels, title, save):
    fig, ax = plt.subplots()
    im = ax.imshow(data)
    if x_labels:
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        ax.set_xticklabels(x_labels, rotation=-90)
        ax.set_yticklabels(x_labels)
    
    ax.set_title(title)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("", rotation=-90, va="bottom")
    fig.tight_layout()
    fig.set_size_inches(12, 12)
    fig.savefig(os.path.join("charts", save) + '.png')
    plt.show()

