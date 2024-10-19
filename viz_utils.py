import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.lines import Line2D

def make_and_viz_dag(leaf_node):
    # TODO: Use package DAFT to visualize the DAG
    raise NotImplementedError("This function is not implemented yet")


def visualize_attention_matrix(matrix, start_end_pos, names, grid_spacing=0.05,
                               size_ratio=1.0, show_legend=False, save_dir=None):
    assert matrix.shape[0] == matrix.shape[1], "Matrix must be square"
    
    examples = [
        (names[i], start_end_pos[i][0], start_end_pos[i][1] - 1)
        for i in range(len(start_end_pos))
    ]

    D = matrix.shape[0]
    
    fig, axes = plt.subplots(D, D, figsize=(D * size_ratio, D * size_ratio))
    fig.subplots_adjust(hspace=grid_spacing, wspace=grid_spacing)
    
    colors = ['gray', 'steelblue']  # Light salmon for 0, pale green for 1
    
    for i in range(D):
        for j in range(D):
            ax = axes[i, j]
            color = colors[matrix[i, j]]
            ax.add_patch(patches.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='white', linewidth=2))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal', 'box')
    
    def add_bracket(fig, ax1, ax2, direction, text, color='red'):
        if direction == 'left':
            x1, y1 = ax1.get_position().x0, ax1.get_position().y1
            x2, y2 = ax2.get_position().x0, ax2.get_position().y0
            mx, my = x1 - 0.02, (y1 + y2) / 2
            
            line1 = Line2D([mx, mx, mx], [y1, y1, y2], color=color, linewidth=2)
            line2 = Line2D([mx, mx + 0.005], [y1, y1], color=color, linewidth=2)
            line3 = Line2D([mx, mx + 0.005], [y2, y2], color=color, linewidth=2)
            fig.add_artist(line1)
            fig.add_artist(line2)
            fig.add_artist(line3)
            
            fig.text(mx - 0.02, my, text, ha='right', va='center', rotation=90, fontsize=10)
            
        elif direction == 'top':
            x1, y1 = ax1.get_position().x0, ax1.get_position().y0
            x2, y2 = ax2.get_position().x1, ax2.get_position().y0
            mx, my = (x1 + x2) / 2, y2 - 0.02
            line1 = Line2D([x1, x1, x2], [my, my, my], color=color, linewidth=2)
            line2 = Line2D([x1, x1], [my, my + 0.005], color=color, linewidth=2)
            line3 = Line2D([x2, x2], [my, my + 0.005], color=color, linewidth=2)
            fig.add_artist(line1)
            fig.add_artist(line2)
            fig.add_artist(line3)
            
            fig.text(mx, my - 0.04, text, ha='center', va='bottom', fontsize=10)
    
    for desc, start, end in examples:
        add_bracket(fig, axes[start, 0], axes[end, 0], 'left', desc)
    
    for desc, start, end in examples:
        add_bracket(fig, axes[-1, start], axes[-1, end], 'top', desc)
    
    # Remove all spines
    for ax in axes.flat:
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    # Add a legend
    legend_elements = [patches.Patch(facecolor=colors[0], edgecolor='white', label='0'),
                       patches.Patch(facecolor=colors[1], edgecolor='white', label='1')]
    if show_legend:
        fig.legend(handles=legend_elements, loc='upper right', title="Binary Values")
    
    plt.suptitle("Binary Matrix Visualization", fontsize=16, y=0.98)
    if save_dir is not None:
        plt.savefig(save_dir)
    plt.show()
    