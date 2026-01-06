"""
Functions to help with plotting
"""

import warnings
from typing import Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure


def dark_mode(
    fig: Figure,
    ax: Union[Axes, np.ndarray],
    cbars: Optional[Sequence[Colorbar]] = None,
    background_color: str = "black",
    secondary_color: str = "white",
) -> Tuple[Figure, Union[Axes, np.ndarray]]:
    """
    Apply dark mode styling to matplotlib figure and axes.

    Args:
        fig: Matplotlib figure
        ax: Single axes or array of axes
        cbars: Optional sequence of colorbars to style
        background_color: Background color for dark mode
        secondary_color: Text and line color for dark mode

    Returns:
        Tuple of styled figure and axes
    """
    try:
        fig.patch.set_facecolor(background_color)

        # Handle suptitle if it exists
        supt_text = fig.get_suptitle()
        if len(supt_text) > 0:
            fig.suptitle(supt_text, color=secondary_color)

        # Apply styling to axes
        axes_to_style = ax.ravel() if isinstance(ax, np.ndarray) else [ax]

        for a in axes_to_style:
            a.set_facecolor(background_color)
            plt.setp(a.spines.values(), color=secondary_color)
            a.tick_params(axis="both", colors=secondary_color)
            a.xaxis.label.set_color(secondary_color)
            a.yaxis.label.set_color(secondary_color)
            a.title.set_color(secondary_color)

        # Style colorbars if provided
        if cbars is not None:
            for cbar in cbars:
                cbar.ax.yaxis.set_tick_params(color=secondary_color)
                plt.setp(cbar.ax.get_yticklabels(), color=secondary_color)

        return fig, ax

    except Exception as e:
        warnings.warn(f"Failed to apply dark mode styling: {e}")
        return fig, ax
