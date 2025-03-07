"""Matplotlib interface to CEBRA-Lens."""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
import torch


class _DecodingPlot:
    """Plot the decoding accuracy across multiple models."""
    def __init__(self, results_dict: dict, palette: str, dataset_label: str):
        """
        Initializes the DecodingPlot class.

        Args:
            results_dict (dict): A dictionary where the keys are model category labels or model file names 
                and the values are 2D arrays containing decoding results.
            palette (str, optional): The color palette to use for the plot. Default is "hls".
            dataset_label (str, optional): The dataset type. Currently only "visual" is supported.
        """
        self.results_dict = results_dict
        self.palette = sns.color_palette(palette, len(results_dict))
        self.dataset_label = dataset_label

        self.fig, self.ax = plt.subplots(figsize=(len(results_dict) * 2, 6))

    def _plot(self):
        """Handles plotting logic"""
        x_positions = list(range(1, len(self.results_dict) + 1))

        if self.dataset_label == "visual":
            for i, (key, results) in enumerate(self.results_dict.items()):
                acc = results[:, 2]  # accuracy
                mean_error = np.mean(acc)
                color = self.palette[i]
                self.ax.scatter(np.ones_like(acc) * x_positions[i], acc, color=color, alpha=0.3)

                # Plot the means
                self.ax.scatter(
                    x_positions[i],
                    mean_error,
                    color=color,
                    s=50,
                    label=f"Mean {key}",
                    zorder=5,
                )

            self.ax.set_xlabel("Model")
            self.ax.set_ylabel("Accuracy (%)")
            self.ax.set_title("Comparison of Accuracy Across Models")
            self.ax.set_xticks(x_positions)
            self.ax.set_xticklabels(self.results_dict.keys())
            self.ax.legend()
            sns.despine()
        else:
            raise NotImplementedError(
                f"Plotting of {self.dataset_label} is not handled yet. Only 'visual' is for now. "
            )
    def plot(self)-> plt.Figure:
        """Generates and returns the plot."""
        self._plot()
        return self.fig


def plot_decoding(results_dict: dict,
    palette: str = "hls",
    dataset_label="visual",
    **kwargs,
) -> plt.Figure:
    """
    Plots the decoding accuracy across multiple models.

    Parameters:
    -----------
    results_dict : dict
        A dictionary where the keys are model category labels or model file names and the values are 2d-arrays containing decoding results gathered by lens.quantification.decoding.decode_models.
    palette: str, optional (default is "hls")
        The color palette to use for the plot.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure displaying the comparison of decoding accuracy across models.
    """
    return _DecodingPlot(
        results_dict=results_dict,
        palette = palette,
        dataset_label=dataset_label,
    ).plot(**kwargs)