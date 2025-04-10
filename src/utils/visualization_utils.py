import os
import matplotlib.pyplot as plt
import seaborn as sns
from HF.logger import logging

def save_figure(fig, filename, dpi=300, bbox_inches='tight', visualizations_dir='visualizations'):
    """
    Save a matplotlib figure to the visualizations directory.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure to save
    filename : str
        The filename (without path)
    dpi : int, default=300
        The resolution in dots per inch
    bbox_inches : str, default='tight'
        Bounding box in inches
    visualizations_dir : str, default='visualizations'
        Directory to save visualizations
    
    Returns:
    --------
    str
        The full path to the saved figure
    """
    try:
        # Create visualizations directory if it doesn't exist
        os.makedirs(visualizations_dir, exist_ok=True)
        
        # Construct the full path
        filepath = os.path.join(visualizations_dir, filename)
        
        # Save the figure
        fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
        logging.info(f"Figure saved to {filepath}")
        
        return filepath
    except Exception as e:
        logging.error(f"Error saving figure: {e}")
        # Save to current directory as fallback
        filepath = filename
        fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
        logging.warning(f"Figure saved to current directory: {filepath}")
        return filepath

def save_plot(filename, dpi=300, bbox_inches='tight', visualizations_dir='visualizations'):
    """
    Save the current matplotlib plot to the visualizations directory.
    
    Parameters:
    -----------
    filename : str
        The filename (without path)
    dpi : int, default=300
        The resolution in dots per inch
    bbox_inches : str, default='tight'
        Bounding box in inches
    visualizations_dir : str, default='visualizations'
        Directory to save visualizations
    
    Returns:
    --------
    str
        The full path to the saved figure
    """
    try:
        # Create visualizations directory if it doesn't exist
        os.makedirs(visualizations_dir, exist_ok=True)
        
        # Construct the full path
        filepath = os.path.join(visualizations_dir, filename)
        
        # Save the current figure
        plt.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
        logging.info(f"Plot saved to {filepath}")
        
        return filepath
    except Exception as e:
        logging.error(f"Error saving plot: {e}")
        # Save to current directory as fallback
        filepath = filename
        plt.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
        logging.warning(f"Plot saved to current directory: {filepath}")
        return filepath

def set_plot_style(style='whitegrid', context='notebook', palette='deep', font_scale=1.2):
    """
    Set the seaborn plot style.
    
    Parameters:
    -----------
    style : str, default='whitegrid'
        The seaborn style
    context : str, default='notebook'
        The seaborn context
    palette : str, default='deep'
        The seaborn color palette
    font_scale : float, default=1.2
        The font scale
    """
    sns.set_style(style)
    sns.set_context(context, font_scale=font_scale)
    sns.set_palette(palette)
    
    # Set matplotlib params for better readability
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    
    logging.info("Plot style set successfully")
