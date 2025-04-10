import os
import sys
from ensemble_model import main as run_ensemble

# Add the project root to the path so we can import HF modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from HF.utils.visualization_utils import set_plot_style

if __name__ == "__main__":
    # Set the plot style
    set_plot_style()
    
    # Run the ensemble model
    run_ensemble()
