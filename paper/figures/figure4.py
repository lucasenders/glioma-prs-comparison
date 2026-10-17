"""Figure 4: paired difference in mean test AUC between each ML or DL model and the standard PRS on the same feature set. Output goes to Figure4.svg and Figure4.png beside this script.

Run with: python3 figure4.py
"""

import heatmap

# Draw the heat map of each model minus the standard PRS
MODELS = ["elastic_net", "random_forest", "xgboost_model", "svm", "mlp", "mlp_narrow", "mlp_shallow", "mlp_small"]
heatmap.draw(MODELS, "standard", "Difference in mean test AUC, model minus standard PRS", "Figure4", 10.5)
