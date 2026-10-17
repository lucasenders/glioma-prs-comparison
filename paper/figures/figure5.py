"""Figure 5: paired difference in mean test AUC between each non-linear model and elastic net on the same feature set. Output goes to Figure5.svg and Figure5.png beside this script.

Run with: python3 figure5.py
"""

import heatmap

# Draw the heat map of each non-linear model minus elastic net
MODELS = ["random_forest", "xgboost_model", "mlp", "mlp_narrow", "mlp_shallow", "mlp_small"]
heatmap.draw(MODELS, "linear", "Difference in mean test AUC, model minus elastic net", "Figure5", 10)
