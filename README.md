# eXirt
This is a Explainable Artificial Intelligence tool focused in ensemble black box models.

# Installation

pip install eXirt

# Import in code
from eXirt import eXirt

# Create explainer

explainer = eXirt.eXirt()
global_explanation_attributes, global_explanation_attributes_scores = explainer.explainRankByEXirt(model, X_train, X_test, y_train, y_test,dataset_name)
