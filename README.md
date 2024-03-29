# eXirt
This is a Explainable Artificial Intelligence tool focused in ensemble black box models.

# Autors

José Ribeiro - site: https://sites.google.com/view/jose-sousa-ribeiro

Lucas Cardoso - site: http://lattes.cnpq.br/9591352011725008

Raíssa Silva - site: https://sites.google.com/site/silvarailors

Vitor Cirilo - site: https://sites.google.com/site/vitorciriloaraujosantos/

Níkolas Carneiro - site: https://br.linkedin.com/in/nikolas-carneiro-62b6568

Ronnie Alves (Leader) - site: https://sites.google.com/site/alvesrco

# The eXirt

The measure Explainable based on Item Response Theory - *eXirt* is one of the XAI measures performed in the developed benchmark. This measure is a new proposal to generate explanations for tree-ensemble models that is based on the generation of attribute relevance ranks by using item response theory.

Just like other XAI measures, *eXirt* only uses the training data, test data, the model itself together with its outputs, figure 1 (A). Initially, the test data and the model to be explained are passed on to the *eXirt*; Figure 1 (B) the tool then creates the so-called "loop of models", shown in figure 1 (C) and (D), which is a basic iterative process relevant for the creation of the table with the respondent's answers used in the IRT run, figure 1 (E) and (F).

The creation of the response matrix, figure 1 (E) and (F), contains the answers of all the classifiers (respondents). The columns refer to the instances of the dataset that was passed on, while the rows refer to the different classifiers. Values equal to 0 (zero) are wrong answers of the prediction, while values 1 (one) are correct answers of the prediction, regardless of the number of classes that the problem may have. This matrix is used to calculate the values of the item parameters (discrimination, difficulty and guessing) for each of the instances, in figure 1 (G).

The implementation of the IRT used was [*Cardoso, L (2020)*](https://github.com/LucasFerraroCardoso/IRT_OpenML), called decodIRT, in a code developed exclusively for the purpose of this paper, as the code first receives the answer matrix, performs the calculations to generate the item parameter values --- different algorithms can be used to calculate the IRT (such as: ternary, dichotomous, fibonacci, golden, brent, bounded or golden2) using [*catsim*](https://github.com/douglasrizzo/catsim) --- and, after this step, generates the rank of most skilled classifiers, figure  1 (G).

<p align="center">
  <img width="900" height="470" src="https://github.com/josesousaribeiro/eXirt/blob/main/figs/eXirt_prepross.png">
</p>
**Figure 1 - Process of the eXirt.**

Among the new features of *decordIRT* is a new score calculation that involves the calculated ability of all respondents and their respective hits and misses, called Total Score. Total Score can be understood as an adaptation of the *True-Score*, whereby the score is calculated by summing up all the hit probabilities for the test items. However, in cases where respondents have a very close ability, the True-Score result can be very similar or even equal, since only the hit chance is considered. To avoid equal score values and to give more robustness to the models' final score, the Total Score also considers the respondent's probability of error, given by: $1- P(U_{ij} = 1\vert\theta_{j})$. Thus, every time the model gets it right, the hit probability is added, and if the model gets it wrong, the error probability is subtracted.

The Total Score resulting from the execution of decodIRT is not yet the final rank of explainability of the model, because in this case it is necessary to calculate the average of the skills found for each attribute, figure 1 (H), involving the different variations and combinations of attributes used in the previous steps.

Ultimately, figure 1 (I) and (J), an explanation rank is generated where each attribute appears with a skill value. In this case, the lower the ability values, the more the attribute explains the analyzed model.

# Installation

You must have installed the R environment in local executions. Then install from the project link: https://cloud.r-project.org/.

Now, just run the commands below in a python environment and you will have your installation normally.

Installation of the "catsim" dependency.

```Nushell
pip install catsim
```

Installation of the "decodIRT" dependency.

```Nushell
!wget https://raw.githubusercontent.com/josesousaribeiro/eXirt-XAI-Benchmark/main/decodIRT/decodIRT_MLtIRT.py
!wget https://raw.githubusercontent.com/josesousaribeiro/eXirt-XAI-Benchmark/main/decodIRT/decodIRT_analysis.py
```

Installation of the "eXirt"

```Nushell
pip install eXirt
```




# Import in code

```Python
from pyexirt.eXirt import Explainer
```

# Create explainer

```Python
explainer = Explainer()
global_explanation_attributes, global_explanation_attributes_scores =
      explainer.explainRankByEXirt(model, X_train, X_test, y_train, y_test,dataset_name)
```
Note:

*global_explanation_attributes = name of relevance features*

*global_explanation_attributes_scores = rank of relevance features with score*

# Relevance rank 

The result of running the eXirt is an attribute relevance rank, sorted in ascending order with respect to the calculated skill. The lower the ability of an attribute, the greater its relevance for explaining the model. In figure 2 is presented for the *phoneme* dataset.

<p align="center">
  <img width="460" height="300" src="https://github.com/josesousaribeiro/eXirt/blob/main/figs/eXirt_phoneme.png">
</p>

**Figure 2 - Result of eXirt to *phoneme* dataset.**


# Item Characteristic Curve of Models

This is an example of the Item Characteristic Curve for 4 different models.
Note, the generated item parameter values are condensed by their averages in the thick black line.

<p align="center">
  <img width="650" height="400" src="https://github.com/josesousaribeiro/eXirt/blob/main/figs/icc_example.png">
</p>


# Cite this work

You can cite the package using the following bibtex entry:

```TeX

@article{de2024explanations,
  title={Explanations based on Item Response Theory (eXirt): A model-specific method to explain tree-ensemble model in trust perspective},
  author={de Sousa Ribeiro Filho, Jos{\'e} and Cardoso, Lucas Felipe Ferraro and da Silva, Ra{\'\i}ssa Lorena Silva and Carneiro, Nikolas Jorge Santiago and Santos, Vitor Cirilo Araujo and de Oliveira Alves, Ronnie Cley},
  journal={Expert Systems with Applications},
  volume={244},
  pages={122986},
  year={2024},
  publisher={Elsevier}
}

````
