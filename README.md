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

The measure Explainable based on Item Response Theory - \textit{eXirt} is one of the XAI measures performed in the developed benchmark. This measure is a new proposal to generate explanations for tree-ensemble models that is based on the generation of attribute relevance ranks by using item response theory.

Just like other XAI measures, \textit{eXirt} only uses the training data, test data, the model itself together with its outputs, figure \ref{fig_metodologia_eXirt} (A). Initially, the test data and the model to be explained are passed on to the \textit{eXirt}; \ref{fig_metodologia_eXirt} (B) the tool then creates the so-called ``loop of models,'' shown in figure \ref{fig_metodologia_eXirt} (C) and (D), which is a basic iterative process relevant for the creation of the table with the respondents' answers used in the IRT run, figure \ref{fig_metodologia_eXirt} (E) and (F).

The ``loop of models'' process, figure \ref{fig_metodologia_eXirt} (D) is inspired on how the measures XAI Dalex \cite{dalex_book} and Lofo \cite{lofo_ref} work, for in this process iterative variations are performed on different sets of attributes of the model, and at each iteration the answers of the classifier prediction are collected and, thus, it is possible to have a high number of responding candidates. It is noteworthy that each classifier is different from the other, as one presents different inputs and one or more attributes of their inputs.

Still referring to the ``loop of models'', 12 different ways of varying the attributes of the models have been implemented, figure  \ref{fig_metodologia_eXirt} (C). An important detail is that variations can be combined, that is, they can occur in more than one attribute of the model in the same iteration. However, the standard use of 2 variations and combinations of up to 2 in parallel is indicated, so as to reduce the computational cost for datasets with high amounts of attributes. 

The implemented variations are nothing more than different types of noise and interference that are inserted into the input values of the model, test set, such as: permutation of the indices, application of noise, modification of value to 0, application of normalization to scale different from that in the data, ordering of values and their indices in an ascending manner, ordering of values and their indices in a descending manner, inversion of index positions (from top to bottom), binning processing of values, multiplication of values by -1, replacement of values by the mean, replacement of values by the standard deviation, and standardization of values.

The layout in figure \ref{fig_metodologia_eXirt} (D) shows several computational models only for ludic reasons, as only one computational model is used, varying its inputs iteratively. 

In the results presented, only multiplication by -1 and the application of binning processes were used, as they have low computational costs.

\begin{figure*}[!h]
\begin{center}
\includegraphics[scale=0.45]{eXirt.pdf}
\caption{Visual summary of all steps and processes performed by the XAI measure, called \textit{eXirt}.}
\label{fig_metodologia_eXirt}
\end{center}
\end{figure*}

The creation of the response matrix, figure \ref{fig_metodologia_eXirt} (E) and (F), contains the answers of all the classifiers (respondents). The columns refer to the instances of the dataset that was passed on, while the rows refer to the different classifiers. Values equal to 0 (zero) are wrong answers of the prediction, while values 1 (one) are correct answers of the prediction, regardless of the number of classes that the problem may have. This matrix is used to calculate the values of the item parameters (discrimination, difficulty and guessing) for each of the instances, in figure \ref{fig_metodologia_eXirt} (G).

The implementation of the IRT used was \cite{cardoso2020decoding_irt}, called decodIRT, in a code developed exclusively for the purpose of this paper, as the code first receives the answer matrix, performs the calculations to generate the item parameter values --- different algorithms can be used to calculate the IRT (such as: ternary, dichotomous, fibonacci, golden, brent, bounded or golden2) \cite{catsim} --- and, after this step, generates the rank of most skilled classifiers, figure  \ref{fig_metodologia_eXirt} (G).

Among the new features of \textit{decordIRT} is a new score calculation that involves the calculated ability of all respondents and their respective hits and misses, called Total Score. Total Score can be understood as an adaptation of the True-Score \cite{lord1984comparison}, whereby the score is calculated by summing up all the hit probabilities for the test items. However, in cases where respondents have a very close ability, the True-Score result can be very similar or even equal, since only the hit chance is considered. To avoid equal score values and to give more robustness to the models' final score, the Total Score also considers the respondent's probability of error, given by: $1- P(U_{ij} = 1\vert\theta_{j})$. Thus, every time the model gets it right, the hit probability is added, and if the model gets it wrong, the error probability is subtracted. The calculation of the Total Score $t_l$ is defined by the following equation \ref{eqn:total_score}, where $i'$ corresponds to the set of items answered correctly, and $i''$ corresponds to the set of items answered incorrectly.

\begin{equation}
\label{eqn:total_score}
t_l = \sum_{i=1}^{i'} P(U_{ij} = 1\vert\theta_{j}) - \sum_{i=1}^{i''} 1 - P(U_{ij} = 1\vert\theta_{j})
\end{equation}

In this regard, a skilled model with high hit probability that ends up getting an item wrong will not have its score heavily discounted. However, for a low ability model with low hit probability, the error will result in a greater discount of the score value. For, it is understood that the final score value should consider both the estimated ability of the respondent and his/her own performance on the test.

The Total Score resulting from the execution of decodIRT is not yet the final rank of explainability of the model, because in this case it is necessary to calculate the average of the skills found for each attribute, figure \ref{fig_metodologia_eXirt} (H), involving the different variations and combinations of attributes used in the previous steps.

Ultimately, figure \ref{fig_metodologia_eXirt} (I) and (J), an explanation rank is generated where each attribute appears with a skill value. In this case, the lower the ability values, the more the attribute explains the analyzed model. Equation $T_{(f,r)}$ is presented, which represents the processes performed by \textit{eXirt}, equation \ref{eqn:exirt_sigma}. 

# Installation

pip install eXirt

# Import in code
from eXirt import eXirt

# Create explainer

explainer = eXirt.eXirt()
global_explanation_attributes, global_explanation_attributes_scores = explainer.explainRankByEXirt(model, X_train, X_test, y_train, y_test,dataset_name)
