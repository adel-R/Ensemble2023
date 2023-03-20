<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![MIT License][license-shield]][license_url]

<h3 align="center">Tous Ensemble - Ensemble Learning project</h3>
  <p align="center">
    This project is the final assignment of the Ensemble Learning class of 2023 at CentraleSupélec as part of the Master in Data Sciences & Business Analytics. It consists in 2 parts:
  <ul>
    <li>Predict Airbnb Prices in New York City using several ensemble methods seen in class.</li>
    <li>>Implement a Decision Tree from scratch on Python allowing to deal with both a regression task and a classification task.</li>
  </ul>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

[![Airbnb Prices in New York City prediction][img/Airbnb_NYC-prices.jfif]](https://github.com/adel-R/Ensemble2023/blob/main/global_results.ipynb)
[![Decision Tree from scratch on Python][img/decision_tree_from_scratch-viz.jfif]](https://github.com/adel-R/Ensemble2023/blob/main/decision_tree_from_scratch/Test_of_homemade_decision_tree.ipynb)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* [![Pandas][Pandas]][Pandas_url]
* [![Numpy][Numpy]][Numpy_url]
* [![Scikit-learn][Scikit-learn]][Scikit-learn_url]
* [![Spacy][Spacy]][Spacy_url]
* [![Matplotlib][Matplotlib]][Matplotlib_url]
* [![Seaborn][Seaborn]][Seaborn_url]
* [![Graphviz][Graphviz]][Graphviz_url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- INSTALLATION -->
## Installation

1. Clone the repo
   ```sh
   git clone https://github.com/adel-R/Ensemble2023
   ```
2. Install the packages contained in the ```requirements.txt``` file
* Unix/macOS
   ```sh
   python -m pip install -r requirements.txt
   ```
* Windows
   ```sh
   py -m pip install -r requirements.txt
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

### Decision Tree from scratch
```python
Final_Results = pd.DataFrame(all_scores)
Final_Results = Final_Results[['Model', 'R2', 'MAE', 'MSE', 'RMSE', 'MAPE', 'error_ratio_rmse', 'error_ratio_mae']] 
Final_Results.sort_values('R2', ascending = False)
```

#### Airbnb Price prediction
```python
run_classification(load_digits())
```

#### Classification task
```python
run_classification(load_digits())
```

#### Regression task
```python
run_regression(load_diabetes())
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- REPOSITORY TREE STRUCTURE -->
## Repository tree structure

```
.
│   Airbnb_Price_Prediction_Project.pdf
│   global_results.ipynb
│   README.md
│   requirements.txt
│
├───.ipynb_checkpoints
│       Airbnb_Price_Prediction_Project-First_Experimentations_Amine_Zaamoun-checkpoint.ipynb
│       draft_adel - Copy-checkpoint.ipynb
│       draft_adel-checkpoint.ipynb
│       global_results-checkpoint.ipynb
│       Untitled-checkpoint.ipynb
│
├───catboost_info
│   │   catboost_training.json
│   │   learn_error.tsv
│   │   time_left.tsv
│   │
│   └───learn
│           events.out.tfevents
│
├───dataset
│       AB_NYC_2019.csv
│       airbnb-listings.csv
│       name_tsne.csv
│       text_tsne.csv
│
├───decision_tree_from_scratch
│   │   Decision_Tree.py
│   │   Test.py
│   │   Test_of_homemade_decision_tree.ipynb
│   │   tree_classification_digits_dataset.png
│   │   tree_classification_iris_dataset.png
│   │   tree_regression_Airbnb_dataset.png
│   │   tree_regression_california_housing.png
│   │   tree_regression_diabetes_dataset.png
│   │
│   ├───.ipynb_checkpoints
│   │       homemade_decision_tree-checkpoint.ipynb
│   │
│   └───__pycache__
│           Decision_Tree.cpython-39.pyc
│
├───drafts
│   │   Airbnb_Price_Prediction_Project-First_Experimentations_Amine_Zaamoun.ipynb
│   │   decision_tree_scratch_draft.ipynb
│   │   draft_adel.ipynb
│   │
│   └───.ipynb_checkpoints
│           draft_adel-checkpoint.ipynb
│
├───functions
│   │   functions.py
│   │
│   └───__pycache__
│           functions.cpython-39.pyc
│
├───img
│       .gitignore
│       Airbnb_NYC-prices.jfif
│       decision_tree_from_scratch-viz.jfif
│       New_York_City_.png
│       plot_Airbnb_Price_NYC.png
│
├───models
│   │   adaboost_tuning.ipynb
│   │   bagging_tuning.ipynb
│   │   catboost_tuning.ipynb
│   │   decision_tree_from_scratch.ipynb
│   │   decision_tree_tuning.ipynb
│   │   extremely_randomized_forest_tuning.ipynb
│   │   lgbm_tuning.ipynb
│   │   random_forest_tuning.ipynb
│   │   sk_gradient_boosting_tuning.ipynb
│   │   sk_hist_gradient_boosting_tuning.ipynb
│   │   stacking_tuning.ipynb
│   │   tree_regression_Airbnb_dataset.png
│   │   voting_tuning.ipynb
│   │   xgboost_tuning.ipynb
│   │
│   ├───.ipynb_checkpoints
│   │       adaboost_tuning-checkpoint.ipynb
│   │       bagging_tuning-checkpoint.ipynb
│   │       catboost_tuning  TO DO-checkpoint.ipynb
│   │       decision_tree_from_scratch-checkpoint.ipynb
│   │       decision_tree_from_scratch_tuning-checkpoint.ipynb
│   │       decision_tree_tuning-checkpoint.ipynb
│   │       draft_adel - Copy-checkpoint.ipynb
│   │       extremely_randomized_forest_tuning-checkpoint.ipynb
│   │       lgbm_tuning-checkpoint.ipynb
│   │       random_forest_tuning-checkpoint.ipynb
│   │       sk_gradient_boosting_tuning-checkpoint.ipynb
│   │       sk_hist_gradient_boosting_tuning-checkpoint.ipynb
│   │       stacking_tuning-checkpoint.ipynb
│   │       voting_tuning-checkpoint.ipynb
│   │       xgboost_tuning-checkpoint.ipynb
│   │
│   ├───saved_models
│   │       adaboost_params.json
│   │       bagging_params.json
│   │       catboost_params.json
│   │       decision_tree_params.json
│   │       extremely_randomized_forest_params.json
│   │       homemade_tree_params.json
│   │       lgbm_params.json
│   │       lgbm_tuned.txt
│   │       random_forest_params.json
│   │       sk_gradient_boosting_params.json
│   │       sk_hist_gradient_boosting_params.json
│   │       vote_params.json
│   │       xgb_model.json
│   │       xgb_params.json
│   │
│   └───saved_scores
│           homemade_decision_tree_score.json
│           homemade_stacking_scores.json
│           sk_stacking_scores.json
│
└───seq2vec_tsne
    │   nlp_tsne_embedding_of_texts.ipynb
    │
    └───.ipynb_checkpoints
            nlp_tsne_embedding of texts-checkpoint.ipynb
            nlp_tsne_embedding_of_texts-checkpoint.ipynb
```

The ```global_results.ipynb``` notebook summarizes all the results obtained, and allows to re-fit all the saved models.

The experimented models are tuned and saved in separate notebooks in the folder ```models```.

All the notebooks rely on helper functions stored in the folder ```functions```.

The decision tree algorithm coded from scratch, the tsne embeddings performed on textual data and some draft notebooks have been stored in separate folders.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Adel Remadi](https://github.com/adel-R)
* [Amine Zaamoun](https://github.com/Zaamine)
* [Victor Maillot](https://github.com/v-maillot)
* [Vanille Bourre](https://github.com/VanilleB16)
* [Mélodie Mirval](https://github.com/melomvl)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/badge/Contributors-5-brightgreen?style=plastic
[contributors_url]: https://github.com/adel-R/Ensemble2023/graphs/contributors
[license-shield]: https://img.shields.io/badge/License-MIT-brightgreen?style=plastic
[license_url]: https://github.com/adel-R/Ensemble2023/blob/main/LICENSE
[Pandas_url]: https://pandas.pydata.org/docs/
[Numpy_url]: https://numpy.org/doc/
[Scikit-learn_url]: https://scikit-learn.org/stable/
[Spacy_url]: https://spacy.io/
[Matplotlib_url]: https://matplotlib.org/
[Seaborn_url]: https://seaborn.pydata.org/index.html
[Graphviz_url]: https://graphviz.org/
[Pandas]: https://pandas.pydata.org/docs/_static/pandas.svg
[Numpy]: https://numpy.org/doc/_static/numpylogo.svg
[Scikit-learn]: https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png
[Spacy]: https://raw.githubusercontent.com/github/explore/8cf1837393d83900e767cc895dcc814d053e2ffe/topics/spacy/spacy.png
[Matplotlib]: https://matplotlib.org/
[Seaborn]: https://seaborn.pydata.org/_static/logo-wide-lightbg.svg
[Graphviz]: https://upload.wikimedia.org/wikipedia/en/4/48/GraphvizLogo.png
