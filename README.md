**Install**

Using pip (recommended)
    
    pip install shapwaterfall
    
**Introduction**

Many times when VMware Data Science Teams present their Machine Learning models' propensity to buy scores (estimated probabilities) to stakeholders, stakeholders ask why a customer's propensity to buy is higher than the other customer. The stakeholder's question was our primary motivation. 

We were further concerned with recent algorithm transparency language in the EU's General Data Protection Regulation (GDPR) and the California Consumer Privacy Act (CCPA). Although the 'right to explanation' is not necessarily clear, our desire is to act in good faith by providing local explainability and interpretability between two references, observations, clients, and customers.

This graph solution provides a local classification model interpretability between two observations, which internally we call customers. It uses each customer's estimated probability and fills the gap between the two probabilities with SHAP values that are ordered from higher to lower importance. We prefer SHAP over others (for example, LIME) because of its concrete theory and ability to fairly distribute effects.

Currently, this package only works for tree and tree ensemble classification models. Our decision to limit the use to tree methods was based on two considerations. We desired to take advantage of the tree explainer's speed. As a business practice, we tend to deploy Random Forest, XGBoost, LightGBM, and  other tree ensembles more often than other classifications methods.

However, we plan to include the kernel explainer in future versions.

The package requires a tree classifier, training data, validation/test/scoring data with a column titled 'Reference', the two observations of interest, and the desired number of important features. The package produces a Waterfall Chart. 

**Command**

shapwaterfall(*clf, X_tng, X_val, ref1, ref2, num_features*)

**Required**

- *clf*: a tree based classifier that is fitted to X_tng, training data.
- *X_tng*: the training Data Frame used to fit the model.
- *X_val*: the validation, test, or scoring Data Frame under observation. Note that the data frame must contain an extra column who's label is 'Reference'.
- *ref1 and ref2*: the first and second reference, observation, client, or customer under study. Can either be a string or an integer. If the column data is a string, use 'ref1' and 'ref2'. Otherwise, use an integer, such as 4 or 107. 
- *num_features*: the number of important features that describe the local interpretability between to the two observations.

**Important Reminder**

The package users have to take care of the following with respect to the 'Reference' column. Otherwise it could result in errors.
 
- X_tng  should only have the features used while training using  model.fit() without the feature ‘Reference’.
- X_sc  should have the feature named  ‘Reference’  which is unique identifier for  the scoring data frame and this should not be a feature used in the model training.
- The features in X_sc  should be in the same order as X_tng. (This is important as I got totally wrong prediction scores when the order was not maintained)
- ‘ref1’ and ‘ref2’ are values of ‘Reference’ feature used for comparison

**Dependent Packages**

The shapwaterfall package requires the following python packages:

	import pandas as pd
	import numpy as np
	import shap
	import matplotlib.pyplot as plt
	import waterfall_chart

**Examples**

**Random Forest on WI Breast Cancer Data**

	# Scikit-Learn WI Breast Cancer Data Example
	# packages
	import pandas as pd
	import numpy as np
	from sklearn.datasets import load_breast_cancer
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.metrics import roc_auc_score
	from sklearn.model_selection import train_test_split
	import shap
	import matplotlib.pyplot as plt
	import waterfall_chart
	from shapwaterfall import shapwaterfall

	# models
	rf_clf = RandomForestClassifier(n_estimators=1666, max_features="auto", min_samples_split=2, min_samples_leaf=2,
                                max_depth=20, bootstrap=True, n_jobs=1)

	# load and organize Wisconsin Breast Cancer Data
	data = load_breast_cancer()
	label_names = data['target_names']
	labels = data['target']
	feature_names = data['feature_names']
	features = data['data']

	# data splits
	X_tng, X_val, y_tng, y_val = train_test_split(features, labels, test_size=0.33, random_state=42)

	print(X_tng.shape) # (381, 30)
	print(X_val.shape) # (188, 30)

	X_tng = pd.DataFrame(X_tng)
	X_tng.columns = feature_names
	X_val = pd.DataFrame(X_val)
	X_val.columns = feature_names

	# fit classifiers and measure AUC
	clf = rf_clf.fit(X_tng, y_tng)
	pred_rf = clf.predict_proba(X_val)
	score_rf = roc_auc_score(y_val,pred_rf[:,1])
	print(score_rf, 'Random Forest AUC')

	# 0.9951893425434809 Random Forest AUC

	# IMPORTANT: add a 'Reference' column to the val/test/score data
	X_val = pd.DataFrame(X_val)
	X_val['Reference'] = X_val.index
	print(X_val.shape) # (188, 31)

	# Use Case 1
	shapwaterfall(clf, X_tng, X_val, 5, 100, 5)
	shapwaterfall(clf, X_tng, X_val, 100, 5, 7)

	# Use Case 2
	shapwaterfall(clf, X_tng, X_val, 36, 94, 5)
	shapwaterfall(clf, X_tng, X_val, 94, 36, 7)
	
**Random Forest on UCI House Vote Data**

	# University of California, Irvine House Votes Data Example
	# packages
	import pandas as pd
	import numpy as np
	from sklearn.datasets import load_breast_cancer
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.metrics import roc_auc_score
	from sklearn.model_selection import train_test_split
	import shap
	import matplotlib.pyplot as plt
	import waterfall_chart
	from shapwaterfall import shapwaterfall

	# models
	rf_clf = RandomForestClassifier()

	# UCI Data 
	df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data')

	names = ['Republican', 'handicap infants', 'water project', 'budget resolution', 'physician fee freeze', 'el salvador aide', 'school religious groups', 'anti satellite', 'nicaraguan contras', 'mx missle', 'immigration', 'synfuels', 'education spending', 'superfund', 'crime', 'duty free exports', 'south africa']

	df.columns = names
	df = df.replace(to_replace =["republican", "y"], value = 1) 
	df = df.replace(to_replace =["democrat", "n", "?"], value = 0) 

	label = df.iloc[:,0]
	features = df.iloc[:,1:17]

	# data splits
	X_tng, X_val, y_tng, y_val = train_test_split(features, label, test_size=0.33, random_state=42)

	print(X_tng.shape)
	print(X_val.shape)

	# fit classifiers and measure AUC
	clf = rf_clf.fit(X_tng, y_tng)

	pred_rf = clf.predict_proba(X_val)
	score_rf = roc_auc_score(y_val,pred_rf[:,1])
	print(score_rf, 'Random Forest AUC')

	# 0.99238683127572 Random Forest AUC

	# IMPORTANT: add a 'Reference' column to the val/test/score data
	X_val = pd.DataFrame(X_val)
	X_val['Reference'] = X_val.index
	print(X_val.shape)

	# Use Case 3
	shapwaterfall(clf, X_tng, X_val, 78, 387, 5)
	shapwaterfall(clf, X_tng, X_val, 387, 78, 7)

	# Use Case 4
	shapwaterfall(clf, X_tng, X_val, 253, 157, 5)
	shapwaterfall(clf, X_tng, X_val, 157, 253, 7)

**Authors**

John Halstead, jhalstead@vmware.com

Rajesh Vikraman, rvikraman@vmware.com

Ravi Prasad K, rkondapalli@vmware.com

Kiran R, rki@vmware.com

**References**

1) Dua, D., Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data]. Irvine, CA: University of California, School of Information and Computer Science.

2) Iliev, K., Putatunda, S. (2019). “SHAP and LIME Model Interpretability”, VMware EDA AA & DS CoE PowerPoint Presentation, Palo Alto, CA, USA.

3) Dataman, D. (2019). “Explain Your Model with the SHAP Values”, Medium: Towards Data Science, available at https://towardsdatascience.com/explain-your-model-with-the-shap-values-bc36aac4de3d.

4) Gillies, S. (2020). “The Shapely User Manual”, Shapely 1.8dev documentation, available at https://shapely.readthedocs.io/en/latest/manual.html.

5) Nayak, A. (2019). “Idea Behind LIME and SHAP: the intuition behind ML interpretation models”, Medium: Towards Data Science, available at https://towardsdatascience.com/idea-behind-lime-and-shap-b603d35d34eb.

6) Molnar, C. (2020). “Interpretable Machine Learning: a Guide for Making Black Box Models Explainable”, E-book available at https://christophm.github.io/interpretable-ml-book/, updated July 20, 2020, Chapters 5.7 (Local Surrogate (LIME)) and 5.10. (SHAP (SHapley Additive exPlanations)).

7) Lundberg, S. (2018). “SHAP Explainers and Plots”, available at https://shap.readthedocs.io/en/latest/#.

8) Owen, S. (2019). “Detecting Data Bias Using SHAP and Machine Learning: What Machine Learning and SHAP Can Tell Us about the Relationship between Developer Salaries and the Gender Pay Gap”, Databricks, available at https://databricks.com/blog/2019/06/17/detecting-bias-with-shap.html.

9) Moffit, C. (2014). “Creating a Waterfall Chart in Python”, Practical Business Python, available at https://pbpython.com/waterfall-chart.html.

10) Sharma, A. (2018). “Decrypting your Machine Learning model using LIME: why should you trust your model?”, Medium: Towards Data Science, available at: https://towardsdatascience.com/decrypting-your-machine-learning-model-using-lime-5adc035109b5.

11) Ribeiro, MT. (2017). “LIME Documentation, Release 0.1”, available at https://buildmedia.readthedocs.org/media/pdf/lime-ml/latest/lime-ml.pdf.

12) Hulstaert, L. (2018). “Understanding model predictions with LIME”, Medium: Towards Data Science, available at https://towardsdatascience.com/understanding-model-predictions-with-lime-a582fdff3a3b.

13) Saabas, A. (2015). “treeinterpreter 0.2.2”, PyPl, available at https://pypi.org/project/treeinterpreter/.

14) Saabas, A. (2015). “Random forest interpretation with scikit-learn”, Diving into Data: A blog on machine learning, data mining and visualization, available at http://blog.datadive.net/random-forest-interpretation-with-scikit-learn/.

15) Singh, M., Kiran R, Harris, S. (2019). “Corona Impact: VMW Bookings and Propensity Models”, Vmware EDA AA & DS CoE PowerPoint Presentation, Palo Alto, CA, USA.

16) Lundberg, S., Lee, S. (2017). “A Unified Approach to Interpreting Model Predictions”, 31st Conference on Neural Information Processing Systems, Long Beach, CA, USA. 

17) Bowen, D., Ungar, L., (2020). “Generalized SHAP: Generating multiple types of explanations in machine learning”, Pre-print, June 15, 2020.

18) Veder, K. (2020). “An Overview of SHAP-based Feature Importance Measures and Their Applications To Classification”, Pre-print, May 8, 2020.

19) Lundberg, S., Erion, G., Lee, S. (2019). “Consistent Individualized Feature Attribution for Tree Ensembles”, Pre-print, March 7, 2019.