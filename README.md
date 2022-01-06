# Spam detection with a Support Vector Machine

E-mail is one of the most secure medium for online communication and transferring data through the web. With its increase in popularity, the number of ill-intentioned/unsolicited emails has also increased rapidly. To filtering data, there are different approaches which automatically classify these messages. The one I picked for this project was a support vector classifier, a method I first experimented with while on [this Skills Bootcamp](https://instituteofcoding.org/skillsbootcamps/course/skills-bootcamp-in-artificial-intelligence/).
An SVM works well in a binary classification task such as this one. It separates spam from non-spam (aka ham) by drawing a separator known as a hyperplane between the classes.

### Learning goals

- Develop a deep understanding of SVM concepts and implement them along with
- Data exploration and preprocessing
- Feature extraction using a dimensionality reduction method
- SVM with custom non linear kernel (SOON).

Like other projects on my GitHub, SVM-spam-detection comes with an accompanying website article detailing some of the decisions I made in my quest to classify spam and non-spam emails: [At this link](https://daniels-kb.github.io/svm-spam-detection)

### Still to-do:

1. Add docstrings, more documentation 
2. Continue exploring solution space with Jupyter Notebook, with the goal of improving accuracy. Hyperparameter tuning, dataset preprocessing and feature extraction methods were all found to have sizeable effects on the accuracy of a model, and thus the accuracy of a hypothetical product implementing this method. While there are other methods like Naive Bayes and Random Forests that may perform better, I will continue focusing on SVMs and kernel functions in this project, as well as the functions in the pipeline leading up to this classifier.  
3. This project largely made use of high level functions implemented by scikit-learn. My next step is to investigate the viability of creating a custom kernel function, tailored to this problem and dataset.
