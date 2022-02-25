# Spam detection with a Support Vector Machine WIP

E-mail is one of the most secure medium for online communication and transferring data through the web. With its increase in popularity, the number of ill-intentioned/unsolicited emails has also increased rapidly. To filtering data, there are different approaches which automatically classify these messages. The one I picked for this project was a support vector classifier, a method I first experimented with while on [this Skills Bootcamp](https://instituteofcoding.org/skillsbootcamps/course/skills-bootcamp-in-artificial-intelligence/).
An SVM works well in a binary classification task such as this one. It separates spam from non-spam (aka ham) by drawing a separator known as a hyperplane between the classes.

#To run 

Navigate to the root folder of this project and just run tox. It puts setup, running and testing in one place. 
Writing or have written unit tests for: Input validation, inter-phase reproducibility, individual preprocessors and their correct function collectively within the pipeline (integration test).
There is also an experimental branch where the benefit of dimensionality reduction methods is explored. That is still in research phase.

### Learning goals

- Develop a deep understanding of SVM concepts and implement them along with
- Data exploration and preprocessing
- Feature extraction using a dimensionality reduction method (SEE experimental-... branch)
- SVM with custom non linear kernel (SOON).

Like other projects on my GitHub, SVM-spam-detection comes with an accompanying website article detailing some of the decisions I made in my quest to classify spam and non-spam emails: [At this link](https://daniels-kb.github.io/svm-spam-detection)

### Still to-do:

1. Add docstrings, more documentation 
2. Continue exploring improving accuracy of model with dimensionality reduction on new branch, with Jupyter Notebook. This should make the model more performant as feature engineering steps produce a high dimensional, sparse input
3. Reproducibility issue
4. Custom SVM kernel or metric?