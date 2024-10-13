# Alec Malenfant
# CS 46200-001 002 Intro to AI
# Oct 13 2024
# Assignment 3

# Problem Description
1.	Using the iris data set of assignment 1 and 2 [https://archive.ics.uci.edu/dataset/53/iris,
using python convert .data files into .csv.] split it into
a.	80% train and 20% test data
b.	70% train and 30% test data
c.	Write in your own words to compare the accuracy of a and b using any graph of your choice on the following algorithms.
    i.	Decision Trees
    ii.	Random Forest

# Solution Description
To run a solution simply open the python file. The program will first open a blank command prompt.
After a few seconds a new window will apear with the accuracy graph of the two data splits.

The code in both problems is almost Identical. The main between the two programs is that the classifier class we
instanciate in the class constructor. One uses DecisionTreeClassifier from sklearn and another uses  
RandomForestClassifier from sklearn. 

## Comparing the Accuracies 
In general, the 80/20 split outperforms the 70/30 split in both the decision tree classifier and 
the random forest classifier. This makes sense because we expect the model with more training data to 
perform better. What else is interesting is how close the accuracy rates are between the two classifiers.
There seems to be no performance benefit from using a random tree classifier to a decision tree classifier
for this data set. one reason we are not seeing the potential benefits from a random forest classifier compared
to a decision tree classifier is because the data set is small and does not contain much noise. 

## Dependencies
The following packages must be installed to run the solution programs:
- matplotlib
- pandas
- sklearn
    - sklearn.model_selection
    - sklearn.tree
    - sklearn.ensemble

## Class Diagrams 
### Assignment03-Problem01i.py
+------------------------+
| DecisionTreeSolution   |
+------------------------+
| - data_split: float    |
| - accuracy: float      |
+------------------------+
| + __init__(DATA_SPLIT) |
| + load_iris()          |
| + get_accuracy()       |
+------------------------+

### Assignment03-Problem01ii.py
+------------------------+
| RandomForestSolution   |
+------------------------+
| - data_split: float    |
| - accuracy: float      |
+------------------------+
| + __init__(DATA_SPLIT) |
| + load_iris()          |
| + get_accuracy()       |
+------------------------+