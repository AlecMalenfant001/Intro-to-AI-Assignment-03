# Alec Malenfant amalenf@pnw.edu
# Assignment 3
# Problem 1 part i
""" Using the descision tree algorithm, this program will split the Iris data set into : 
80% training data & 20% test data and 70% trainging data & 30% test data.
Then it will compare the accuracy of each data split with a graph. 
"""
import matplotlib.pyplot as plt
import pandas as pd
import traceback
import csv
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier



class DecisionTreeSolution:

    def load_iris(self):
        ## This function changes the working directory to the directory of the current script,
        ## converts the 'iris.data' file to 'iris.csv' with appropriate column titles, and then
        ## loads the 'iris.csv' file into a pandas DataFrame. It handles file not found errors
        ## and other exceptions by printing the traceback.
        ##
        ## Parameters:
        ## None
        ##
        ## Returns:
        ## pandas.DataFrame: The loaded Iris dataset

        # Change working directory to the path of this python file
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        # convert .data to .csv
        try:
            with open('./iris.data', 'r') as in_file:
                with open('./iris.csv', 'w', newline='') as out_file:
                    writer = csv.writer(out_file)
                    writer.writerow(
                        ('septal length', 'septal width', 'petal length', 'petal width', 'species'))  # column titles
                    reader = csv.reader(in_file)
                    for row in reader:
                        writer.writerow(row)
        except FileNotFoundError:
            print("Error : There is no 'iris.data' file to convert to .csv")
            print(traceback.format_exc())
        except Exception as e:
            print(traceback.format_exc())

        # Load the Iris dataset
        try:
            iris = pd.read_csv('./iris.csv')
            return iris
        except FileNotFoundError:
            print("Error : 'iris.csv' not found")
            print(traceback.format_exc())
        except Exception as e:
            print(traceback.format_exc())

    def get_accuracy(self):
        # Return the accuracy of the model
        # 
        # Parameters:
        # None
        #
        # Returns:
        # float: The accuracy of the model
        return self.accuracy

    def __init__(self, DATA_SPLIT):
        ## This constructor initializes the class with a data split ratio, loads the Iris dataset,
        ## splits it into training and testing sets, trains a decision tree classifier, and calculates
        ## the accuracy of the model.
        ##
        ## Parameters:
        ## DATA_SPLIT (float): The proportion of the dataset be test data.

        self.data_split = DATA_SPLIT
        self.accuracy = 0

        # load iris data set into .csv
        iris = self.load_iris()

        # Split data set into X and y
        X = iris.iloc[:, :4]  # plant measurement data
        y = iris.iloc[:, 4]  # plant species data

        # Split data sets into training and testing. random_state controls the shuffling to get reproducible results
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=24601, test_size=self.data_split)

        # Train decision tree model
        dtc = DecisionTreeClassifier()
        dtc.fit(X_train, y_train)

        # Model predicts species for test data
        y_pred = dtc.predict(X_test)

        # Calculate Accuracy
        self.accuracy = dtc.score(X_test,y_test)


if __name__ == '__main__':
   # Create data split of 80% training data and 20% testing data
    DTS_8020 = DecisionTreeSolution(DATA_SPLIT=0.2)
    accuracy_8020 = DTS_8020.get_accuracy()
    
    # Create data split of 70% training data and 30% testing data
    DTS_7030 = DecisionTreeSolution(DATA_SPLIT=0.3)
    accuracy_7030 = DTS_7030.get_accuracy()
    
    # Create a bar graph comparing the accuracy of the two data splits
    splits = ['80/20', '70/30']
    accuracies = [accuracy_8020, accuracy_7030]
    plt.bar(splits, accuracies, color=['blue', 'red'])
    plt.xlabel('Data Split')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Decision Trees Trained on Different Data Splits')
    plt.ylim(0.5, 1)  # Accuracy ranges from 0 to 1
    
     # Add accuracy values on top of the bars
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom')

    plt.show()