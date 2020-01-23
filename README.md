# Guassian NB C++ Implementation   

This implementation is using the ***Iris Dataset***

**Guassian Naive Bayes Classifier** is a type of NB that only deals with features that have continuous values.  

***If you are using a different dataset, you must understand your data and how each feature relate to the classes***. 

### How does it works? ###
The classifier calculates the mean value and standard deviation of each feature(X) in relation to class (k).        
It applies this equation to each given test input to calculate probability:     
    
![image](https://github.com/Lemon-cmd/GuassianNB/blob/master/Image%201-23-20%20at%202.49%20PM.jpg). 

For example, if the current input entry is [1.4, 2.4, 3.5, 5.0].   
For iteration, calculate the probability for each element of input.    

Basically, probability(C) = probability(INP(0)) * probability(INP(1)) ... * proability(INP(n));      
 
***After calculating probability of all classes based on input, a max method is used to determine the highest probability and select the best class.***  

### How to use? ###   
***You can simply run the bash script for building and running the code.***     

***   

### Other NB Implementations: ###    
**Multinominal Naive Bayes Classifier** :  [https://github.com/Lemon-cmd/NaiveBayesClassifier]  
**Boolean Multinominal NB** : ***Still in progress***
