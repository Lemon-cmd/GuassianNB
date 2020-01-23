#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <math.h>
#include <map>
#include <algorithm> //for find method
#include <stdlib.h>  //for converting string -> double

#define pi 3.14159265359 

using namespace std;

class GuassianNB 
{
    private:  

        struct metadata
        {
            /* A struct for collecting metadata per class */
            double mean = 0.0;
            double stdeviation = 0.0;
            double occurrences = 0.0;
            double range = 0.0;
            double probX = 0.0;
        };
    

        vector <vector <string> > train_data;   //vector or list that holds train data
        vector <vector <string> > test_data;    //vector that holds test data; in this case they're the same
        vector <int> ignoredXs;                 //vector that holds column index to be ignored

        map <string, map<int, metadata*> > Y;  // Y or outcomes

        int train_size;
        int test_size; 

        string predictY(vector <string> &inputs)
        {
            /* This method predict the best Y and return the result (best Y and best Y probability) */

            map<string, double> compareM;       // a map for collecting the result

            for (auto &y : Y)
            {
                double probability;  // variable for probability

                /* Calculate the probability of X(i) -> Class (k) */
                //ignore the last column in dataset b/c it's the class colum
                for (int col = 0; col < inputs.size() - 1; col ++) 
                {
                    double currentX = atof(inputs[col].c_str());    //convert currentX from str -> double

                    //this is the exponent part of the equation
                    double exponent = exp(-(pow(currentX - y.second[col]->mean, 2) / (2 * pow(y.second[col]->stdeviation, 2))));

                    //the full equation
                    double equation = 1 / (sqrt(2 * pi * pow(y.second[col]->stdeviation, 2))) * exponent;

                    //probability = equation(X(i)) * .... equation(X(n))
                    probability *= (equation) * y.second[col]->probX;
                }
                //set the map
                compareM[y.first] = probability;
            } 

            //find the best Y with best probability
            string bestY;
            double bestYP = 0.0;

            //a simple max function
            for (auto &y : compareM)
            {
                if (bestYP <= y.second)
                {
                    bestY = y.first; 
                    bestYP = y.second;
                }
            }
            return bestY;
        }

        void setY()
        {
            /* A method for simply setting the set of Y values into the map Y; it initalizes the map struct to its key */
            for (auto &row : train_data) 
            {
                if (Y.find(row[row.size()-1]) == Y.end()) // if the key does not exist
                {
                    map <int, metadata*> entry;
                    Y[row[row.size()-1]] = entry;
                }
            }
        }

        void rangeX()
        {
            //A Method for finding Min and Max value of each feature in relation to Class
            for (auto &y : Y)
            {
                for (int col = 0; col < train_data[0].size() - 1; col ++ )
                {
                    double min = double(INT_MAX);
                    double max = 0.0;

                    for (auto &row : train_data)
                    {
                        if (row[row.size()-1] == y.first)
                        {
                            double currentX = atof(row[col].c_str());
                            if (currentX <= min)
                            {
                                min = currentX;
                            }

                            if (currentX >= max)
                            {
                                max = currentX;
                            }
                        }
                    }
                    //subtract max - min to get range
                    y.second[col]->range = (max - min);
                }
            }
        }

        void meanX()
        {
            /* A method for calculating the mean of each feature onto each class; 
               Set metadata from {Y : {Feature Col -> mean, std, etc...}}
            */

            for (auto &y : Y)
            {
                for (int col = 0; col < train_data[0].size() - 1; col ++)
                {
                    for (auto &row : train_data)
                    {
                        if (row[row.size()-1] == y.first)
                        {
                            // making sure that column # does not exists in the item map
                            if (y.second.find(col) == y.second.end())
                            {
                                metadata* item = new metadata();
                                item->occurrences += 1;
                                item->mean += atof(row[col].c_str());
                                y.second[col] = item;
                            }
                            else
                            {
                                //increment the summation aka the mean variable at this moment
                                y.second[col]->mean += atof(row[col].c_str());
                                //count the number of occurences
                                y.second[col]->occurrences += 1;
                            }
                        }
                    }
                    //get the mean by (summation of all X values in the current column 
                    //over the # of its occurences corresponding to the current class)
                    y.second[col]->mean = y.second[col]->mean / y.second[col]->occurrences;
                    y.second[col]->probX = y.second[col]->occurrences / train_size;
                }
            }
            
            //find range of each X in this method
            rangeX();
        }

        void stdX()
        {
            /* A method for finding the standard deviation of values in each Feature corresponding to the current class */
            for (auto &y : Y)
            {
                for (int col = 0; col < train_data[0].size() - 1; col ++)
                {
                    double summation = 0.0;

                    for (auto &row : train_data)
                    {
                        //if the current row contains the current Y; perform the calculation
                        if (row[row.size()-1] == y.first)
                        {
                            // summation of (x - mean(x)) to the power of 2
                            summation += pow(atof(row[col].c_str()) - y.second[col]->mean, 2);
                        }
                    }
                    // take the sqrt root of summation / N of X
                    y.second[col]->stdeviation = sqrt(summation / (y.second[col]->occurrences));
                }
            }
        }

        void clean(vector <vector<string> > &target)
        {
            //Method for cleaning unusuable features
            vector <vector <string> > new_data;
            
            //loop through the target vector
            for (auto &row : target)
            {
                vector <string> entry;          //vector for new row
                bool unusuable = false;         //bool value for checking row with no data

                for (int col = 0; col < row.size(); col ++)
                {
                    //if there are targeted columns to be removed
                    if (ignoredXs.size() != 0)
                    {
                        if (find(ignoredXs.begin(), ignoredXs.end(), col) == ignoredXs.end())
                        {
                            entry.push_back(row[col]);
                        }   
                    }
                    //simply push all values if there are no targeted columns
                    else 
                    {
                        entry.push_back(row[col]);
                    }
                    
                    //a data in a column is blank;
                    //this will be change to replacing " " or ? with a NULL instead; removing a whole entry is bad for training
                    if (row[col] == "?" || row[col] == " ")
                    {
                        unusuable = true;
                    }
                }
                //only push if entry contains no blank data
                if (unusuable == false)
                    new_data.push_back(entry);
                    
            }

            // set the old data vector to new data vector
            target = new_data;
        }

        void load(const string filename, vector <vector <string> > &target)
        {
      
            // load file onto the target vector
            ifstream file(filename);    // grab file
            string line;                // a string for holding  current line

            // loop through the file
            while (getline(file, line))
            {
                //if the current line is not empty
                if (!line.empty())
                {
                    //split by comma, or period
                    for (int c = 0; c < line.size(); c ++)
                    {
                        if (line[c] == ',') line[c] = ' ';
                    }
                    
                    vector<string> row;    
                    istringstream input(line);
                    
                    //split by space
                    for (string s; input >> s;)
                    {
                        row.push_back(s);  //append individual string onto the vector
                    }
                    //insert the row onto the data
                    target.push_back(row);
                }
            }
        }   
    public: 
        GuassianNB(vector<int> &ignores)
        {   
            //set a vector for columns to be ignored
            ignoredXs = ignores;
        }

        void loadTrainD(string traindata)
        {
            load(traindata, train_data);    // load train file
            clean(train_data);              // clean train data
            train_size = train_data.size(); // set size of vector so that I don't have to call the size() a lot
        }

        void loadTestD(string testdata)
        {
            load(testdata, test_data);      // load test file
            clean(test_data);               // clean test data
            test_size = test_data.size();
        }

        void displayY()
        {
            // Method for displaying the Y map
            for (auto &y: Y)
            {
                for (auto &y2 : y.second)
                {
                    cout << y.first << " " << y2.first << "  "  << y2.second->mean <<  " "  << y2.second->stdeviation << endl;
                }
            }
        }

        void displayTrain()
        {
            //display the data vector
            for (auto &entry: train_data)
            {
                for (auto &item : entry)
                {
                    cout << item << " ";
                }
                cout << "\n";
            }
        }
         void displayTest()
        {
            //display the data vector
            for (auto &entry: test_data)
            {
                for (auto &item : entry)
                {
                    cout << item << " ";
                }
                cout << "\n";
            }
        }

        void predict()
        {
            /* Predict Method */

            setY(); meanX(); stdX();    // call these three methods for setting unique classes, set mean of X's -> per class, set stdev of X -> per class

            double positives = 0.0;    // a variable for counting # of positives

            for (auto &row : test_data)
            {
                string output = predictY(row);
                if (output == row[row.size() - 1]) // if best Y == actual Y value
                {
                    positives += 1;
                }
            }

            double accuracy = (positives / double(test_size)) * 100; // accuracy

            cout << "Accuracy of the model: " << accuracy << "%" << endl;

        }
      
};

int main()
{
    vector<int> ignores;

    GuassianNB classifier = GuassianNB(ignores);
    classifier.loadTrainD("iris.data");
    classifier.loadTestD("iris.data");

    classifier.predict();

}