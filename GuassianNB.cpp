#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <math.h>
#include <map>

using namespace std;

class GuassianNB 
{
    private:  
        vector <vector <string> > data; 
        vector <string > Y;
        vector <int> ignored_cols;

        vector <vector <string> > Features;
        
        void split_data()
        {
            //grab the Y column
            for (auto &row : data)
            {
                Y.push_back(row[row.size()-1]);
            }
          
            //grab X's columns
            for (auto &rowX : data)
            {
                vector <string> X;
                for (int col = 0; col < rowX.size()-1; col ++)
                {
                    X.push_back(rowX[col]);
                }
                Features.push_back(X);
            }
        }

        void insert(const vector<string> &item)
        {   
            data.push_back(item);
        }

        void load(string &filename)
        {
            // load file onto the vector data
            ifstream file(filename);
            string line;

            while (getline(file, line))
            {
                vector<string> item;
                istringstream input(line);
                //split the line by space
                for (string temp; input >> temp;)
                {
                    //remove comma from each string
                    for (int i = 0; i < temp.size(); i ++)
                    {
                        if (temp[i] == ',') temp[i] = ' ';
                    }
                    //append each string onto the current vector
                    item.push_back(temp);
                }
                //insert the row onto the data
                insert(item);
            }
        } 

    public: 
        GuassianNB(string filename, vector<int> &ignores)
        {   
            ignored_cols = ignores;
            load(filename); //load data
            split_data();
        }

        void displayData()
        {
            int col_size = data[0].size();
            int data_size = data.size();

            //display the data vector
            for (auto &entry: data)
            {
                for (auto &item : entry)
                {
                    cout << item << " ";
                }
                cout << "\n";
            }

            cout << "\n data size: " << data_size << " column size: " << col_size << endl;
        }

        void display()
        {
            for (auto &y : Y)
            {
                cout << y << endl;
            }

            for (auto &row : Features)
            {
                for (auto &col : row)
                {
                    cout << col << " ";
                }
                cout << "\n";
            }
        }

};

int main()
{
    vector<int> ignores;
    ignores.push_back(0);
    
    GuassianNB classifier = GuassianNB("iris.data", ignores);
    classifier.display();

}