# Cricket_World_Cup_Analytics

### This project aims to analyze and derive insights from the World Cup dataset, focusing on specific aspects, e.g., batting performance, bowling statistics, etc.

**Dataset information:**
The dataset I have chosen is taken from the Kaggle (https://www.kaggle.com/datasets/enggbilalalikhan/odi-world-cup-2023-complete-dataset/data ).
which is the ICC Cricket World Cup 2023 dataset with detailed batting and bowling stats, match schedules, and results.

##**Technologies used:**
Python, Jupyter Notebooks
Pandas, NumPy, Matplotlib, Seaborn.

##**Things found from EDA**

##**->from the Match dataset**  

    ->from the teams participated New Zealand,Pakistan,Bangladesh,South Africa,India,England,Afghanistan,Australia,Netherlands,Sri Lanka
    ->match fixtures in which india and australia won nine matches in total world cup.
    ->we came to know about venues of match played
##**->From batting dataset:**

    ->Virat kohli and quinton dekock are the top runscores with 765 and 705 runs total in world cup and virat kohli also top the runs average 69.55 with 
    -> number of fours hit by batsmen which was 35 as for the highest number of sixes rohit sharma hit mmost sixes in the tournament. 
    ->we could see all the records on the batting was given to indian batters.
##**->From bowling dataset:**

     ->we could see shami is the highest wicket taker in the world cup next goes to adam zampa.
     ->shami was also highest wicket taker in single match which is seven wickets.
     ->As for the bowling average and maiden overs in all matches bumrah has best bowling average of 3.88 runs per over and have 8 maiden recorded in world cup
    
**Modeling**

From this modeling i wanted to know the rules behind win of worldcup team in a match for that i used decision trees inorder to know the rules for each tewam win a particular match.
the below picture represents the model of rules to win for team new zealand which have accuracy of slightly more than 90%.

![image](https://github.com/MrunalMahakala/Cricket_World_Cup_Analytics/assets/50626560/503ae2fd-2589-4bac-8c18-2ed201bcb997)

    
**challenges:**

The dataset only contains the 2023 World Cup details.
and the dataset has 3 individual table dataframes so have difficulty in merging datasets effectively without redundency and usefully.
the dataset doesn't contain any information on players beacause the dataset was created just recently so may update the dataset.

**Future Considerations:**
In the future, I would like to explore more about the dataset and find an effective way to combine the data frame tables in dataset and develop more complex machine learning models

###Running the code


    1)First install necessary packages.
    2)Download ipynb file.
    3)(optional) in the model snippet can change the target team to any team for  the model to know the accuracy of model to the team.
