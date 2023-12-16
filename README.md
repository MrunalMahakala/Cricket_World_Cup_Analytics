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

    -> Among the teams that participated were New Zealand, Pakistan, Bangladesh, South Africa,
        India, England, Afghanistan, Australia, Netherlands, Sri Lanka
    -> Match fixtures in which India and Australia 
    won nine matches in total world cup.
    -> We came to know about the venues of matches played
##**->From batting dataset:**

    ->Virat Kohli and Quinton de Kock are the top runscorers with 765
        and 705 runs total in World Cup
        and Virat Kohli also top the runs average at 69.55 with 
    -> number of fours hit by batsmen which was 35. 
    As for the highest number of sixes, Rohit Sharma hit the most sixes in the tournament. 
    -> We could see all the records on the batting were given to Indian batters.
##**->From bowling dataset:**

     -> We could see Shami is the highest wicket-taker in the World Cup
     and next goes to Adam Zampa.
     -> Shami was also the highest wicket-taker in a single match
     which was seven wickets.
     ->As for the bowling average and maiden overs in 
     all matches bumrah has 
     a best bowling average of 3.88 runs per over 
     and have 8 maidens recorded in the world cup
    
**Modeling**

From this modeling I wanted to know the rules behind win of worldcup team in a match for that I used decision trees in order to know the rules for each team win a particular match.
the below picture represents the model of rules to win for team New Zealand which have an accuracy of slightly more than 90%.

![image](https://github.com/MrunalMahakala/Cricket_World_Cup_Analytics/assets/50626560/503ae2fd-2589-4bac-8c18-2ed201bcb997)

    
**challenges:**

The dataset only contains the 2023 World Cup details.
and the dataset has 3 individual table dataframes so it has difficulty merging datasets effectively without redundancy and usefully.
the dataset doesn't contain any information on players because the dataset was created just recently so may update the dataset.

**Future Considerations:**
In the future, I would like to explore more about the dataset and find an effective way to combine the data frame tables in dataset and develop more complex machine learning models

###Running the code


    1)First install the necessary packages.
    2)Download ipynb file.
    3)(optional) in the model snippet can change the target team to any team 
    for the model to know the accuracy of the model to the team.
