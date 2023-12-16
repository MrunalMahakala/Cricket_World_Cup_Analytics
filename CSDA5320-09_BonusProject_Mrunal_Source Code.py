#!/usr/bin/env python
# coding: utf-8

# In[1]:


# libaraies used 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler



# In[2]:


# loading the dataset
batting_dataset=pd.read_csv('batting_summary.csv')
bowling_dataset=pd.read_csv('bowling_summary.csv')
Match_Schedule_dataset=pd.read_csv('match_schedule_results.csv')


# In[3]:


batting_dataset.head()


# In[4]:


bowling_dataset.head()


# In[5]:


Match_Schedule_dataset.head()


# In[6]:


#Match analysis
Match_Schedule_dataset.info()
Match_Schedule_dataset.isnull().sum()


# In[7]:


# changing Datatype into correct formats beacuse all of them are in object format

Match_Schedule_dataset['Venue'] = Match_Schedule_dataset['Venue'].astype('string')
Match_Schedule_dataset['Date'] = pd.to_datetime(Match_Schedule_dataset['Date'] + '-2023', format='%B %d-%Y')
Match_Schedule_dataset['Winner'] = Match_Schedule_dataset['Winner'].astype('string')
Match_Schedule_dataset['Team2'] = Match_Schedule_dataset['Team2'].astype('string')
Match_Schedule_dataset['Team1'] = Match_Schedule_dataset['Team1'].astype('string')


# In[8]:


#Matches happened in Venues

plt.figure(figsize=(20, 8))
sns.countplot(x='Venue', data=Match_Schedule_dataset)
plt.title('Matches Held At Each Venue',color='black',size=15)
plt.ylabel('Number of Matches',color='black',size=12)
plt.xlabel('Venues',color='black',size=6)
plt.show()


# In[9]:


# Teams Partcipated in world cup
Teams= Match_Schedule_dataset['Winner'].unique()
print("Teams Participated in Worldcup :")
for Team in Teams:
    print(Team)


# In[10]:


# Win count of teams participated in world cup

win_Count = Match_Schedule_dataset['Winner'].value_counts().index


plt.figure(figsize=(20, 8))
sns.countplot(x='Winner', data=Match_Schedule_dataset, order=win_Count)
plt.title('Matches Won By Team',color='black',size=15)
plt.ylabel('Number of Winned Matches',color='black',size=12)
plt.xlabel('Team',color='black',size=6)
plt.show()


# In[11]:


#Batting Analytics

top_scorer_in_Worldcup = batting_dataset.groupby('Batsman_Name')['Runs'].sum().reset_index().sort_values(by='Runs', ascending=False)
plt.figure(figsize=(18,4))

#top scorer
plt.bar(top_scorer_in_Worldcup['Batsman_Name'].head(10), top_scorer_in_Worldcup['Runs'].head(10))
plt.title('Run-Scorer Batsman by Top 10 ',size=12,color='darkblue')
plt.xlabel('Name of Batsman ',size=12,color='black')
plt.ylabel('Total Runs',size=12,color='black')
plt.xticks(rotation=360,color='darkblue')
for index, value in enumerate(top_scorer_in_Worldcup['Runs'].head(10)):
    plt.text(index, value + 0.5, str(value), ha='center', va='bottom', fontsize=10)
plt.show()


# In[12]:


plt.figure(figsize=(20, 4))
sixs=batting_dataset[batting_dataset['6s'] > 0]
total_6s = sixs.groupby('Batsman_Name')['6s'].sum().sort_values(ascending=False)
sns.barplot(x=total_6s.head(10).index, y=total_6s.head(10).values, color='lightblue')
plt.title("Players who hit Most 6s in World Cup", size=15, color='black')
plt.xlabel("Players", size=10, color='black')
plt.ylabel("Number of 6s", size=10, color='black')



for index, value in enumerate(total_6s.head(10).values):
    plt.text(index, value + 0.5, str(value), ha='center', va='bottom', fontsize=10)

plt.show()


# In[13]:


plt.figure(figsize=(20, 4))
fours=batting_dataset[batting_dataset['4s'] > 0]
total_4s = fours.groupby('Batsman_Name')['4s'].sum().sort_values(ascending=False)
sns.barplot(x=total_4s.head(10).index, y=total_4s.head(10).values, color='lightblue')
plt.title("Players who hit Most 4s in World Cup", size=15, color='black')
plt.xlabel("Players", size=10, color='black')
plt.ylabel("Number of 4s", size=10, color='black')



for index, value in enumerate(total_6s.head(10).values):
    plt.text(index, value + 0.5, str(value), ha='center', va='bottom', fontsize=10)

plt.show()


# In[16]:


# players with highest avg score


# Counting the number of innings for each batsman
count_innings = batting_dataset['Batsman_Name'].value_counts().reset_index()
Batsmens_score = batting_dataset.groupby('Batsman_Name')['Runs'].sum().reset_index()
count_innings.columns = ['Batsman_Name', 'Innings']
# merging battsmen score and innings
Avg_score = pd.merge(count_innings, Batsmens_score, on='Batsman_Name', how='inner')
Avg_score


# In[22]:


# batting average by name of the batsman
Avg_score['Batting_Avg'] = Avg_score['Runs'] / Avg_score['Innings']
#Calculate average batting average by name of the batsman
average_batting_average = Avg_score.groupby('Batsman_Name')['Batting_Avg'].mean().reset_index()

# Filter 'Innings' is zero to avoid division by zero
batting_average_data = Avg_score[Avg_score['Innings'] > 0]
 

# Sort by 'Batting_Average'
average_batting_average = average_batting_average.sort_values(by='Batting_Avg', ascending=False)

plt.figure(figsize=(18, 6))
sns.barplot(x='Batsman_Name', y='Batting_Avg', data=average_batting_average.head(10), color='darkblue')
plt.title('Highest Batting Average(Top 10 Batsmen)', size=17, color='black')
plt.xlabel('Batsmans Name', size=10, color='black')
plt.ylabel('Batting Average', size=11, color='black')
plt.xticks(rotation=90, color='darkblue')

for index, value in enumerate(average_batting_average['Batting_Avg'].head(10)):
    plt.text(index, value + 0.01, f'{value:.2f}', ha='center', va='bottom', fontsize=10)

plt.show()


# In[23]:


#bowling Analytics Section

#1.Top wicket taking bowlers

best_bowling_by_wickets = bowling_dataset.groupby('Bowler_Name')['Wickets'].sum().reset_index()#to
######################plot of wicket taking bowlers#####
plt.figure(figsize=(18, 4))
best_bowling_by_wickets = best_bowling_by_wickets.sort_values(by='Wickets', ascending=False)
sns.barplot(x='Bowler_Name', y='Wickets', data=best_bowling_by_wickets.head(10), color='lightblue')

plt.title('Top wicket taking bowlers by every match they played', size=15, color='black')
plt.xlabel('Name of Bowler', size=10, color='black')
plt.ylabel('Wickets Took by bowler', size=10, color='black')
plt.xticks(rotation=90, color='green')

for index, value in enumerate(best_bowling_by_wickets['Wickets'].head(10)):
    plt.text(index, value + 0.5, str(value), ha='center', va='bottom', fontsize=10)

plt.show()


# In[24]:


#2.Top bowlers based on the number of wickets taken on single match
plt.figure(figsize=(20, 4))
bowling_dataset = bowling_dataset.sort_values(by='Wickets', ascending=False)
sns.barplot(x='Bowler_Name', y='Wickets', data=bowling_dataset.head(10), color='lightblue', edgecolor='black')
plt.title('Top bowlers based on the number of wickets taken on match', size=15, color='black')
plt.xlabel('Bowler_Name', size=10, color='black')
plt.ylabel('Wickets', size=10, color='black')
plt.xticks(rotation=90, color='black')

for index, value in enumerate(bowling_dataset['Wickets'].head(8)):
    plt.text(index, value + 0.5, str(value), ha='center', va='bottom', fontsize=10)

plt.show()


# In[25]:


bowling_dataset


# In[54]:


#3 bowlers with more maidens 

Maidens_overs = bowling_dataset.groupby('Bowler_Name')['Maidens'].sum().reset_index()
####maiden_overs#######
# Sorting the DataFrame by 'Maidens'
plt.figure(figsize=(20, 4))
Maidens_overs = Maidens_overs.sort_values(by='Maidens', ascending=False)


sns.barplot(x='Bowler_Name', y='Maidens', data=Maidens_overs.head(10), color='lightblue')
plt.title('Bowlers Max Maidens Bowled', size=14, color='black')
plt.xlabel('BowlerName', size=12, color='black')
plt.ylabel('Maidens from bowler', size=12, color='lightblue')

plt.show()



# In[27]:


#4.finding the best average economy for bowler

min_overs_threshold = 20
filtered_data = bowling_dataset.groupby('Bowler_Name').filter(lambda x: x['Overs'].sum() >= min_overs_threshold)



plt.figure(figsize=(16, 4))
Bowler_Economy = filtered_data.groupby('Bowler_Name')['Economy'].mean().reset_index()
Bowler_Economy = Bowler_Economy.sort_values(by='Economy', ascending=True)
sns.barplot(x='Bowler_Name', y='Economy', data=Bowler_Economy.head(10), color='lightgreen')
plt.title('average economy rates of bowlers', size=15, color='black')
plt.xlabel('Bowler_Name', size=10, color='black')
plt.ylabel('Economy', size=10, color='black')


for index, value in enumerate(Bowler_Economy['Economy'].head(10)):
    plt.text(index, value + 0.01, f'{value:.2f}', ha='center', va='bottom', fontsize=10)

plt.show()


# In[28]:


#machine Learning algorithm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix



# In[ ]:





# In[29]:


merged_data = pd.merge(pd.merge(batting_dataset, bowling_dataset, on=['Match_no','Match_Between']), Match_Schedule_dataset, on='Match_no')
merged_data=merged_data.drop(['Scorecard URL'],axis=1)
merged_data


# In[30]:


# dummies=merged_data(pd.get_dummies(['Winner']))


# winner_dummies = pd.get_dummies(merged_data['Winner'])

# # Concatenate the dummy variables with the original DataFrame
# merged_data = pd.concat([merged_data, winner_dummies], axis=1)
merged_data

# # Drop the original 'Winner' column if needed
# merged_data = merged_data.drop('Winner', axis=1)


# In[32]:


full_data=merged_data
merged_data.corr()


# In[69]:


####################Model Section######################
#columns in the merged dataset
trget_NZ = (full_data['Winner']=='New Zealand')
target_ind=(full_data['Winner']=='India')
target_aus=(full_data['Winner']=='Australia')
target_SA=(full_data['Winner']=='South Africa')

selected_f=full_data[['Runs_x','Maidens','Runs_y','Wickets','Economy','Balls','4s','6s']]



train_X_nz, test_X_nz, train_y_nz, test_y_nz = train_test_split(selected_f, target_SA, test_size=0.30,random_state=18)

# the decision tree model
decisions_model_nz = DecisionTreeClassifier(max_depth=4)
decisions_model_nz.fit(train_X_nz, train_y_nz)

# Make predictions on the test data
predict_y_nz = decisions_model.predict(test_X_nz)

# Evaluate the model accuracy
model_accuracy_nz = accuracy_score(test_y_nz, predict_y_nz)
print("Model Accuracy:", model_accuracy_nz)

confuse_matrix_nz = confusion_matrix(test_y_nz, predict_y_nz)
confuse_matrix_nz




# In[70]:


report_nz = classification_report(test_y_nz, predict_y_nz)
print(f'Classification Report:\n{report_nz}')


# In[71]:


plot_tree(decisions_model, feature_names=['Runs_x','Maidens','Runs_y','Wickets','Economy','Balls','4s','6s'], class_names=[str(c) for c in decisions_model.classes_], filled=True, rounded=True)
plt.show()


# In[43]:


# Merge with match data based on 'Match_no'x
merged_data_bowling = pd.merge(bowling_dataset, Match_Schedule_dataset,on=['Match_no'])

merged_data_bowling


# In[ ]:





# In[ ]:





# In[ ]:




