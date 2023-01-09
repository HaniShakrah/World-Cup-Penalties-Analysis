import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#All figures below can be found in the 'World Cup Penalties Analysis' repository on Github (https://github.com/HaniShakrah/World-Cup-Penalties-Analysis)
#Dataset is taken from https://www.kaggle.com/datasets/jandimovski/world-cup-penalty-shootouts-2022. I cleaned the .csv to only include Qatar 2022 data, but will later include analysis on all tournaments since 1994. 
#This dataset only includes shots taken during penalty shootouts, and does not include penalties taken during the course of regulation or extra time. I added the ones missing for the 2022 tournament only.
#For full description of data and explanation of column meanings, refer to https://www.kaggle.com/datasets/pablollanderos33/world-cup-penalty-shootouts

#Read data (all penalties taken, whether during a penalty shootout or not, at the 2022 World Cup)
data = pd.read_csv('qatarall.csv', low_memory=False)

#Filter only relevant attributes
data = data[['Team','Zone','Foot','Keeper','OnTarget','Goal']]

#-----FIGURE 1 ----- Histogram to see zone counts (Qatar only)
#We see that zones 7 and 9 (bottom corners) are by far the most common... something to keep in mind. Sample sizes for other zones are too small to draw any sizable conclusions. 
sns.set_theme(style="darkgrid")
sns.histplot(data, x='Zone',binwidth=1, hue='Foot', multiple='stack', discrete='True')
bins = np.arange(data['Zone'].min(), data['Zone'].max()+1)
plt.xticks(bins)
plt.show()


#-----FIGURE 2 ----- Histogram to see where keepers dove
#Overall, keepers dove to the right just as much as they did to the left. Rarely did keepers remain in the center; early indication shooting in the middle may be a great strategy.
sns.set_theme(style="darkgrid")
sns.histplot(data, x='Keeper',binwidth=1, hue='Goal', multiple='stack', discrete='True')
plt.show()


#Function to return dataframe containing shots targeted at a specific zone only. 
def copy(df, x, y ):
    for i in range (x, y):
        zone = []
        df2 = df.copy()
        zone = df2.loc[df2['Zone'] == i]
        return(zone)

zone1 = copy(data, 1, 2)
zone2 = copy(data, 2, 3)
zone3 = copy(data, 3, 4)
zone4 = copy(data, 4, 5)
zone5 = copy(data, 5, 6)
zone6 = copy(data, 6, 7)
zone7 = copy(data, 7, 8)
zone8 = copy(data, 8, 9)
zone9 = copy(data, 9, 10)

#Function to calcualte success rate for shots at each zone
def percentmade(*zone):
    list = []
    for x in zone:
        goalcount = x['Goal'].sum()
        totalcount = x['Goal'].count()
        percent = goalcount/totalcount
        list.append(percent)
    return(list)


percentmadelist= percentmade(zone1, zone2, zone3, zone4, zone5, zone6, zone7, zone8, zone9)
df = pd.DataFrame()
df['Zone'] = [1,2,3,4,5,6,7,8,9]
df['% Made'] = percentmadelist

#-----FIGURE 3 ----- Visualization to see % of shots made by zone
#Shooting at a center zone (2,5,8) clearly had success, while the bottom corners, despite being most popular, had the lowest conversion rate.
sns.barplot(data=df,  x='Zone', y='% Made').set(title='% Made by Zone, 2022 World Cup Only')
plt.show()

#Numbers could be skewed due to limited number of shots from just Qatar, so we are going to insert all shootout data since 1994.

#read in all shootout info and combine with Qatar 2022 data
allshootouts = pd.read_csv('shootoutsalltime.csv', low_memory=False)
frames = [data, allshootouts]
all = pd.concat(frames)
all = all.drop(columns=['Game_id', 'Penalty_Number' ,'Elimination'])

#1994 and on
all = all.iloc[77:,:]

#Checks to make sure all values in dataset are what I expect them to be
all['zone_test'] = all['Zone'].isin([1,2,3,4,5,6,7,8,9])
all['foot_test'] = all['Foot'].isin(['L', 'R'])
all['keeper_test'] = all['Keeper'].isin(['L','R','C'])
all['target_test'] = all['OnTarget'].isin([0,1])
all['goal_test'] = all['Goal'].isin([0,1])

#print(all.loc[all['zone_test'] == False])
#print(all.loc[all['foot_test'] == False])
#print(all.loc[all['keeper_test'] == False])
#print(all.loc[all['target_test'] == False])
#print(all.loc[all['goal_test'] == False])

#One correction to be made on a few rows. Keeper dive was recorded as a lowecase 'l' and not an uppercase 'L'. Consistency is important for below visualizations
all.loc[all["Keeper"] == 'l', 'Keeper'] = 'L'
all = all.drop(columns = ['zone_test', 'foot_test', 'keeper_test', 'target_test', 'goal_test'])

#Split by zone
zone1_all = copy(all, 1, 2)
zone2_all = copy(all, 2, 3)
zone3_all = copy(all, 3, 4)
zone4_all = copy(all, 4, 5)
zone5_all = copy(all, 5, 6)
zone6_all = copy(all, 6, 7)
zone7_all = copy(all, 7, 8)
zone8_all = copy(all, 8, 9)
zone9_all = copy(all, 9, 10)

#Function to calcualte success rate for shots at each zone, this time for all shootouts and 2022 in-game penalties. 
def percentmade(*zone):
    list = []
    for x in zone:
        goalcount = x['Goal'].sum()
        totalcount = x['Goal'].count()
        percent = goalcount/totalcount
        list.append(percent)
    return(list)

percentmadelist_all = percentmade(zone1_all, zone2_all, zone3_all, zone4_all, zone5_all, zone6_all, zone7_all, zone8_all, zone9_all)
df2=pd.DataFrame()
df2['zone'] = [1,2,3,4,5,6,7,8,9]
df2['percent_made'] = percentmadelist_all

# -----FIGURE 4 ----- Visualization to see % of shots made by zone (all data)
#Inference from before gets more support, as the bottom corners have 2 of the worst % success rates. There is evidence also against the prior hypothesis that shooting down the middle is more successful, as zones 2, 5, and 8 do not show any statistical differences.
z = sns.barplot(data=df2,  x='zone', y='percent_made')
for index, row in df2.iterrows():
    z.text(row.name, row.percent_made, round(row.percent_made, 2), ha='center')
plt.show()


ground_all = all.copy()
ground_all = ground_all.loc[(ground_all['Zone'] == 7) | (ground_all['Zone'] == 8) | (ground_all['Zone'] == 9)]

air_all = all.copy()
air_all = air_all.loc[(air_all['Zone'] == 1) | (air_all['Zone'] == 2) | (air_all['Zone'] == 3) | (air_all['Zone'] == 4) | (air_all['Zone'] == 5) | (air_all['Zone'] == 6)] 

ground_percent_made = percentmade(ground_all)
air_percent_made = percentmade(air_all)

#According to previous world cups, there is a better chance of scoring a penalty when keeping the ball high (0.739), in comparison to keeping it on the ground(0.637). 
df3 = pd.DataFrame()
df3['ShotType'] = ['Ground', 'Air']
df3['Percent_Made'] = [ground_percent_made, air_percent_made]
print(df3)

#-----FIGURE 5 ----- Plot to see where keepers dive based on shooter's foot 
#For right foot shooters, it looks like there was a slight inclination for keepers to dive to the left. Pair that with the suggestion to avoid the lower zones, and it seems as if a higher shot to the shooters right(for right footers at least), is ideal. Let's test this.
sns.set_theme(style="darkgrid")
sns.displot(all, x='Keeper', col='Foot', hue='Goal', multiple = 'stack', discrete= True)
plt.show()

#Creating dataframes for different shots. Across body shot refers to a right footer shooting to their left, and vice versa for left footers. Open body shots refer to right footers shooting to their right, and vice versa. 
right_across = all.copy()
right_across = right_across.loc[(right_across['Foot'] == 'R') & ((right_across['Zone'] == 1) | (right_across['Zone']==4) | (right_across['Zone']==7))]

right_open = all.copy()
right_open = right_open.loc[(right_open['Foot'] == 'R') & ((right_open['Zone'] == 3) | (right_open['Zone']==6) | (right_open['Zone']==9))]

left_across = all.copy()
left_across = left_across.loc[(left_across['Foot'] == 'L') & ((left_across['Zone'] == 3) | (left_across['Zone']==6) | (left_across['Zone']==9))]

left_open = all.copy()
left_open = left_open.loc[(left_open['Foot'] == 'L') & ((left_open['Zone'] == 1) | (left_open['Zone']==4) | (left_open['Zone']==7))]

#Shots at center zones regardless of foot 
center = all.copy()
center = center.loc[((center['Zone'] == 2) | (center['Zone']==5) | (center['Zone']==8))]

#-----FIGURE 6 ----- Below creates each pie chart separately to combine into one figure using subplot() function
#The only notable finding is the high percentage of shots taken by right footers to right zones (open body shots). This further supports the above analysis. 

#Pie chart for across body shots (right footers)
make_RA=len(right_across[right_across['Goal'] == 1])
miss_RA=len(right_across[right_across['Goal'] == 0])
labels = 'Scored', 'Missed'
dataRA = [make_RA, miss_RA]
colors = sns.color_palette('pastel')
plt.subplot(2,3,1)
plt.pie(dataRA, labels=labels, colors=colors, autopct='%.0f%%')
plt.title('Right Footed Shooters - Across Body')

#Pie chart for open body shots (right footers)
make_RO=len(right_open[right_open['Goal'] == 1])
miss_RO=len(right_open[right_open['Goal'] == 0])
labels = 'Scored', 'Missed'
dataRO = [make_RO, miss_RO]
colors = sns.color_palette('pastel')
plt.subplot(2,3,2)
plt.pie(dataRO, labels=labels, colors=colors, autopct='%.0f%%')
plt.title('Right Footed Shooters - Open Body')

#Pie chart for across body shots (left footers)
make_LA=len(left_across[left_across['Goal'] == 1])
miss_LA=len(left_across[left_across['Goal'] == 0])
labels = 'Scored', 'Missed'
dataLA = [make_LA, miss_LA]
colors = sns.color_palette('pastel')
plt.subplot(2,3,3)
plt.pie(dataLA, labels=labels, colors=colors, autopct='%.0f%%')
plt.title('Left Footed Shooters - Across Body')

#Pie chart for open body shots (left footers)
make_LO=len(left_open[left_open['Goal'] == 1])
miss_LO=len(left_open[left_open['Goal'] == 0])
labels = 'Scored', 'Missed'
dataLO = [make_LO, miss_LO]
colors = sns.color_palette('pastel')
plt.subplot(2,3,4)
plt.pie(dataLO, labels=labels, colors=colors, autopct='%.0f%%')
plt.title('Left Footed Shooters - Open Body')

#Pie chart for all shots at center zones 
make_C=len(center[center['Goal'] == 1])
miss_C=len(center[center['Goal'] == 0])
labels = 'Scored', 'Missed'
dataC = [make_C, miss_C]
colors = sns.color_palette('pastel')
plt.subplot(2,3,5)
plt.pie(dataC, labels=labels, colors=colors, autopct='%.0f%%')
plt.title('All Shooters - Center')
plt.show()

#-----FIGURE 7 ----- Graph of success rates for different shot types
#Another graphic to see that the open body shot for right footers have a great success rate. It also refutes the case that shooting down the middle is successful. For the Qatar 2022 data, shooting down the middle worked, but further analysis proves otherwise.
df = pd.DataFrame()
df['Shot'] = ['Left_Open', 'Left_Across', 'Center', 'Right_Open', 'Right_Across']
df['% Made'] = [68.2, 68.0, 65.6, 73.6, 68.5]
plt.title('Comparison of Shot Types')
plt.plot(df['Shot'],df['% Made'], color='green', ls=':', marker = 'D')
plt.show()

#As we have seen, there is evidence that shooting in one of the bottom zones is less successful. Right footed shooters also tend to have more success shooting to their right. 
#Obviously, individual player recommendations should be made on a game to game basis taking in consideration factors like the opposing keeper and shooter tendencies, but this is a great general starting point to see trends of shooters and keepers on the world's biggest stage. 
