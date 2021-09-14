#!/usr/bin/env python
# coding: utf-8

# > **Tip**: Welcome to the Investigate a Dataset project! You will find tips in quoted sections like this to help organize your approach to your investigation. Before submitting your project, it will be a good idea to go back through your report and remove these sections to make the presentation of your work as tidy as possible. First things first, you might want to double-click this Markdown cell and change the title so that it reflects your dataset and investigation.
# 
# # Project: Investigate a Dataset (Medical Appointment No-Shows)
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# > Every medical practice has appointment cancellations and no-shows. As a healthcare marketer, you must be aware of the high cost of no-shows and cancellations: decreased revenue, underutilized staff, lost commissions and decreased staff morale. If not addressed properly, appointment cancellations and no-shows can add up to revenue losses, estimates pointing to an average of $  40,000  to $ 60,000 per year.
# 
# This is a dataset from the Brazilian Public Health System it consists of more than 100k appointments. 28.5% of the patients missed thier appoinment, the purpose of this investigation is to know why and what effect on this percentage.
# 
# **Data Dictionary**
# 1 - PatientId: Identification of a patient
# 
# 2 - AppointmentID:Identification of each appointment
# 
# 3 - Gender
# 
# 4 - Age
# 
# 5- Neighbourhood: Where the hospitals takes place.
# 
# 6 - Scholarship
# 
# 7 - Hipertension
# 
# 8 - Diabetes
# 
# 9- Alcoholism
# 
# 10- Handcap
# 
# 11- SMS_received
# 
# 12- No-show : "Yes" means there is a no - show , "No" means the patient showed
# 
# 
# **Questions to help in the investigation :**
# 
# 1-What is the Percentage of patients who attend the appointment and the patients who missed it?
# 
# 2- Does the gender effect on the ratio of the no-show , and if it effects which gender has more no-show?
# 
# 3- Is age an influencing factor?
# 
# 4- Which the illness diseases effect on the percentage of no-show?
# 
# 5- Does the location of the hospitals(neighbourhood) effect on the 28.5% percentage?
# 
# 6-  Does sending SMS to the patients help in reducing the no-show ratio?
# 
# 7- Does The appointment day effect on the 28.5% percentage?
# 
# 8- Does the gab between the scheduled day and the appointment day effect on the no show ratio?
# 
# 
# 

# <a id='investigation'></a>
# # Data Wrangling 
# 
# 
# ## 1- Understand the Data / Investigation 

# In[2]:


#set the environment 
import pandas as pd
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#upload the data 
df = pd.read_csv("noshow.csv")


# In[4]:


#check the dataset
df.head()


# In[5]:


#nderstand more about the data 
df.info()


# <a id='info'></a>
# **From the data info we can tell:**
# 
# The data consists of 110572 rows and 14 columns.
# 
# The SechduledDay and AppointmentDay need to be converted to datetime.
# 
# 
# 

# In[6]:


#Analyze the statical data for the dataset
df.describe()


# <a id='describe'></a>
# **The min age is -1 and the max age is 115 which is outliers , and it will be droped in the Data Cleaning section**
# 
# **We need to check the value count of the  3 deseases it seems like they are having just 0 and 1 values and for the Handcape 4 values**
# 

# In[7]:


#before complete investigation I will lowercase the columns to make it easier 
df.rename(columns=lambda x : x.strip().lower(), inplace=True)


# In[8]:


df.head(1)


# In[9]:


#hypertension counts unique values
df.hipertension.value_counts()


# In[10]:


#diabetes counts unique values
df.diabetes.value_counts()


# In[11]:


#alcoholism counts unique values
df.alcoholism.value_counts()


# In[12]:


#handcap counts unique values
df.handcap.value_counts()


# <a id='diseases'></a>
# **In general, patients with no diseases in each disease are more than others**

# In[13]:


df.gender.value_counts()


# <a id='diseases'></a>
# **In general, the number of females are higher than males , but the question is who has more no-show?**

# In[14]:


#Total number of unique observations over the index axis.
df.nunique()


# <a id='unique'></a>
# **Some patients has more than record**

# 
# <a id='cleaning'></a>
# ## 2- Data Cleaning 
#  
# 1- Check for missing rows 
# 
# 2- Check for duplicated data 
# 
# 3- Convert sechduleday and appointmentday to datetime 
# 
# 4- Drop age's outliers 
# 
# 5 - See the gab between the secheduleday and the appoinmentday to check if it effects the no-show ratio
# 
# 6- Add a column for weekday of the secheduleday to see later if it effects 
# 
# 7- Add two columns to divide the no-show column to attend patients and missing patients for further investigation
# 

# In[61]:


# 1- check missing rows
df.isnull().sum().all()


# In[16]:


# 2- Check for duplicated data
sum(df.duplicated())


# In[17]:


# 3- Convert sechduleday and appointmentday to datetime
df["scheduledday"] = df["scheduledday"].astype("datetime64")
df["appointmentday"] = df["appointmentday"].astype("datetime64")


# In[18]:


#check if they are converted
df.info()


# In[19]:


# 4 - query age's outliers 
age_outliers = df.query("age < 0 | age == 115")


# In[20]:


#drop age's outliers 
df.drop(index = age_outliers.index ,inplace=True)


# In[21]:


#check the age 
df["age"].describe()


# In[22]:


# 5- See the gab between the secheduleday and the appoinmentday and create a new column with it 
df['gab'] = (df['appointmentday']- df['scheduledday']).dt.days


# In[23]:


#check the gab column
df.head()


# In[24]:


#gab counts unique values
df.gab.value_counts()


# <a id='gab'></a>
# **It seems like there are some rows with gab less than 0 , which means that the appointment day was before the scheduled day , to avoid its effect in the visualization I will drop it**

# In[25]:


#filter the gabs less than 0 
gab_outliers = df[df["gab"]<0]


# In[26]:


#drop the gab_outliers
df.drop(index=gab_outliers.index , inplace=True)


# In[27]:


#check gab counts unique values
df.gab.value_counts()


# In[28]:


df.head()


# In[29]:


# 7- Add a column for weekday of the secheduleday to see later if it effects
df['weekday'] = df['appointmentday'].dt.day_name() 


# In[30]:


#Check the new column 
df.head(1)


# In[31]:


df.info()


# In[32]:


#weekday unique value counts
df.weekday.value_counts()


# In[33]:


#8- Add two columns to divide the no-show column to attend patients and missing patients for further investigation
attend = df['no-show'] == "No" 
missing = df['no-show'] == "Yes"

df['attend'] = attend 
df['missing'] = missing 


# In[34]:


#check the two new columns
df.head()


# <a id='cleaning'></a>
# **After cleaning lets check the new dataset**

# In[35]:


df.head()


# In[36]:


#dataset shape
df.shape


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# 
# 

# ### Research Question 1 : What is the Percentage of patients who attend the appointment and the patients who missed it?
# 

# In[37]:


#plot pie chart for the show and noshow count
df['no-show'].value_counts().plot(kind='pie',autopct='%1.1f%%' , fontsize=15,labels=['Show','No-Show'],explode=(.1 , 0) , colors = ["pink" , "purple"] , shadow = True);
plt.suptitle('Show and no-Show percentage ',fontsize=20)
plt.axis('off');


# **As shown 71.5% of patients did not miss the appointment , and 28.5% miss their appointment.**
# 
# **The below exploration explan why this percentage exist.**
# 
# 

# 
# ## Section 1: exploring the questions that with univariate variables
# 

# ### Research Question 2 : Does the gender effect on the ratio of the no-show , and if it effects which gender has more no-show?
# 

# In[38]:


#plot bar chart for the gender to compare between who attend and who missed /gender
plt.figure(figsize=(20,8))
plt.subplot(1,2,1)
ax = df["gender"][missing].value_counts(normalize=True).plot(kind='bar', color = "pink")
plt.title("The proportion of the patients who missed the appointment ratio to the gender")
plt.ylabel("Proportion")
plt.xlabel("Gender")
plt.legend( loc='upper right'  )

for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate('{:0.1f}%'.format(height*100), (x + width/2, y + height*1.02), ha='center')
    
plt.subplot(1,2,2)

ay = df["gender"][attend].value_counts(normalize=True).plot(kind='bar', color = "pink")
plt.title("The proportion of the patients who attend the appointment ratio to the gender")
plt.ylabel("Proportion")
plt.xlabel("Gender")
plt.legend( loc='upper right'  )

for i in ay.patches:
    width_y = i.get_width()
    height_y = i.get_height()
    x, y = i.get_xy()
    ay.annotate('{:0.1f}%'.format(height*100), (x + width_y/2, y + height_y*1.02), ha='center')


# In[39]:


#lets see the total percentage of the females to the males in the dataset
df['gender'].value_counts().plot(kind='pie',autopct='%1.1f%%' , fontsize=15,labels=['Female','Male'],explode=(.1 , 0) , colors = ["pink" , "purple"] , shadow = True);
plt.suptitle('Gender',fontsize=20)
plt.axis('off');


# <a id='gender'></a>
# **The gender does not effect on the no-show but we can tell that females are taking care of thier health more than males**

# ### Research Question 3 :Is age an influencing factor ?
# 
# 

# In[40]:


#plot histogram for age
df["age"][attend].hist(alpha = 0.5 , label = "age of attending patients " , color = "coral" ) 
df["age"][missing].hist(alpha=0.5  , label = "age of missing patients " , color = "brown" )
plt.ylabel("Count")
plt.xlabel("Age")

plt.legend();


# **The age effects on the no-show ratio, as the age increase the missing patients decrease , the range of approx. (50-60) has  most attend count.**
# 

# In[41]:


#plot the age by ranges

df["age"][attend].hist(alpha = 0.5 , label = "age of attending patients " , color = "coral" ) 
df["age"][missing].hist(alpha=0.5  , label = "age of missing patients " , color = "brown" )
plt.ylabel("Count")
plt.xlabel("Age")

plt.legend();


# ### Research Question 4 :Which the illness diseases effect on the percentage of no-show  ?

# In[42]:


#plot bar charts for (hipertension , diabetes , alcoholism ,handcap ) for both attend and missing
plt.figure(figsize = [20, 10]) 
plt.subplot(2, 2, 1)
sb.countplot(data = df, x = df["hipertension"], hue = df["no-show"] )
plt.subplot(2, 2, 2)
sb.countplot(data = df, x = df["diabetes"], hue = df["no-show"] )
plt.subplot(2, 2, 3)
sb.countplot(data = df, x = df["alcoholism"], hue = df["no-show"] )
plt.subplot(2, 2, 4)
sb.countplot(data = df, x = df["handcap"], hue = df["no-show"] )


# **Note:"No" means patients attend (no-show) and "Yes" means patients did not attend**

# **We can get a conclusion from the four graphs that number of patients without hipertension and diabetes are more likely to did'nt attend the appointment , but in general we can say that these four factors do not effect on the no-show percentage.**

# ### Research Question 5:   Does the location of the hospitals(neighbourhood) effect on the 28.5% percentage ?

# In[43]:


#plot the neighbourhood for the attend section
plt.figure(figsize=(18,4))
plt.title('Patients distribution by neighbourhood')
sb.countplot(x=df["neighbourhood"][attend], data=df);
plt.xticks(rotation='90')
plt.show()


# In[44]:


#plot the neighbourhood for the missing section
plt.figure(figsize=(18,4))
plt.title('Patients distribution by neighbourhood')
sb.countplot(x=df["neighbourhood"][missing], data=df);
plt.xticks(rotation='90')
plt.show()


# **Jardim Cumburi is the highest count for both show (attend) and noshow(missing) , but in general neighbourhood does not effect on the percentage**

# ### Research Question 6:   Does sending SMS to the patients help in reducing the no-show ratio ?

# In[45]:


df["sms_received"][attend].hist(alpha = 0.5 , label = "age of attending patients " , color = "lightgreen" ) 
df["sms_received"][missing].hist(alpha=0.5  , label = "age of missing patients " , color = "darkseagreen"  )
plt.ylabel("Count")
plt.xlabel("SMS-Received")
plt.legend( );


# **Sending SMS to the patients doesn't reduce the no-show ratio**

# ### Research Question 7:   Does The appointment day effect on the 28.5% percentage ?

# In[46]:


#replace No and Yes to 0 and 1 
df["no-show"].replace(["Yes" , "No"], [1 , 0] , inplace= True)


# In[47]:


#check
df.head()


# In[48]:


#plot the weekday
sb.violinplot(data=df, x="weekday", y='no-show',  innner=None, figsize=(20,20))



# **Saturday has the low show and no-show count , in the dataset it has 31 appoinments, and other days are crowded.**

# ## Section 2: exploring the questions that with bivariate variables

# **Bivariate variables lies on the relationship between two variables so before answering the questions need to know the corr between the variables.**

# In[49]:


# get the correlation for all the numeric variables 
df.corr()


# In[50]:


#to see it better 
plt.figure(figsize=(9,5))
sb.heatmap(df.corr().round(3), annot = True , cmap = 'BuPu')
plt.title('Correlation heatmap', size='15')

plt.xticks(rotation='45');


# **Acording to the map there is a relation between age and hipertension = 0.5(low positive correlation )**
# 
# **Also there is a relation btween diabetes and age = 0.2 (very low positive correlation)**

# In[51]:


#relationship between age and hipertension 
plt.figure(figsize=(10,10))
df.groupby('hipertension').age.hist(alpha=0.5);
plt.title('The Relation between Hipertension and Age', fontsize=20)
plt.ylabel('Count', fontsize=15)
plt.legend(['Age','Hipertension'])


# In[52]:


#plot the relation between diabetes and age
plt.figure(figsize=(10,10))
df.groupby('diabetes').age.hist(alpha=0.5);
plt.title('The Relation between Diabetes and Age', fontsize=20)
plt.ylabel('Count', fontsize=15)
plt.legend(['Age','Diabetes'])


# ### Research Question 8:  Does the gab between the scheduled day and the appointment day effect on the no show ratio?
# 

# In[58]:


#plot the relation between diabetes and age
plt.figure(figsize=(10,10))
df.groupby('attend').gab.hist(alpha=0.5);
plt.title('The Relation between the gab and noshow', fontsize=20)
plt.ylabel('Count', fontsize=15)
plt.legend(['gab','attend'])


# **As the gab  between the scheduled day and the appoinment day increase the number of attending patients decrease**

# # Conclusion:

# **After investigation we can tell the following:**
# 
# 1-  71.5% of patients did not miss the appointment , and 28.5% miss their appointment.
# 
# 2- The gender does not effect on the no-show but we can tell that females are taking care of thier health more than males
# The age effects on the no-show ratio, as the age increase the missing patients decrease , the range of approx. (50-60) has most attend count.
# 
# 3- Number of patients without hipertension and diabetes are more likely to did'nt attend the appointment , but in general we can say that these four factors do not effect on the no-show percentage.
# 
# 4- Jardim Cumburi is the highest count for both show (attend) and noshow(missing) , but in general neighbourhood does not effect on the percentage.
# 
# 5- Sending SMS to the patients doesn't reduce the no-show ratio.
# 
# 6- Saturday has the low show and no-show count , in the dataset it has 31 appoinments, and other days are crowded.
# 
# 7- Acording to the map there is a relation between age and hipertension :low positive correlation 
# Also there is a relation btween diabetes and age :very low positive correlation.
# 
# 8- The heatmap shows that there is a strong negative relation between gab and appointment ID , and a very very weak negative relation between age and scholorship(-0.1).
# 
# 9-As the gab between the scheduled day and the appoinment day increase the number of attending patients decrease.
# 

# # References
# **The Data Visualization section in this course**
# 
# https://matplotlib.org/stable/tutorials/index.html
# 
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html

# In[63]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])


# In[ ]:




