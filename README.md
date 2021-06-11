# Auto-ticket-multigroup-classifier
--------------------
**Problem**
Manual assignment of incidents is time consuming and requires human efforts. There may be mistakes due to human errors and resource consumption is carried out ineffectively because of the misaddressing. On the other hand, manual assignment increases the response and resolution times which result in user satisfaction deterioration / poor customer service.

**Business Value Proposition**
In the support process, incoming incidents are analyzed and assessed by organization’s support teams to fulfill the request. In many organizations, better allocation and effective usage of the valuable support resources will directly result in substantial cost savings.

**Abstract**
An attempt at Leveraging Machine Learning and Artifical intelligence to automatically classify tickets and assign them to the right owner in a timely manner to save effort, increase user satifaction and improve throughput in the ticketing pipeline of an organization.

**Objective:**
To undertake a multi-faceted project that demonstrates your understanding and mastery of the key conceptual and technological aspects of Deep Learning.
To develop an understanding of how challenging human-level problems can be approached and solved using a combination of tools and techniques.
To understand current scenarios in deep learning, understand the practicalities and the trade-offs that need to be made when solving a problem in real life.

## 1. Problem interpretation:
  * Understand the data  
  * Make an abstract or an overview based on your approach  
  * Break the problem into smaller tasks  
  * Discuss among your teammates and share responsibilities</br>  

## 2. Data analysis and preprocessing: 
  * Visual displays are powerful when used well, so think carefully about the information the display.  
  * Include any insightful visualization  
  * Share and explain particularly meaningful features, interactions or summary of data  
  * Display examples to input in your model  
  * Explain changes to be incorporated into data so that it becomes ready for the model</br>  

## 3. Modeling:
  * What kind of neural network you have used and why?  
  * What progress you have made towards your intended solution?</br>  
## Since we’re interested in associating text with a relevant classifier, 
## we can use the categorical variable “Assignment Group” as our label/target varibale in each row in our dataframe.

## Top classifier group GRP_0 is having 3976 entires if we compare with total count (8500) which is about 47% of the data.
## Total 74 unique Labels are there.
## Total Count of Rowas are 8500. where as there as missing values in Short description and Description attributes

# Handling the Missing values
## Added a text of "######" into the missing datas as in future we need to findout the similary of Short description and Description
So as not to loss any information.

Also while comparing the similarity between Short Description and Description the text of  "######" will not be matched hence will automatically ignored in our final description text.

![image](https://user-images.githubusercontent.com/65825617/121719678-91b61580-cb00-11eb-8d42-555bbea188f9.png)

##### From Observation from analysing the Short Description column
* There are 8 null values
* The top most issue the is repeated in the column is "password reset" with 38 occurances
* There are 7482 unique words
* From the WordCloud ,the top most text are "issue", "outlook", "cant" , "login" etc


## Top classifier group GRP_0 is having 3976 entires if we compare with total count (8500) which is about 47% of the data.
## That means we need to check is there any class imbalance is there. If so we need to use relevant groups for our analysis to avoid noises in our datasets.

![image](https://user-images.githubusercontent.com/65825617/121720501-d2159380-cb00-11eb-8d49-15eb16ee48f9.png)

##### Observations from analysis of the Assignment Group column

* There are no null values
* We see that there are total 74 assignments groups in the data
* Top 1 group worked on 46% of the tickets
* Top 1-Top 3 worked on 57% of the tickets
* Top 1-Top 5 worked on 64% of the tickets
* Top 1-Top 10 worked on 75% of the tickets
* Bottom 10 groups contain 1 or 2 cases asssigned to them

### Text Preprocessing
Text preprocessing is the process of transferring text from human language to machine-readable format for further processing. After a text is obtained, we start with text normalization. Text normalization includes:
-Converting the all texts into one language i.e. in Engish
- converting all letters to lower or upper case
- removing punctuations, accent marks and other diacritics
- converting numbers into words or removing numbers
- removing white spaces
- removing stop words, sparse terms, and particular words
- text canonicalization

#Let us find out the similary between Translated Description and Translated Short Description
#To do so we will use Fuzzywizzy to compare the similarity.

df1["NotMatch"]=""
df1["Final Description"]=""
from fuzzywuzzy import process, fuzz
for i in range(len(df1["Description-Translated"])):
  
  if fuzz.token_set_ratio(str(df1["Description-Translated"][i]),str(df1["Short Description-Translated"][i])) <90:
    df1["Final Description"][i]=str(df1["Description-Translated"][i])+" "+str(df1["Short Description-Translated"][i])
    df1["NotMatch"][i]="Not Matched"
  elif fuzz.token_set_ratio(str(df1["Description-Translated"][i]),str(df1["Short Description-Translated"][i])) >=90:
    df1["NotMatch"][i]="Matched"
    df1["Final Description"][i]=str(df1["Description-Translated"][i])     
        
# Let us find out the similarity between imbalance datasets and try to merge them into common groups/similar groups

# Finding similarity group of imbalance data and then mapping all texts with matching grops to overcome the imbalanced without eliminating any information

# As All the texts of imbalance group are matching with GRP_49, so we are merging all imbalance groups into "GRP_49" to make a balance dataset

##Over Sampling the the minority groups by Using Random Over Sampler

# Created different ML models on Balanced Datasets
##Comparision between different models

![image](https://user-images.githubusercontent.com/65825617/121721292-bced3480-cb01-11eb-881a-ac95afe5d4bc.png)

![image](https://user-images.githubusercontent.com/65825617/121721413-dd1cf380-cb01-11eb-836b-071190b40a1b.png)


***Comparision Deep Learning Model

![image](https://user-images.githubusercontent.com/65825617/121721699-28370680-cb02-11eb-9f86-695e89fa4b61.png)

## Summary
The Best Machine Learning Algorithm for this problem is:-
•OneVesRestLogistic (Train and Test 92% Accuracy using Up-Sampling)
•The Best AI Algorithm is for this problem is:-
•GRU (Train and Test 93% Accuracy using Up-Sampling)
 

•As observed the given dataset is highly imbalanced for group zero and there is limited data available for the rest of the classes.
•Addition to the imbalance dataset, another issue is the number of training samples. In our dataset we have 8500 examples including training and test sets. These numbers are not sufficient for DM models which generally requires data in Millions.

#Limitation
We are also not sure about measurement errors. One error can be assigning wrong class labels to many examples. If these misclassifications are observed in minority-class then our model will not be able to predict the correct group during production run. 

# Future Scope
The scope of the selected model is to assign tickets to the right group for issue resolution. As observed, most of the tickets are related to Password reset, Account lock, Unable to log in for which resolutions are easily available. So, we can extend this project to provide resolution to commonly occurring issues and only sending non common tickets to the next level where expert involvement is required.  

•We can also extend scope of this project by implementing an Automated Question Answering model using BOT technique, where users can type issues in the question field and bot provide answers in form of resolution by running a selected AI model in background.

•Urgency classification: We can extend the scope to categories issues in terms of their urgency. If text contains words such as ‘right away’, ‘immediately’, ‘ASAP’ etc. in such a case our model should identify these words and priorities the tickets.

•Sentiment Grading: Other scope can be using sentiment analysis to analyze text entered by the user and identify the degree of sentiment expressed and decide dissatisfied users. By doing so we can assign tickets to experience people to handle such users and thus keep the overall satisfaction level intact.

• Other than expanding the scope on the Application side, we can think of improving scope by trying different word embedding methods. In our project we have used the GloVe word embedding method during AI model building and we got a fair accuracy score. We can also explore other latest word embedding techniques like FastText, Deep Contextualized Word Representations while building models for Automated Question answering and Sentiment grading.



