#!/usr/bin/env python
# coding: utf-8

# # Analyze A/B Test Results 
# 
# This project will assure you have mastered the subjects covered in the statistics lessons. We have organized the current notebook into the following sections: 
# 
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# - [Final Check](#finalcheck)
# - [Submission](#submission)
# 
# Specific programming tasks are marked with a **ToDo** tag. 
# 
# <a id='intro'></a>
# ## Introduction
# 
# A/B tests are very commonly performed by data analysts and data scientists. For this project, you will be working to understand the results of an A/B test run by an e-commerce website.  Your goal is to work through this notebook to help the company understand if they should:
# - Implement the new webpage, 
# - Keep the old webpage, or 
# - Perhaps run the experiment longer to make their decision.
# 
# Each **ToDo** task below has an associated quiz present in the classroom.  Though the classroom quizzes are **not necessary** to complete the project, they help ensure you are on the right track as you work through the project, and you can feel more confident in your final submission meeting the [rubric](https://review.udacity.com/#!/rubrics/1214/view) specification. 
# 
# >**Tip**: Though it's not a mandate, students can attempt the classroom quizzes to ensure statistical numeric values are calculated correctly in many cases.
# 
# <a id='probability'></a>
# ## Part I - Probability
# 
# To get started, let's import our libraries.

# In[1]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# ### ToDo 1.1
# Now, read in the `ab_data.csv` data. Store it in `df`. Below is the description of the data, there are a total of 5 columns:
# 
# <center>
# 
# |Data columns|Purpose|Valid values|
# | ------------- |:-------------| -----:|
# |user_id|Unique ID|Int64 values|
# |timestamp|Time stamp when the user visited the webpage|-|
# |group|In the current A/B experiment, the users are categorized into two broad groups. <br>The `control` group users are expected to be served with `old_page`; and `treatment` group users are matched with the `new_page`. <br>However, **some inaccurate rows** are present in the initial data, such as a `control` group user is matched with a `new_page`. |`['control', 'treatment']`|
# |landing_page|It denotes whether the user visited the old or new webpage.|`['old_page', 'new_page']`|
# |converted|It denotes whether the user decided to pay for the company's product. Here, `1` means yes, the user bought the product.|`[0, 1]`|
# </center>
# Use your dataframe to answer the questions in Quiz 1 of the classroom.
# 
# 
# >**Tip**: Please save your work regularly.
# 
# **a.** Read in the dataset from the `ab_data.csv` file and take a look at the top few rows here:

# In[2]:


ab_df=pd.read_csv("ab_data.csv")
df=ab_df.copy()
df.head()


# **b.** Use the cell below to find the number of rows in the dataset.

# In[3]:


r,h = df.shape
print("We have {} of rows in our dataset".format(r))


# **c.** The number of unique users in the dataset.

# In[4]:


L=df['user_id'].unique()
print("We have {} of unique in our dataset and {} douplicated users".format(len(L),df['user_id'].duplicated().sum()))


# **d.** The proportion of users converted.

# In[5]:


print("The proportion of users converted = {} ".format(df['converted'].mean()))


# **e.** The number of times when the "group" is `treatment` but "landing_page" is not a `new_page`.

# In[6]:


# df_filter variable to put the new dataset after filtering 
df_filter= df.query('group == "treatment" and landing_page != "new_page"')
print("We have {} number of times when the group is treatment landing_page not a new_page".format(df_filter.shape[0]))


# **f.** Do any of the rows have missing values?

# In[7]:


if df.isnull().values.any() == False:
    print('No row(s) have missing values')
else:
    print('There are row(s) which have missing values\n'+'\n'.join(map(
    lambda x : str(x[1])
    ,(filter(lambda z: z[0] != False, zip(df.isnull().any(), df.columns.values)))
)))


# ### ToDo 1.2  
# In a particular row, the **group** and **landing_page** columns should have either of the following acceptable values:
# 
# |user_id| timestamp|group|landing_page|converted|
# |---|---|---|---|---|
# |XXXX|XXXX|`control`| `old_page`|X |
# |XXXX|XXXX|`treatment`|`new_page`|X |
# 
# 
# It means, the `control` group users should match with `old_page`; and `treatment` group users should matched with the `new_page`. 
# 
# However, for the rows where `treatment` does not match with `new_page` or `control` does not match with `old_page`, we cannot be sure if such rows truly received the new or old wepage.  
# 
# 
# Use **Quiz 2** in the classroom to figure out how should we handle the rows where the group and landing_page columns don't match?
# 
# **a.** Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz.  Store your new dataframe in **df2**.

# In[8]:


# Remove the inaccurate rows, and store the result in a new dataframe df2
df2=df.query('group == "control" and landing_page == "old_page" or  group == "treatment" and landing_page == "new_page"')


# In[9]:


# Double Check all of the incorrect rows were removed from df2 - 
# Output of the statement below should be 0
#df3=df2.query(('group == "control" & landing_page == "new_page"') or ('group == "treatment" & landing_page == "old_page"'),inplace = False)
df3=df2.copy()
r=df3.shape[0]
print("After Double check row(s) will be {}".format(r))


# ### ToDo 1.3  
# Use **df2** and the cells below to answer questions for **Quiz 3** in the classroom.

# **a.** How many unique **user_id**s are in **df2**?

# In[10]:


print('We have {} unique user ids in our df2'.format(df2.user_id.nunique()))


# **b.** There is one **user_id** repeated in **df2**.  What is it?

# In[11]:


df2.user_id.duplicated().sum()


# **c.** Display the rows for the duplicate **user_id**? 

# In[12]:


duplicate = df2[df2.duplicated('user_id')]
duplicate


# **d.** Remove **one** of the rows with a duplicate **user_id**, from the **df2** dataframe.

# In[13]:


# Remove one of the rows with a duplicate user_id..
# Hint: The dataframe.drop_duplicates() may not work in this case because the rows with duplicate user_id are not entirely identical. 
df2.drop_duplicates(subset=['user_id'],inplace=True)
# Check again if the row with a duplicate user_id is deleted or not
duplicate = df2[df2.duplicated('user_id')]
duplicate


# ### ToDo 1.4  
# Use **df2** in the cells below to answer the quiz questions related to **Quiz 4** in the classroom.
# 
# **a.** What is the probability of an individual converting regardless of the page they receive?<br><br>
# 
# >**Tip**: The probability  you'll compute represents the overall "converted" success rate in the population and you may call it $p_{population}$.
# 
# 

# In[14]:


print("The proportion of users converted = {}".format(df2['converted'].mean()))


# **b.** Given that an individual was in the `control` group, what is the probability they converted?

# In[15]:


control_df = df2.query('group == "control"')
print("The proportion of users converted = {}".format(control_df['converted'].mean()))


# **c.** Given that an individual was in the `treatment` group, what is the probability they converted?

# In[16]:


treatment_df = df2.query('group == "treatment"')
print("The proportion of users converted = {}".format(treatment_df['converted'].mean()))


# >**Tip**: The probabilities you've computed in the points (b). and (c). above can also be treated as conversion rate. 
# Calculate the actual difference  (`obs_diff`) between the conversion rates for the two groups. You will need that later.  

# In[17]:


# Calculate the actual difference (obs_diff) between the conversion rates for the two groups.
print("The obs_diff = {}%".format((control_df['converted'].mean()*100)-(treatment_df['converted'].mean()*100)))


# **d.** What is the probability that an individual received the new page?

# In[18]:


newpage_df = df2.query('landing_page == "new_page"')
print("The probability of new page = {} %".format(newpage_df['converted'].mean()*100))


# In[19]:


oldpage_df = df2.query('landing_page == "old_page"')
print("The probability of old page = {} %".format(oldpage_df['converted'].mean()*100))


# **e.** Consider your results from parts (a) through (d) above, and explain below whether the new `treatment` group users lead to more conversions.

# **It is too close.**
# 
# **The proportion of users converted = 11.959708724499627 %.**
# 
# **The proportion of users treatment converted = 11.880806551510565 %.**
# 
# ### No profe new page has alot of visitors

# <a id='ab_test'></a>
# ## Part II - A/B Test
# 
# Since a timestamp is associated with each event, you could run a hypothesis test continuously as long as you observe the events. 
# 
# However, then the hard questions would be: 
# - Do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  
# - How long do you run to render a decision that neither page is better than another?  
# 
# These questions are the difficult parts associated with A/B tests in general.  
# 
# 
# ### ToDo 2.1
# For now, consider you need to make the decision just based on all the data provided.  
# 
# > Recall that you just calculated that the "converted" probability (or rate) for the old page is *slightly* higher than that of the new page (ToDo 1.4.c). 
# 
# If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should be your null and alternative hypotheses (**$H_0$** and **$H_1$**)?  
# 
# You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the "converted" probability (or rate) for the old and new pages respectively.

# >null hypotheses **$p_{new}$** > **$p_{old}$**

# ### ToDo 2.2 - Null Hypothesis $H_0$ Testing
# Under the null hypothesis $H_0$, assume that $p_{new}$ and $p_{old}$ are equal. Furthermore, assume that $p_{new}$ and $p_{old}$ both are equal to the **converted** success rate in the `df2` data regardless of the page. So, our assumption is: <br><br>
# <center>
# $p_{new}$ = $p_{old}$ = $p_{population}$
# </center>
# 
# In this section, you will: 
# 
# - Simulate (bootstrap) sample data set for both groups, and compute the  "converted" probability $p$ for those samples. 
# 
# 
# - Use a sample size for each group equal to the ones in the `df2` data.
# 
# 
# - Compute the difference in the "converted" probability for the two samples above. 
# 
# 
# - Perform the sampling distribution for the "difference in the converted probability" between the two simulated-samples over 10,000 iterations; and calculate an estimate. 
# 
# 
# 
# Use the cells below to provide the necessary parts of this simulation.  You can use **Quiz 5** in the classroom to make sure you are on the right track.

# **a.** What is the **conversion rate** for $p_{new}$ under the null hypothesis? 

# In[20]:


newpage_df = df2
print("The probability of new page = {} ".format(newpage_df['converted'].mean()))


# **b.** What is the **conversion rate** for $p_{old}$ under the null hypothesis? 

# In[21]:


oldpage_df = df2
print("The probability of old page = {} ".format(oldpage_df['converted'].mean()))


# **c.** What is $n_{new}$, the number of individuals in the treatment group? <br><br>
# *Hint*: The treatment group users are shown the new page.

# In[22]:


treatment_df = df2.query('group == "treatment"')
n_new = treatment_df.shape[0]
print("The treatment group users are shown the new page = {} ".format(n_new))


# **d.** What is $n_{old}$, the number of individuals in the control group?

# In[23]:


control_df = df2.query('group == "control"')
n_old = control_df.shape[0]
print("The treatment group users are shown the old page = {} ".format(n_old))


# **e. Simulate Sample for the `treatment` Group**<br> 
# Simulate $n_{new}$ transactions with a conversion rate of $p_{new}$ under the null hypothesis.  <br><br>
# *Hint*: Use `numpy.random.choice()` method to randomly generate $n_{new}$ number of values. <br>
# Store these $n_{new}$ 1's and 0's in the `new_page_converted` numpy array.
# 

# In[24]:


# Simulate a Sample for the treatment Group
new_page_converted_v = np.random.choice([1,0],size=treatment_df.shape[0],p=[newpage_df['converted'].mean(),(1-newpage_df['converted'].mean())])
new_page_converted = new_page_converted_v.mean()
new_page_converted


# **f. Simulate Sample for the `control` Group** <br>
# Simulate $n_{old}$ transactions with a conversion rate of $p_{old}$ under the null hypothesis. <br> Store these $n_{old}$ 1's and 0's in the `old_page_converted` numpy array.

# In[25]:


# Simulate a Sample for the control Group
old_page_converted_v = np.random.choice([1,0],size=control_df.shape[0],p=[oldpage_df['converted'].mean(),(1-oldpage_df['converted'].mean())])
old_page_converted = old_page_converted_v.mean()
old_page_converted


# **g.** Find the difference in the "converted" probability $(p{'}_{new}$ - $p{'}_{old})$ for your simulated samples from the parts (e) and (f) above. 

# In[26]:


diff_converted = new_page_converted - old_page_converted
diff_converted


# 
# **h. Sampling distribution** <br>
# Re-create `new_page_converted` and `old_page_converted` and find the $(p{'}_{new}$ - $p{'}_{old})$ value 10,000 times using the same simulation process you used in parts (a) through (g) above. 
# 
# <br>
# Store all  $(p{'}_{new}$ - $p{'}_{old})$  values in a NumPy array called `p_diffs`.

# In[27]:


# Sampling distribution 
p_diffs = []
#size = df2.shape[0]
#for _ in range(10000):
#    df_samp = df2.sample(size,replace=True)
#    new_page_converted = df_samp.query('landing_page == "new_page"').converted.mean()
#    old_page_converted = df_samp.query('landing_page == "old_page"').converted.mean()
#    p_diffs.append(new_page_converted -  old_page_converted) 

new_converted_simulation = np.random.binomial(n_new, new_page_converted , 10000)/n_new
old_converted_simulation = np.random.binomial(n_old, old_page_converted, 10000)/n_old
p_diffs = new_converted_simulation - old_converted_simulation
p_diffs

#p_diffs


# In[28]:


p_diffs = np.array(p_diffs)


# **i. Histogram**<br> 
# Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.<br><br>
# 
# Also, use `plt.axvline()` method to mark the actual difference observed  in the `df2` data (recall `obs_diff`), in the chart.  
# 
# >**Tip**: Display title, x-label, and y-label in the chart.

# In[29]:


plt.hist(p_diffs);


# **j.** What proportion of the **p_diffs** are greater than the actual difference observed in the `df2` data?

# In[30]:


p_diffs = np.random.normal(0, p_diffs.std(), p_diffs.size)
obs_diff = df2.query('group == "treatment"').converted.mean() - df2.query('group == "control"').converted.mean()

plt.hist(p_diffs)
plt.axvline(obs_diff , color="r");


# In[31]:


(p_diffs > obs_diff).mean()

#(treatment_df.converted.mean() - control_df.converted.mean())).mean()


# **k.** Please explain in words what you have just computed in part **j** above.  
#  - What is this value called in scientific studies?  
#  - What does this value signify in terms of whether or not there is a difference between the new and old pages? *Hint*: Compare the value above with the "Type I error rate (0.05)". 

# >**P-value =0.90490000000000004 So it is > 0.05.**

# 
# 
# **l. Using Built-in Methods for Hypothesis Testing**<br>
# We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. 
# 
# Fill in the statements below to calculate the:
# - `convert_old` : number of conversions with the old_page
# - `convert_new`: number of conversions with the new_page
# - `n_old`  : number of individuals who were shown the old_page
# - `n_new`  : number of individuals who were shown the new_page
# 

# In[32]:


import statsmodels.api as sm

# number of conversions with the old_page
convert_old = df2[(df2.group == "control")]["converted"]

# number of conversions with the new_page
convert_new =df2[(df2.group == "treatment")]["converted"]

# number of individuals who were shown the old_page
n_old = convert_old.shape[0]
# number of individuals who received new_page
n_new =  convert_new.shape[0]


# **m.** Now use `sm.stats.proportions_ztest()` to compute your test statistic and p-value.  [Here](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportions_ztest.html) is a helpful link on using the built in.
# 
# The syntax is: 
# ```bash
# proportions_ztest(count_array, nobs_array, alternative='larger')
# ```
# where, 
# - `count_array` = represents the number of "converted" for each group
# - `nobs_array` = represents the total number of observations (rows) in each group
# - `alternative` = choose one of the values from `[‘two-sided’, ‘smaller’, ‘larger’]` depending upon two-tailed, left-tailed, or right-tailed respectively. 
# >**Hint**: <br>
# It's a two-tailed if you defined $H_1$ as $(p_{new} = p_{old})$. <br>
# It's a left-tailed if you defined $H_1$ as $(p_{new} < p_{old})$. <br>
# It's a right-tailed if you defined $H_1$ as $(p_{new} > p_{old})$. 
# 
# The built-in function above will return the z_score, p_value. 
# 
# ---
# ### About the two-sample z-test
# Recall that you have plotted a distribution `p_diffs` representing the
# difference in the "converted" probability  $(p{'}_{new}-p{'}_{old})$  for your two simulated samples 10,000 times. 
# 
# Another way for comparing the mean of two independent and normal distribution is a **two-sample z-test**. You can perform the Z-test to calculate the Z_score, as shown in the equation below:
# 
# $$
# Z_{score} = \frac{ (p{'}_{new}-p{'}_{old}) - (p_{new}  -  p_{old})}{ \sqrt{ \frac{\sigma^{2}_{new} }{n_{new}} + \frac{\sigma^{2}_{old} }{n_{old}}  } }
# $$
# 
# where,
# - $p{'}$ is the "converted" success rate in the sample
# - $p_{new}$ and $p_{old}$ are the "converted" success rate for the two groups in the population. 
# - $\sigma_{new}$ and $\sigma_{new}$ are the standard deviation for the two groups in the population. 
# - $n_{new}$ and $n_{old}$ represent the size of the two groups or samples (it's same in our case)
# 
# 
# >Z-test is performed when the sample size is large, and the population variance is known. The z-score represents the distance between the two "converted" success rates in terms of the standard error. 
# 
# Next step is to make a decision to reject or fail to reject the null hypothesis based on comparing these two values: 
# - $Z_{score}$
# - $Z_{\alpha}$ or $Z_{0.05}$, also known as critical value at 95% confidence interval.  $Z_{0.05}$ is 1.645 for one-tailed tests,  and 1.960 for two-tailed test. You can determine the $Z_{\alpha}$ from the z-table manually. 
# 
# Decide if your hypothesis is either a two-tailed, left-tailed, or right-tailed test. Accordingly, reject OR fail to reject the  null based on the comparison between $Z_{score}$ and $Z_{\alpha}$. We determine whether or not the $Z_{score}$ lies in the "rejection region" in the distribution. In other words, a "rejection region" is an interval where the null hypothesis is rejected iff the $Z_{score}$ lies in that region.
# 
# >Hint:<br>
# For a right-tailed test, reject null if $Z_{score}$ > $Z_{\alpha}$. <br>
# For a left-tailed test, reject null if $Z_{score}$ < $Z_{\alpha}$. 
# 
# 
# 
# 
# Reference: 
# - Example 9.1.2 on this [page](https://stats.libretexts.org/Bookshelves/Introductory_Statistics/Book%3A_Introductory_Statistics_(Shafer_and_Zhang)/09%3A_Two-Sample_Problems/9.01%3A_Comparison_of_Two_Population_Means-_Large_Independent_Samples), courtesy www.stats.libretexts.org
# 
# ---
# 
# >**Tip**: You don't have to dive deeper into z-test for this exercise. **Try having an overview of what does z-score signify in general.** 

# In[33]:


import statsmodels.api as sm
# ToDo: Complete the sm.stats.proportions_ztest() method arguments
z_score, p_value = sm.stats.proportions_ztest([convert_new.sum(),convert_old.sum()],[n_new,n_old],alternative='larger')
print(z_score, p_value)


# **n.** What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?<br><br>
# 
# >**Tip**: Notice whether the p-value is similar to the one computed earlier. Accordingly, can you reject/fail to reject the null hypothesis? It is important to correctly interpret the test statistic and p-value.

# >**p-value = 0.905058312759   lead us to accept the null hypothsis.**

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# ### ToDo 3.1 
# In this final part, you will see that the result you achieved in the A/B test in Part II above can also be achieved by performing regression.<br><br> 
# 
# **a.** Since each row in the `df2` data is either a conversion or no conversion, what type of regression should you be performing in this case?

# >**Logistic regression.**

# **b.** The goal is to use **statsmodels** library to fit the regression model you specified in part **a.** above to see if there is a significant difference in conversion based on the page-type a customer receives. However, you first need to create the following two columns in the `df2` dataframe:
#  1. `intercept` - It should be `1` in the entire column. 
#  2. `ab_page` - It's a dummy variable column, having a value `1` when an individual receives the **treatment**, otherwise `0`.  

# In[34]:


df2["intercept"] = 1
df2 = df2.join(pd.get_dummies(df2["group"]))
df2.rename(columns = {"treatment" : "ab_page"} , inplace=True)
df2.head()


# **c.** Use **statsmodels** to instantiate your regression model on the two columns you created in part (b). above, then fit the model to predict whether or not an individual converts. 
# 

# In[35]:


logistic_regression = sm.Logit(df2["converted"] , df2[["intercept" , "ab_page"]]).fit()


# **d.** Provide the summary of your model below, and use it as necessary to answer the following questions.

# In[36]:


logistic_regression.summary2()


# **e.** What is the p-value associated with **ab_page**? Why does it differ from the value you found in **Part II**?<br><br>  
# 
# **Hints**: 
# - What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in **Part II**? 
# - You may comment on if these hypothesis (Part II vs. Part III) are one-sided or two-sided. 
# - You may also compare the current p-value with the Type I error rate (0.05).
# 

# >**p-value with ab_page = 0.0150.**

# **f.** Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# >**X variables realted to Y variable we assume that there is no relationship between X variable to another X variable.**

# **g. Adding countries**<br> 
# Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives in. 
# 
# 1. You will need to read in the **countries.csv** dataset and merge together your `df2` datasets on the appropriate rows. You call the resulting dataframe `df_merged`. [Here](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.join.html) are the docs for joining tables. 
# 
# 2. Does it appear that country had an impact on conversion?  To answer this question, consider the three unique values, `['UK', 'US', 'CA']`, in the `country` column. Create dummy variables for these country columns. 
# >**Hint:** Use `pandas.get_dummies()` to create dummy variables. **You will utilize two columns for the three dummy variables.** 
# 
#  Provide the statistical output as well as a written response to answer this question.

# In[37]:


# Read the countries.csv
country_df = pd.read_csv("countries.csv")


# In[38]:


# Join with the df2 dataframe
df_merged = country_df.set_index('user_id').join(df2.set_index('user_id'), how='inner')
df_merged.head()


# In[39]:


# Create the necessary dummy variables
df_merged["intercept"] = 1 
df_merged[["CA" , "UK" , "US"]] = pd.get_dummies(df_merged["country"])
df_merged.head()


# **h. Fit your model and obtain the results**<br> 
# Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if are there significant effects on conversion.  **Create the necessary additional columns, and fit the new model.** 
# 
# 
# Provide the summary results (statistical output), and your conclusions (written response) based on the results. 
# 
# >**Tip**: Conclusions should include both statistical reasoning, and practical reasoning for the situation. 
# 
# >**Hints**: 
# - Look at all of p-values in the summary, and compare against the Type I error rate (0.05). 
# - Can you reject/fail to reject the null hypotheses (regression model)?
# - Comment on the effect of page and country to predict the conversion.
# 

# In[40]:


# Fit your model, and summarize the results
df_merged["Page_CA"] = df_merged["CA"] * df_merged["ab_page"]
df_merged["Page_UK"] = df_merged["UK"] * df_merged["ab_page"]
df_merged["Page_US"] = df_merged["US"] * df_merged["ab_page"]
sm.Logit(df_merged["converted"] , df_merged[["intercept" , "ab_page" , "CA" , "UK" , "Page_CA" , "Page_UK"]]).fit().summary2()


# >**we don't rejecting the null hypothesis .**
# >**The Coutry dosen't have impact on conversion .**

# <a id='finalcheck'></a>
# ## Final Check!
# 
# Congratulations!  You have reached the end of the A/B Test Results project!  You should be very proud of all you have accomplished!
# 
# > **Tip**: Once you are satisfied with your work here, check over your notebook to make sure that it satisfies all the specifications mentioned in the rubric. You should also probably remove all of the "Hints" and "Tips" like this one so that the presentation is as polished as possible.
# 
# 
# <a id='submission'></a>
# ## Submission
# You may either submit your notebook through the "SUBMIT PROJECT" button at the bottom of this workspace, or you may work from your local machine and submit on  the last page of this project lesson.  
# 
# 1. Before you submit your project, you need to create a .html or .pdf version of this notebook in the workspace here. To do that, run the code cell below. If it worked correctly, you should get a return code of 0, and you should see the generated .html file in the workspace directory (click on the orange Jupyter icon in the upper left).
# 
# 
# 2. Alternatively, you can download this report as .html via the **File** > **Download as** submenu, and then manually upload it into the workspace directory by clicking on the orange Jupyter icon in the upper left, then using the Upload button.
# 
# 
# 3. Once you've done this, you can submit your project by clicking on the "Submit Project" button in the lower right here. This will create and submit a zip file with this .ipynb doc and the .html or .pdf version you created. Congratulations!

# In[41]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Analyze_ab_test_results_notebook.ipynb'])

