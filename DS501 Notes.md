## 1.  Prompt & Framing

### 1.1 Framing a Business Question to a Data Science Project

- Should Facebook add a feature of "videos your friends liked"?

Feature: show user videos their friends have liked

Objective of Facebook: increase user engagement, which is measured by interactions between friends

Adding a video feature could increase time on site because videos are longer (than liking or commenting a friend's post), but it could have a negative impact on human interactions because it's a passive action.

Choose key metrics is the first step for A/B/ testing. Long term metrics: engagement, short term metrics: click through rate, video views, time on site, etc

🌟 **think in different dimensions** 🌟

- Should Facebook add a feature of "close friends notification"?
  - Feature: text notification for close friend's update
  - Step 1 - Clarify what is "close".	
	- Data: like, comment, share, private message, newsfeed. -\> coding question
	- Key: mutual interaction instead of one-way (commenting on famous people's posts doesn't make you a close friend)
  - Step 2 - Define KPI metrics
	- short term: text open rate
	- long term: user engagement
  - Step 3 - Design A/B testing
	- **self opt-in** before pushing notification
	- Problems to use people who opt in and out as comparison groups: they are already different
	- Problems to do A/B testing within opt-in group: user experience bad as they already signed up
	- Solution: **randomize -\> opt-in**	
	- hypothesis: user might spend more time on FB?

- Is "Amazon Prime Now" Feasible?
  - Product: upgrade of Amazon Prime, delivery in 2 hours
  - **Analyze cost and benefit**
	- Cost: transportation and logistic cost big increase (can't wait for more boxes, not very efficient)
	  - who pays for the cost? 
		- increase price for end user? test elasticity to decide price point
		- charge membership
		- may very well increase sales in general (Prime is very successful in that sense)
	- Benefit: profit gain
	  - but sometimes companies optimize growth instead of revenue: early stage Uber
  - predict enrollment rate
	- need other data/experiments to understand potential enrollment rate and the market
	- external market research, compare with local retail market, third party data
	- what kind of products people want Prime Now?
	  - groceries
	  - emergency products

- Use 1Point3Acres Data to Predict Chance of Program Admission
  - applicants background (TOEFL, GRE, School) -\> probability of admission at SCS CMU
	- assume all data is available
  - **selection bias**: people who got an offer likes to post, the least competitive candidate is the least likely to post
	- all applicants for a program from school (complete dataset)
	- survey response rate around 1%, can be less than 0.1%. 
	  - how to increase response rate?
  - be familiar with products/team, otherwise difficult to come up with a feature on the fly
	- use the product as a customer
	- think about the problem from
	  - supply side - Uber driver
	  - demand side - Uber rider
	  - platform side - Uber
	- think as the owner/CEO, what if you are Jeff Bezos

### 1.2 How are you graded?

- a checklist of points
  - asked clarifying questions
  - propose reasonable hypothesis
  - use data as evidence to backup hypothesis
  - assumptions in data analysis, do they hold in this case
  - is your recommendation useful
- systematically
  - hypothesis
  - frame a question
  - what data i need
  - analysis
  - results
- Strategy 总分总
  - 5000 ft view
  - expand each item one by one, in order
  - go back to high level view between each step
  - use whiteboard
  - "This is what I am going to say." Say it. "This is what I said."
- Tips
  - not brute force points
  - Ask "Am I on the right track?" every some time, take feedback
  - teach them something new!

### 1.3 Evaluate the "Snap Stories" Feature

- Clarifying the feature
- Analysis
  - objective
	- user engagement/new user signup/active user or 
	- increase revenue (may contain ads)
  - metric design
	- what data: interaction, active time, app open time, ads click rate
	- short term, long term, individual, interaction
	- -\> conversion rate
  - A/B test
	- metrics
	  - pick a key metric
	  - other metrics as g\* metric
	  - discuss short term long term tradeoff
	- randomization unit
	  - social network products can't really test individually 
	  - people are independent
	- check data quality
	- hypothesis test
	  - continuous: T-test
	  - small dataset: Chi-square test
	  - when to stop experiment
		- pre-define
		- sequential test
  - ML model to improve feature relevance
	- recommender system: more relevant stories

- Question prompt: a scenario, new feature, etc
- Ask clarifying question
  - link business objective to quantifiable outcomes
	- revenue
	- profit
	- growth
	- user generated content quality
	- page view
	- active user
  - understand how to measure the objective
	- how to measure if a product is good? launch a product change, is it good for the business?
	- how to define "close" friends using observable data
	- how to guess a hidden attribute like "first name of a user"
  - how to understand a change in a metric
	- observe a change in business health, understand why
	- conversion rate dropped 20%, why
- Take home
  - discover potential issues/errors/dirtiness
  - clean up the data: -999 to N/A
  - discover interesting/useful/actionable pattern from data
  - make a recommendation/conclusion
  - back up your claim with proper analysis & charts
  - document hypothesis you have made

## 2. Hypothesis: Situation

- reasonable hypothesis
  - user behavior change after parents sign up
	- behavior reduce, fewer photo share
	- set up secret group
  - if give discount/coupon
	- more sales in the short term
	- more revenue but lower margin
	- 垃圾用户 increases
	- fraud increases
	- user retention reduces
	- return reduces? only bought to get free shipping
- think as if you are a PM
  - requires product sense and intuition
  - spontaneously come up with many product features and think about ways to test out these ideas
  - in a big firm, 20% features A/B tested were launched

## 3. Data & Metrics

### 3.1 Data

- what data to use as evidence for/against the hypothesis
  - not to confirm the hypothesis, to test the hypothesis
  - be open-minded if data doesn't support your hypothesis
  - e.g. measure how good your search engine is
	- Google search query count can't be used by iteself to 
	  - if feature good, user search increase
	  - if feature bad, user need to search more to get the answer
	  - same direction of change for query count
	- how many results a user clicked
	  \- 
- what is available? what can be measured?
  - e.g. customer satisfaction/preference/intent
  - survey is biased, and response rate is low, net promoter score
  - usually user won't directly tell you what you want
  - need to infer with the data available
- how are they collected
  - what data can a cell phone collect?
	- location, active time, time, os, battery life (Uber increases price if your battery life is low because you NEED that Uber), contact (if granted), zone, active or not, internet speed, sensor (3D) -\> user activity classification
- data quality
  - garbage in garbage out
  - how to check data quality
  - how to clean
  - is there sampling bias

### 3.2 Metric

- define metrics
  - AARRR framework
- Normalization

- Data Sense
  - direction of change?
	- increase or decrease
	- e.g. Google Docs
	  - active time: the higher the better? could be because features are difficult to use
  - meaning
  - imagine what unseen data will look like
	- distribution: e.g. Amazon price distribution: long tail
	- range: positive, a couple thousand dollars
	- missing data?
	- majority values: 0\~500
  - extrapolation
	- give last month's metric, what will happen next month? 
	  - think about seasonality
  - relationship between 2 metrics, draw on whiteboard
	- e.g. active user vs. time
	- linear
	- flatten out
	- exponential growth
	- step function
  - skew
  - missing data
  - seasonality
  - one-time events
  - 80-20 rule
	- is there a categorical variable, where top X account for Y%
	- e.g. top 10 cities of Uber account for x% of revenue
- if coupon -\> conversion rate increase, but unit price decreased so revenue may not increase
  - teneblazation: if you offer a 80% discount for a 1-year subscription, a lot of people will buy it, but won't buy again for the next year
  - not just buy, need to not cancel, not return, not contact customer support to complain

### 2.4 analysis & understanding

- choose appropriate method
  - A/B test
  - modeling
  - funnel analysis: lay out user flow with steps, drop off, conversion rate
- suggest proper segments
  - location/age/etc
- %change vs absolute change
- Simpson's paradox
  - consider change in baseline mix
- correlation vs causation

### 2.5 conclusion & recommendation

- recommend action
  - clear & concise: TL;DR
  - focus on high level thinking first 总分总
  - tailor your solution for the audience
	-  if audience is PM/executive/statistician
	- 5000 ft view for executive
	- ground level view (all technical details) for your peers/tech lead
	- middle ground view for engineer/PM/ops/business partners
- morph into a technical problem in stats/ML
  - Evaluate product: AB test
  - Hint at some prediction/modeling -\> supervised learning
  - Find user segments
	- data/product sense
	- unsupervised learning
  - write SQL query to accomplish a task
  - throw in a couple of probability questions
- various types of products
  - simple product launch
  - marketing campaign
  - social network
	- Facebook, LinkedIn, Snapchat
	- user may not be independent, A/B test can be challenging
  - 2-sided marketplace
	- Uber, Amazon, Airbnb, eBay
	- supply side, demand side, platform side
  - eCommerce
  - subscription
  - UGC
  - search 
  - ads