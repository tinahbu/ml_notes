1. Like video: 添加给大家看朋友like的video， 如何评估这个feature 值不值得加 

2. Close friend notification: 打算给用户的发text notification 告诉他们close friends 的Update, 如何评估这个feature 值不值得加, 注意是close

3. Prime Now, 就是prime的升级版，两小时送货上门，假设你提出来这个新的prime now的proposal，一周后给老板和其他部门老大分析这个事情可不可行，说说你需要哪些variable来进行分析，你怎么跟老板present

4. 1Point3Acres Smart Admit: 一亩三分地有很多录取数据：Build a data product “Smart Admit”: given an applicant’s background, predict chance of admission into a program

5. How do you explain this graph? (week 5 is this week)

   ![q5](/Users/tina.bu/Google Drive/ML-Interview/q5.png)

6. Dashboard for Careers: Careers team’s mission is to help job seekers find their jobs (search, recommendation). For a weekly report of the team, what metrics are you going to present?

7. Feed diversity: Feed are overly optimized that users start to complain about feed having low diversity. How do you define diversity? (need to get a formula here) How do you measure and monitor to see whether diversity increased/decreased over time? How to decide on whether this is an issue to worry about?

8. Newsletter: The business sends two newsletters a week: one on Mondays for sales related contents, one on Wednesdays for explorational newsletters. Create a dashboard to monitor newsletter performance over time

9. Revenue goes flat yoy, what data would you use to explain why

10. AB test, slow roll out: How do you explain the result

![Screen Shot 2017-09-29 at 11.37.06 AM](/Users/tina.bu/Google Drive/ML-Interview/Screen Shot 2017-09-29 at 11.37.06 AM.png)

1. Likes/DAU grows +10% year/ year. Explain why
2. Choosing the right analysis method:
   - Know past shopping behavior, predict whether or not user will buy a new item
   - See an image, tell whether it’s a cat/dog/human/etc
   - Identify Platinum/Gold/Silver users
   - Two new products, which earns more revenue?
   - Given 10 slot machines with different payouts, get the best ROI for your $$$?
   - Know past shopping behavior, predict price user is willing to pay for a new item
   - We currently provide product X in country Y1 and earn $Z; what if we provide X in countries Y2, Y3, Y4?
   - Is conversion rate consistent across Android/iOS/Web only users?

13. Interpret AB testing result, Treatment effect for each group as below: 

    

    - Trt1: -5, CI: (-7.5, -2.5)

    - Trt2: -15, CI: (-17, -13)

    - Trt3: -12, CI: (-28, -4)

      

    How to interpret C.I.? Which treatment to choose? Increase test power / accuracy?

14. Map: Where did I park? User开车出行,然后下车后可能还需要再走一段距离. 希望Map能告诉user自己的车的位置. 怎么实现?

15. Friend Recommendation: How do you test a new friend recommendation algorithm?

16. Given data

    - User1        user2        #msg
    - 001                002                2
    - 001                003                4
    - ...
    - 003                001                4

    What is the distribution of #msgs? Who is the top exchanger for each user?

17. Roll out experiment, What’s the expected waiting time to receive treatment? 1K users

    - Day 1: 10 users out of 1000 -> trt
    - Day 2: 10 users out of 990 -> trt
    - Day 3: 10 users out of 980 -> trt
    - …
    - Day 100: 10 users -> trt

18. Roll out experiment: What’s the expected waiting time for the user to first enroll in the treatment? 1K users

    

    - Day 1: 10 users out of 1000 -> trt

    - Day 2: 10 users out of 1000 -> trt

    - Day 3: 10 users out of 1000 -> trt

    - …

    - Day 1000: 10 users -> trt

      

19. Optimal Dice Roll, you have a 6-sided fair dice, your reward is value of very last roll. You have a max of 3 rolls. What is optimal stopping strategy and expected reward amount.

20. X ~ N(0, sigma^2), Y ~ N(0, sigma^2), X & Y independent. P(X > 2Y)=? P(X > |2Y|)=?

21. Coding: find sqrt(x)

22. Coding: 某个小动物可以一次跳上一个台阶或者两个台阶，问爬上第N级台阶可以有几种方法，写个简单的code出来。

23. Coding: Random number generation: Give you a dice with 7 sides, how do you generate random numbers between 1 and 10 with equal probability?

24. Coding: Random sampling on streaming data: A stream of values x1,x2,x3 …; may or may not end. How do you sample k random items from the stream (without replacement), so that after seeing n data points, the probability that any individual data point is in the sample is k/n, with only one pass through the stream, and uses storage proportional to k (not the total, n, possibly infinite, size of the stream).