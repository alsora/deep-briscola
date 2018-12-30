# deep-briscola

Self play implementation branch



### Results obtained

* 70 000 epochs training using DQRN and self play

	Settings:
	* Hand ordered by value   
	* Reward at the end of the single turn
	* No reward for winning the match

	The agents is not so good at playing against past copies of himself. He wins 55% of matches against them which is not a bad result but it is not a good one either. An other issue is that the variance of winning rate doesn't decline over time. We can see from the graph below that its time series is almost stationary. The agent is not properly learning.
		
	<p float="left">
		<img src="Training 70000 epochs/Graphics/last.png" align="middle" />
	</p>
	The best result obtained against a random bot is 75% of winning rate. This is not a good result since a very simple hard coded bot win 79.5% of matches against the random bot. We can see from the graph below that the trained agent doesn't allow the random bot to win the match with the maximum score achievable (120) but neither he manages to win with that score. We can see from the graphs below that early in the training the extreme scores are contained. 


	<p float="left">
		<img src="Training 70000 epochs/Graphics/againstRandom_58000_QAgent.png" align="center" width=400 height = 250 />
		<img src="Training 70000 epochs/Graphics/againstRandom_58000_Random Agent.png" align="center" width=400 height = 250/>
	</p>
	<p float="left">
		<img src="Training 70000 epochs/Graphics/againstRandom_4000_QAgent.png" align="center" width=400 height = 250 />
		<img src="Training 70000 epochs/Graphics/againstRandom_4000_Random Agent.png" align="center" width=400 height = 250 />
	</p>



	The results suggest that the agent is not learning how to win but how to prevent defeats with extreme points. 

	TODO list:
	* Changing the reward function in order to make clearer that we want the agent to win.
	* Trying new architecture like PPO 
	* Changing the settings

	Current goal:
	* Make an agent better than the simple hard coded bot




