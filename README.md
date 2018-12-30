# deep-briscola

Self play implementation branch



### Results obtained

* 70 000 epochs training using DQRN and self play

	* The agents is not so good at playing against past copies of himself. He wins 55% of matches against them which is not a bad result but it is not a good one either.
	* An other issue is that the variance of winning rate doesn't decline over time. We can see from the graph below that its time series is almost stationary. The agent is not properly learning.


	Solarized dark             |  
	:-------------------------:|
	![](https://github.com/alsora/deep-briscola/blob/self_play/Training\ 70000\ epochs/Graphics/last.png)  |


		

	* The best result obtained against a random bot is 75% of winning rate. 
	* This is not a good result since a very simple hard coded bot win 79.5% of matches against the random bot. We can see from the graph below that the trained agent doesn't allow the random bot to win the match with the maximum score achievable (120) but neither he manages to win with that score.


		<p float="left">
			<img src="Training 70000 epochs/Graphics/againstRandom_58000_QAgent.png" align="center" width=400 height = 250 />
			<img src="Training 70000 epochs/Graphics/againstRandom_58000_Random Agent.png" align="center" width=400 height = 250/>
		</p>
		<p float="left">
			<img src="Training 70000 epochs/Graphics/againstRandom_4000_QAgent.png" align="center" width=400 height = 250 />
			<img src="Training 70000 epochs/Graphics/againstRandom_4000_Random Agent.png" align="center" width=400 height = 250 />
		</p>






