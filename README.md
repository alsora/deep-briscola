# deep-briscola

AC3 algorithm


### Tutorial links 
[Theory](https://jaromiru.com/2017/02/16/lets-make-an-a3c-theory/)

[Practice](https://jaromiru.com/2017/03/26/lets-make-an-a3c-implementation/)



### Results obtained

* 45 000 epochs training using AC3 against Random bot

	Settings:
	* Hand ordered by value   
	* Reward at the end of the single turn (0.01)
	* Reward for winning the match (1)

  ##### Test the trained agent against a random bot 
	<p float="left">
		<img src="https://github.com/alsora/deep-briscola/blob/ac3/Result_45000_epochs_AC3/final%20test%20against%20Random__AC3.png" align="center" width=400 height = 250 />
		<img src="https://github.com/alsora/deep-briscola/blob/ac3/Result_45000_epochs_AC3/final%20test%20against%20Random__RandomAgent.png" align="center" width=400 height = 250/>
	</p>

  ##### Test the trained agent against a hard coded bot 
	<p float="left">
		<img src="https://github.com/alsora/deep-briscola/blob/ac3/Result_45000_epochs_AC3/final%20test%20against%20ai__AC3.png" align="center" width=400 height = 250 />
		<img src="https://github.com/alsora/deep-briscola/blob/ac3/Result_45000_epochs_AC3/final%20test%20against%20ai__AIAgent.png" align="center" width=400 height = 250 />
	</p>

  The agent still doesn't learn.
  
  The problem is very likely to lie in the structure of the enviroment.
  
  
  The algorithm and the hyper-parameters were set for the (Cart-Pole)[https://github.com/openai/gym/wiki/CartPole-v0].
  I 've made some tests with other enviroments of open ai and it doesn't work with them.
  We can try to optimize them.
  
  
  
  
  
