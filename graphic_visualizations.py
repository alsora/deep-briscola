import numpy as __np
import pandas as __pd
import matplotlib.pyplot as __plt

def stats_plotter(agents, points, winners, evaluation_dir,name,epoch):
    N = len(points[0])
    colors = ['green', 'lightblue']

    for i in range(len(agents)):
        __plt.figure(figsize = (10,6))
        res = __plt.hist(points[i], bins=15, edgecolor = 'black', color = colors[i],
            label = agents[i].name + " " + str(i) + " points")
        __plt.title(agents[i].name + " " + str(i) + " won {:.2f}".format( winners[i]/N*100) + "%")
        __plt.vlines(__np.mean(points[i]),
            0,
            max(res[0])/10,
            label = 'Points mean',
            color = 'black',
            linewidth = 3)
        __plt.vlines([__np.mean(points[i]) - __np.std(points[i]),
            __np.mean(points[i]) + __np.std(points[i])],
            ymin=0,
            ymax=max(res[0])/10,
            label = 'Points mean +- std',
            color = 'red',
            linewidth = 3)
        __plt.xlim(0,120); __plt.legend(); 
        __plt.savefig(f"{evaluation_dir}/{name}_{epoch}_{agents[i].name}")
        __plt.close()
        
def eval_visua_for_self_play(average_points_hist,
                             FLAGS,
                             victory_rates_hist,
                             evaluation_dir,
                             epoch,
                             name='fig'):
    '''
    Return the std calculated on all the points of the two player 
    '''
    df  = __np.array(average_points_hist)
    evaluation_num = len(df[:,0,0])
    eval_df = __np.array(average_points_hist)[evaluation_num-1,:,:]
    eval_df = __pd.DataFrame(eval_df.T, columns = ["Agent 0","Agent 1"])
    
    eval_df.plot()
    __plt.hlines([__np.mean(eval_df.values),
                __np.mean(eval_df.values)+__np.std(eval_df.values),
                __np.mean(eval_df.values)-__np.std(eval_df.values)],
               0, len(df[0,0,:]), color = ['green','red','red'], 
               label = 'mean+-std')
    __plt.title(f"""Eval:{evaluation_num}_
              epoch:{evaluation_num*FLAGS.evaluate_every}_
              pctWinBest:{max(victory_rates_hist[-1])}_
              std:{__np.std(eval_df.values).round(2)}""".replace('\n','').replace(' ',''))
    __plt.ylim(0,120)
    __plt.xlabel("Evaluation step")
    __plt.ylabel("Points")
    __plt.legend()
    __plt.savefig(f"{evaluation_dir}/{name}_{epoch}")
    __plt.close()

    return __np.std(eval_df.values).round(2)


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    if iteration == total: 
        print()


def summ_vis_self_play(victory_rates_hist,
                       std_hist,
                       FLAGS):
    df = __np.vstack([__np.array(victory_rates_hist).T,__np.array(std_hist)]).T
    vict_rate = __pd.DataFrame(df, columns = ["Agent 0 win_rate","Agent 1 win_rate", "Std"])
    
    vict_rate['Agent 0 win_rate'].plot(secondary_y=False, 
                                       color = 'lightgreen',
                                       label='Agent 0 (left)')
    vict_rate['Agent 1 win_rate'].plot(secondary_y=False, 
                                       color = 'lightblue',
                                       label='Agent 1 (left)')
    __plt.hlines([__np.mean(vict_rate.values[:,0]),
                __np.mean(vict_rate.values[:,1])],
               0, len(vict_rate)-1, color = ['green','blue'], 
               label = 'means')
    __plt.ylabel('WinRate')
    __plt.legend()
    
    vict_rate.Std.plot(secondary_y=True, label="Std (right)", color = 'red', 
                       alpha = 0.8, linestyle='-.')
    __plt.ylabel('StandardDeviation', rotation=270, labelpad=15)
    __plt.legend()
    __plt.savefig(f"{FLAGS.evaluation_dir}/last")
    __plt.close()










































