import numpy as __np
import pandas as __pd
import matplotlib.pyplot as __plt


def stats_plotter(agents, points, total_wins, output_prefix = ''):
    num_evaluations = len(points[0])
    colors = ['green', 'lightblue']

    for i in range(len(agents)):
        __plt.figure(figsize = (10,6))
        res = __plt.hist(points[i], bins=15, edgecolor = 'black', color = colors[i],
            label = agents[i].name + " " + str(i) + " points")
        __plt.title(agents[i].name + " " + str(i) + " won {:.2%}".format(total_wins[i]/num_evaluations))
        __plt.vlines(__np.mean(points[i]),
            ymin=0,
            ymax=max(res[0])/10,
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
        __plt.xlim(0,120)
        __plt.legend()

        if output_prefix:
            # if an output path is specified, save the plot
            __plt.savefig(f"{output_prefix}_{agents[i].name}")
        else:
            # else show it
            __plt.show()
        __plt.close()


def evaluate_summary(winners, points, agents, evaluation_dir):
    fig, ax = __plt.subplots(figsize=(12,8))
    __plt.bar([0,1], winners, edgecolor = 'blue', color = 'yellow')
    __plt.ylim((min(winners)-5,max(winners)+5))          
    __plt.xticks([0,1], [ag.name for ag in agents])
    __plt.ylabel("# of victories")
    __plt.text(0.25, 0.1, f'STD points: {round(__np.std(points[0]),2)}', {"size" : 18},
                horizontalalignment='center', color = 'black',
                verticalalignment='center', transform=ax.transAxes,
                bbox=dict(facecolor='cyan', alpha=0.4))                
    __plt.text(0.75, 0.1,  f'STD points: {round(__np.std(points[1]),2)}', {"size" : 18}, 
                horizontalalignment='center', color = 'black',
                verticalalignment='center', transform=ax.transAxes,
                bbox=dict(facecolor='cyan', alpha=0.4))                
    __plt.text(0.25, 0.2, f'MEAN points: {round(__np.mean(points[0]),2)}', {"size" : 18}, 
                horizontalalignment='center', color = 'black',
                verticalalignment='center', transform=ax.transAxes,
                bbox=dict(facecolor='cyan', alpha=0.4))                
    __plt.text(0.75, 0.2,  f'MEAN points: {round(__np.mean(points[1]),2)}', {"size" : 18}, 
                horizontalalignment='center', color = 'black',
                verticalalignment='center', transform=ax.transAxes,
                bbox=dict(facecolor='cyan', alpha=0.4))       
    __plt.title(evaluation_dir[evaluation_dir.find('/')+1:])         
    __plt.savefig(evaluation_dir)
    __plt.close()

def training_summary(x, vict_hist, point_hist, labels, FLAGS, evaluation_dir):

    fig, ax = __plt.subplots(2,1, figsize=(12,8), sharex=True)
    fig.subplots_adjust(hspace=0)
    ax[0].set_title(f"Summary of {FLAGS.num_epochs} epochs", {'size' : 21})

    y1 = __np.asarray(vict_hist).T[0]/FLAGS.num_evaluations
    y2 = __np.asarray(vict_hist).T[1]/FLAGS.num_evaluations
    ax[0].plot(x, y1, linestyle ='--', label = labels[0], color = 'green')
    ax[0].plot(x, y2, linestyle ='--', label = labels[0], color = 'red')
    ax[0].set_ylabel('Victory %', {'size' : 15})
    ax[0].hlines(__np.mean(y1),x[0],x[-1], alpha = 0.2, color = 'green')
    ax[0].hlines(__np.mean(y2),x[0],x[-1], alpha = 0.2, color = 'red')
    ax[0].legend()
    
    y1 = __np.mean(__np.asarray(point_hist)[:,0,:],1)
    y2 = __np.mean(__np.asarray(point_hist)[:,1,:],1)
    y3 = __np.std(__np.asarray(point_hist)[:,0,:],1) 
    y4 = __np.std(__np.asarray(point_hist)[:,1,:],1) 
    
    ax[1].plot(x, y1, linestyle ='--', label = labels[0], color = 'green')
    ax[1].plot(x, y2, linestyle ='--', label = labels[0], color = 'red')
    ax[1].scatter(x, y1, y3, label = labels[0]+' std', color = 'green')
    ax[1].scatter(x, y2, y4, label = labels[0]+' std', color = 'red')
    ax[1].set_ylabel('Mean point obtained', {'size' : 15})
    ax[1].set_xlabel('Epoch', {'size' : 15})
    ax[1].hlines(__np.mean(y1),x[0],x[-1], alpha = 0.2, color = 'green')
    ax[1].hlines(__np.mean(y2),x[0],x[-1], alpha = 0.2, color = 'red')
    ax[1].legend()
    
    __plt.savefig(evaluation_dir)
    __plt.close()





def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = ' '):
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
    df = __np.vstack([__np.array(victory_rates_hist).T,__np.array(std_hist)]).T / FLAGS.num_evaluations
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

