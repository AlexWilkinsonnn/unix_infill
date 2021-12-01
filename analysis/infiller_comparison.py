import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import pickle


ML_dir = "/unix/dune/awilkinson/infill_work/analysis/data/ML_infill"
nonML_dir = "/unix/dune/awilkinson/infill_work/analysis/data/nonML_infill"

if False:
    cnt, acc_cnt = 0, 0
    in2, in10, in20, in40 = 0, 0, 0, 0
    in2percent, in5percent, in10percent, in20percent, in50percent, acc = 0, 0, 0, 0, 0, 0
    for idx, filename in enumerate(os.listdir(ML_dir)):
        with open(os.path.join(ML_dir, filename), "r") as f:
            arr = np.load(f)
            overlay = arr[0]
            true = arr[1]
        
        # plt.imshow(overlay.T, cmap='coolwarm', vmin=-100, vmax = 100, aspect='auto')
        # plt.show()

        # plt.imshow(true.T, cmap='coolwarm', vmin=-100, vmax = 100, aspect='auto')
        # plt.show()
        # print('\n')
        for x, wire in enumerate(overlay):
            if np.any(wire < 0):
                # print(x)
                for y, adc in enumerate(wire):
                    if np.abs(true[x,y]) > 10:
                        acc_cnt += 1
                        if np.abs(adc) > 10:
                            acc += 1
                    
                    elif np.abs(true[x,y]) < 10:
                        acc_cnt += 1
                        if np.abs(adc) < 10:
                            acc += 1                   

                    if true[x,y] != 0:
                        cnt += 1
                        # print(adc/true[x,y])
                        # print('{} - {}'.format(adc, true[x,y]))

                        if np.abs(true[x,y]) > 10:
                            acc_cnt += 1
                            if np.abs(adc) > 10:
                                acc += 1

                        if np.abs(adc - true[x,y]) < 2:
                            in2 += 1
                            in10 += 1
                            in20 += 1
                            in40 += 1

                        elif np.abs(adc - true[x,y]) < 10:
                            in10 += 1
                            in20 += 1
                            in40 += 1

                        elif np.abs(adc - true[x,y]) < 20:
                            in20 += 1
                            in40 += 1

                        elif np.abs(adc - true[x,y]) < 40:
                            in40 += 1

                        if 0.98 < np.abs(adc/true[x,y]) < 1.02:
                            in2percent += 1
                            in5percent += 1
                            in10percent += 1
                            in20percent += 1
                            in50percent += 1

                        elif 0.95 < np.abs(adc/true[x,y]) < 1.05:
                            in5percent += 1
                            in10percent += 1
                            in20percent += 1
                            in50percent += 1

                        elif 0.9 < np.abs(adc/true[x,y]) < 1.1:
                            in10percent += 1
                            in20percent += 1
                            in50percent += 1

                        elif 0.8 < np.abs(adc/true[x,y]) < 1.2:
                            in20percent += 1    
                            in50percent += 1                    

                        elif 0.5 < np.abs(adc/true[x,y]) < 1.5:
                            in50percent += 1

    print('ML:')
    print('in 2: {:2.2%}'.format(float(in2)/float(cnt)))
    print('in 10: {:2.2%}'.format(float(in10)/float(cnt)))
    print('in 20: {:2.2%}'.format(float(in20)/float(cnt)))
    print('in 40: {:2.2%} \n'.format(float(in40)/float(cnt)))
    print('in 2 percent: {:2.2%}'.format(float(in2percent)/float(cnt)))
    print('in 5 percent: {:2.2%}'.format(float(in5percent)/float(cnt)))
    print('in 10 percent: {:2.2%}'.format(float(in10percent)/float(cnt)))
    print('in 20 percent: {:2.2%}'.format(float(in20percent)/float(cnt)))
    print('in 50 percent: {:2.2%} \n'.format(float(in50percent)/float(cnt)))
    print('Binary accuracy: {:2.2%} \n'.format(float(acc)/float(acc_cnt)))

    cnt, acc_cnt = 0, 0
    in2, in10, in20, in40 = 0, 0, 0, 0
    in2percent, in5percent, in10percent, in20percent, in50percent, acc = 0, 0, 0, 0, 0, 0
    for idx, filename in enumerate(os.listdir(nonML_dir)):
        with open(os.path.join(nonML_dir, filename), "r") as f:
            arr = np.load(f)
            overlay = arr[0]
            true = arr[1]

        for x, wire in enumerate(overlay):
            if np.any(wire < 0):
                for y, adc in enumerate(wire):
                    if np.abs(true[x,y]) > 10:
                        acc_cnt += 1
                        if np.abs(adc*0.98) > 10:
                            acc += 1
                    
                    elif np.abs(true[x,y]) < 10:
                        acc_cnt += 1
                        if np.abs(adc*0.98) < 10:
                            acc += 1     

                    if true[x,y] != 0:
                        cnt += 1

                        if np.abs(adc*0.98 - true[x,y]) < 2:
                            in2 += 1
                            in10 += 1
                            in20 += 1
                            in40 += 1

                        elif np.abs(adc*0.98 - true[x,y]) < 10:
                            in10 += 1
                            in20 += 1
                            in40 += 1

                        elif np.abs(adc*0.98 - true[x,y]) < 20:
                            in20 += 1
                            in40 += 1

                        elif np.abs(adc*0.98 - true[x,y]) < 40:
                            in40 += 1

                        if 0.98 < np.abs(adc*0.98/true[x,y]) < 1.02:
                            in2percent += 1
                            in5percent += 1
                            in10percent += 1
                            in20percent += 1
                            in50percent += 1

                        elif 0.95 < np.abs(adc*0.98/true[x,y]) < 1.05:
                            in5percent += 1
                            in10percent += 1
                            in20percent += 1
                            in50percent += 1

                        elif 0.9 < np.abs(adc*0.98/true[x,y]) < 1.1:
                            in10percent += 1
                            in20percent += 1
                            in50percent += 1

                        elif 0.8 < np.abs(adc*0.98/true[x,y]) < 1.2:
                            in20percent += 1    
                            in50percent += 1                    

                        elif 0.5 < np.abs(adc*0.98/true[x,y]) < 1.5:
                            in50percent += 1

    print('nonML:')
    print('in 2: {:2.2%}'.format(float(in2)/float(cnt)))
    print('in 10: {:2.2%}'.format(float(in10)/float(cnt)))
    print('in 20: {:2.2%}'.format(float(in20)/float(cnt)))
    print('in 40: {:2.2%} \n'.format(float(in40)/float(cnt)))
    print('in 2 percent: {:2.2%}'.format(float(in2percent)/float(cnt)))
    print('in 5 percent: {:2.2%}'.format(float(in5percent)/float(cnt)))
    print('in 10 percent: {:2.2%}'.format(float(in10percent)/float(cnt)))
    print('in 20 percent: {:2.2%}'.format(float(in20percent)/float(cnt)))
    print('in 50 percent: {:2.2%} \n'.format(float(in50percent)/float(cnt)))
    print('Binary accuracy: {:2.2%} \n'.format(float(acc)/float(acc_cnt)))


if False:
    tick = range(1, 6001) 
    tick_adc_ML = [0]*6000
    tick_adc_true = [0]*6000
    for idx, filename in enumerate(os.listdir(ML_dir)):
        with open(os.path.join(ML_dir, filename), "r") as f:
            arr = np.load(f)
            overlay = arr[0]
            true = arr[1]
        
        for y, column in enumerate(overlay.T):
            if np.any(true[:,y] == 0): #THIS MAY BE WRONG THINK ABOUT IT. IT IS, THIS WILL ALWAYS BE TRUE. SAYING IS THE TICK SEES A DEAD CHANNEL TAKE ALL THE ADC VALS OF THAT TICK AS DEAD
                for x, adc in enumerate(column):
                    tick_adc_ML[y] += np.abs(adc)
                    tick_adc_true[y] += np.abs(true[x,y])

        if idx % 50 == 0:
            print(idx)

    tick_adc_nonML = [0]*6000
    for idx, filename in enumerate(os.listdir(nonML_dir)):
        with open(os.path.join(nonML_dir, filename), "r") as f:
            arr = np.load(f)
            overlay = arr[0]
        
        for y, column in enumerate(overlay.T):
            if np.any(true[:,y] == 0):
                for x, adc in enumerate(column):
                    tick_adc_nonML[y] += np.abs(adc)

        if idx % 50 == 0:
            print(idx)

    adc_hist_data = [tick, tick_adc_true, tick_adc_ML, tick_adc_nonML]
    with open('adc_hist_data.pkl', 'w') as f:
        pickle.dump(adc_hist_data, f)

    plt.hist(tick, bins=len(tick), weights=tick_adc_true, histtype='step', label='true', linewidth=0.5)
    plt.hist(tick, bins=len(tick), weights=tick_adc_ML, histtype='step', label='ML', linewidth=0.5)
    plt.hist(tick, bins=len(tick), weights=tick_adc_nonML, histtype='step', label='averaging', linewidth=0.5)
    plt.legend()
    plt.show()

if False:
    with open('adc_hist_data.pkl', 'r') as f:
        tick, tick_adc_true, tick_adc_ML, tick_adc_nonML = pickle.load(f)

    tick_adc_nonML_new = [ i * 0.98 for i in tick_adc_nonML ]
    plt.rc('font', family='serif')
    plt.hist(tick, bins=int(len(tick)/100), weights=tick_adc_true, histtype='step', label='True', linewidth=0.7)
    plt.hist(tick, bins=int(len(tick)/100), weights=tick_adc_ML, histtype='step', label='Network', linewidth=0.7, alpha=1)
    plt.hist(tick, bins=int(len(tick)/100), weights=tick_adc_nonML_new, histtype='step', label='Averaging', linewidth=0.7, alpha=1)
    plt.xlim(2000,4000)
    plt.xlabel('time tick', fontsize=20)
    plt.ylabel('total ADC in dead regions', fontsize=20)
    plt.ylim(bottom=30000000, top=42000000)
    # plt.yscale('log')
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    tx = ax.yaxis.get_offset_text()
    tx.set_fontsize(20)
    handles, labels = ax.get_legend_handles_labels()
    new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]
    plt.legend(handles=new_handles, labels=labels, prop={'size': 20})
    plt.show()


ML_dir = "/unix/dune/awilkinson/infill_work/analysis/data/special/ML_infill"
nonML_dir = "/unix/dune/awilkinson/infill_work/analysis/data/special/nonML_infill"

tick = range(1, 6001) 
tick_adc_ML = [0]*6000
tick_adc_true = [0]*6000
for idx, filename in enumerate(os.listdir(ML_dir)):
    if filename.endswith("Event873_TPC2.npy"):
        with open(os.path.join(ML_dir, filename), "r") as f:
            arr = np.load(f)
            overlay = arr[0]
            true = arr[1]
            
        for y, column in enumerate(overlay.T):
            tick_adc_ML[y] += np.abs(overlay[369,y])
            tick_adc_true[y] += np.abs(true[369,y])

tick_adc_nonML = [0]*6000
for idx, filename in enumerate(os.listdir(nonML_dir)):
    if filename.endswith("Event873_TPC2.npy"):
        with open(os.path.join(nonML_dir, filename), "r") as f:
            arr = np.load(f)
            overlay = arr[0]
    

        for y, column in enumerate(overlay.T):
            tick_adc_nonML[y] += np.abs(overlay[369,y])*0.98

# plt.imshow(overlay.T, aspect='auto', cmap='coolwarm', vmin=-20, vmax=20)
# plt.show()
 
tick_adc_nonML_new = [ i * 0.98 for i in tick_adc_nonML ]
plt.rc('font', family='serif')
plt.hist(tick, bins=len(tick), weights=tick_adc_true, histtype='step', label='True', linewidth=0.7)
plt.hist(tick, bins=len(tick), weights=tick_adc_ML, histtype='step', label='Network', linewidth=0.7)
plt.hist(tick, bins=len(tick), weights=tick_adc_nonML_new, histtype='step', label='Averaging', linewidth=0.7)
plt.xlim(1,6000)
plt.xlabel('time tick', fontsize=20)
plt.ylabel('ADC', fontsize=20)
# plt.ylim(bottom=30000000, top=42000000)
# plt.yscale('log')
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=16)
ax.tick_params(axis='both', which='minor', labelsize=16)
tx = ax.yaxis.get_offset_text()
tx.set_fontsize(20)
handles, labels = ax.get_legend_handles_labels()
new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]
plt.legend(handles=new_handles, labels=labels, prop={'size': 20})
plt.show()
plt.legend()

if False:
    tick = range(1, 6001) 
    tick_adc_ML = [0]*6000
    tick_adc_true = [0]*6000
    for idx, filename in enumerate(os.listdir(ML_dir)):
        if filename.endswith("Event842_TPC6.npy"):
            with open(os.path.join(ML_dir, filename), "r") as f:
                arr = np.load(f)
                overlay = arr[0]
                true = arr[1]
                
            for y, column in enumerate(overlay.T):
                tick_adc_ML[y] += np.abs(overlay[29,y])
                tick_adc_true[y] += np.abs(true[29,y])

    tick_adc_nonML = [0]*6000
    for idx, filename in enumerate(os.listdir(nonML_dir)):
        if filename.endswith("Event842_TPC6.npy"):
            with open(os.path.join(nonML_dir, filename), "r") as f:
                arr = np.load(f)
                overlay = arr[0]
        

            for y, column in enumerate(overlay.T):
                tick_adc_nonML[y] += np.abs(overlay[29,y])*0.98

    # plt.imshow(overlay.T, aspect='auto', cmap='coolwarm', vmin=-20, vmax=20)
    # plt.show()
    
    tick_adc_nonML_new = [ i * 0.98 for i in tick_adc_nonML ]
    plt.rc('font', family='serif')
    plt.hist(tick, bins=len(tick), weights=tick_adc_true, histtype='step', label='True', linewidth=0.7)
    plt.hist(tick, bins=len(tick), weights=tick_adc_ML, histtype='step', label='Network', linewidth=0.7)
    plt.hist(tick, bins=len(tick), weights=tick_adc_nonML_new, histtype='step', label='Averaging', linewidth=0.7)
    plt.xlim(1,6000)
    plt.xlabel('time tick', fontsize=20)
    plt.ylabel('total ADC in dead regions', fontsize=20)
    # plt.ylim(bottom=30000000, top=42000000)
    # plt.yscale('log')
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    tx = ax.yaxis.get_offset_text()
    tx.set_fontsize(20)
    handles, labels = ax.get_legend_handles_labels()
    new_handles = [Line2D([], [], c=h.get_edgecolor()) for h in handles]
    plt.legend(handles=new_handles, labels=labels, prop={'size': 20})
    plt.show()
    plt.legend()