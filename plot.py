import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='train and test')
    
    parser.add_argument('--dataset'   , default = 'keti' , type = str,
                        choices=['motion', 'seizure', 'wifi', 'keti', 'PAMAP2'])
    parser.add_argument('--model'     , default ='ResNet', type = str,
                        choices=['ResNet', 'MaCNN', 'MaDNN'])
    args = parser.parse_args()
    return args

def get_data(lines):
    epsilon = []
    data    = []
    for line in lines:
        epsilon.append(float(line.split(',')[0].split(':')[1]))
        data.append(float(line.split(',')[1].split(':')[1]))
    return epsilon, data

args = parse_args()

TNR = fm.FontProperties(fname='../Fonts/times.ttf')
font_name = fm.FontProperties(fname='../Fonts/times.ttf').get_name()

file_name = args.model
dataset   = args.dataset


f     = open(f'./results/{file_name}/{dataset}_False.txt', encoding='utf-8')
lines = f.readlines()
data  = []


epsilon, d  = get_data(lines)
data.append(d)

f     = open(f'./results/{file_name}/{dataset}_True.txt', encoding='utf-8')
lines = f.readlines()
epsilon, d  = get_data(lines)
data.append(d)

base_line = [data[0][0] for idx in range(len(data[0]))]

plt.plot(epsilon, data[0]  , label = 'FSGM Attack')
plt.plot(epsilon, data[1]  , label = 'FSGM Aid'   )
plt.plot(epsilon, base_line, label = 'Baseline'   )
plt.grid()
plt.xlabel('Epsilon'               , fontproperties = TNR, fontsize=15)
plt.ylabel('Accuracy (%)'          , fontproperties = TNR, fontsize=15)
plt.title( f'{file_name} : {dataset}'    , fontproperties = TNR, fontsize=15)
plt.xticks(fontproperties = TNR    , fontsize=15)
plt.yticks(fontproperties = TNR    , fontsize=15)
plt.legend(loc = 'lower left', prop={'family':font_name, 'size':10})
plt.savefig(f'./figure/{file_name}_{dataset}.png')
#plt.show()