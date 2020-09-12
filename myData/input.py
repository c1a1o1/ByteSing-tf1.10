#coding:utf-8
import pinyin as py
import numpy as np
import textgrid
phonemes = ['b', 'p', 'f', 'm', \
            'd', 't', 'n', 'l', \
            'g', 'k', 'h', \
            'j', 'q', 'x', \
            'zh', 'ch', 'sh', 'r', \
            'z', 'c', 's',\
            'a',  'ai', 'ao',  'an',  'ang', \
            'o',  'ou', 'ong', \
            'e',  'ei', 'en',  'eng', 'er', 'ev', \
            'i',  'ix', 'iii', \
            'ia', 'iao','ian', 'iang','ie', \
            'in', 'ing','io',  'iou', 'iong', \
            'u',  'ua', 'uo',  'uai', 'uei', \
            'uan','uen','uang','ueng', \
            'v',  've', 'van', 'vn', \
            'ng', 'mm', 'nn',\
            'rr', 'sp']
initialOrFinal = ['0', '1', '2', '3']
pitch = ['0','50','51','52','53','54','55','56','57','58','59',\
         '60','61','62','63','64','65','66','67','68','69',\
         '70','71','72','73','74','75','76','77','78','79','80']
frameLength = 20 # 设置帧长为20ms帧移为5ms
frameHop = 5

# one-hot编码
def onehotEncoding(instance, class1):
    temp = [0] * len(class1)
    temp[class1.index(instance)] = 1
    return temp


durationInput = []
# 时长模型输入（以音素为单位）
with open('./myData/train.txt', encoding='utf-8') as f:
    metadatas = [line.strip().split('|') for line in f]
# 音素one-hot编码
# 音素类型（声母、韵母、零声母）one-hot编码
# 拼接[音素，音素类型，所属音符理论时长]
for line in metadatas:
    phoneme = onehotEncoding(line[-1], phonemes)
    soundType = onehotEncoding(line[-2], initialOrFinal)
    time = [float(line[0])]
    durationInput.append(phoneme+soundType+time)
# 时长模型预期输出：每个音素的开始、结束时间（训练时由人工标注信息获得），最开始一般会比乐谱多一个sil
with open('./myData/001.interval', encoding='utf-8') as f:
    i = 0
    j = 0
    durationOutput = []
    for line in f:
        if j != 12:
            j = j+1
            continue
        line = line.split('\n')[0]
        if i == 0:
            startTime = float(line)
            i = i+1
        elif i == 1:
            endTime = float(line)
            i = i+1
        else:
            i = 0
            if j == 12:
                durationOutput.append([startTime, endTime, line.split('"')[1]])
            else:
                if durationOutput[-1][2] != line.split('"')[1]:
                    durationOutput.append([startTime, endTime, line.split('"')[1]])
                else:
                    durationOutput[-1][1] = endTime
            



# 声学模型输入（以帧为单位），训练时由人工标注获得，reference时由时长模型的结果来获得

# 各帧的音素one-hot编码
# 此帧所属音符的音调（C4,G3）的one-hot编码
# 此帧位置P0：1.在音素中的前比例，2.在音素中的后比例，3.音素在整个语音中的比例，都为[0,1]间的浮点数
# P0的embedding
# 拼接[音素，音调，位置embedding]

