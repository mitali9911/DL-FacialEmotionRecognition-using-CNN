import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('fer2013.csv')
#check data shape
print('\n',"Data Shape :",data.shape,'\n')
print("Preview 1st five row of data :",'\n')
print(data.head(5))
print('\n',"Check usage values :",'\n')
print(data.Usage.value_counts())
emotion_map={0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Sad',5:'Surprise',6:'Neutral'}
emotion_counts=data['emotion'].value_counts(sort=False).reset_index()
emotion_counts.columns=['emotion','number']
emotion_counts['emotion']=emotion_counts['emotion'].map(emotion_map)
print('\n',"Emotion Counts :",'\n')
print(emotion_counts)


fig = plt.figure(1, (14, 14))

k = 0
for label in sorted(data.emotion.unique()):
    for j in range(7):
        px = data[data.emotion==label].pixels.iloc[k]
        px = np.array(px.split(' ')).reshape(48, 48).astype('float32')

        k += 1
        ax = plt.subplot(7, 7, k)
        ax.imshow(px, cmap=plt.get_cmap('gray'))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(emotion_map[label])
        plt.tight_layout()

img_array = data.pixels.apply(lambda x: np.array(x.split(' ')).reshape(48, 48, 1).astype('float32'))
img_array = np.stack(img_array, axis=0)
plt.show()

#Plotting bar graph of class distribution
plt.figure(figsize=(6,4))
sns.barplot(x=emotion_counts.emotion,y=emotion_counts.number,alpha=0.8)
plt.title('Class distribution')
plt.ylabel('Number',fontsize=12)
plt.xlabel('Emotions',fontsize=12)
plt.show()

width, height = 48, 48
datapoints = data['pixels'].tolist()

#getting features for training
X = []
for xseq in datapoints:
    xx = [int(xp) for xp in xseq.split(' ')]
    xx = np.asarray(xx).reshape(width, height)
    X.append(xx.astype('float32'))

X = np.asarray(X)
X = np.expand_dims(X, -1)

#getting labels for training
y = pd.get_dummies(data['emotion']).values

#storing them using numpy
np.save('fdataX', X)
np.save('flabels', y)

print("Preprocessing Done")
print("Number of Features: "+str(len(X[0])))
print("Number of Labels: "+ str(len(y[0])))
print("Number of examples in dataset:"+str(len(X)))
print("X,y stored in fdataX.npy and flabels.npy respectively")
