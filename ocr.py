from PIL import Image
import pytesseract
import argparse
import cv2
import os
import numpy as np
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer




# load the example image and convert it to grayscale
image = cv2.imread("example6.png")
#cv2.imshow("image",image)
#cv2.waitKey(0)

#print type(image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("grayscale",gray)
#cv2.waitKey(0)

#print type(gray)

_ ,gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

#cv2.imshow("grayscale_tesh",gray)
#cv2.waitKey(0)
 
# make a check to see if median blurring should be done to remove
# noise
gray = cv2.medianBlur(gray, 3)
#cv2.imshow("grayscale_blurr",gray)
#cv2.waitKey(0)
 
# write the grayscale image to disk as a temporary file so we can
# apply OCR to it
filename = "{}.png".format(os.getpid())
print (filename)
cv2.imwrite(filename, gray)



# load the image as a PIL/Pillow image, apply OCR, and then delete
# the temporary file
text = pytesseract.image_to_string(Image.open(filename))
os.remove(filename)
print(text)
text=text.split()
test=' '.join(text)
x_test=[]
from nltk import tokenize
#nltk.download('punkt')
x_test=tokenize.sent_tokenize(test)
x_size=len(x_test)



###########################################################################################################
######## or or orr or    audio  speech #####################


import speech_recognition as sr
 
# obtain audio from the microphone
r = sr.Recognizer()
with sr.Microphone() as source:
    
     
    print("Say something!")
    audio = r.listen(source)
   
# recognize speech using Google Speech Recognition
    try:
         
        # for testing purposes, we're just using the default API key
        # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
        # instead of `r.recognize_google(audio)`
        x_test = r.recognize_google(audio)
        print("Google Speech Recognition thinks you said " + x_test)
    except sr.UnknownValueError:
         
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
    
    



###########################################################################################################

###########################################################################################################

######## emotion detection
from PIL import Image
import pytesseract
import argparse
import cv2
import os
import numpy as np
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer


###importing data sets
import pandas as pd

dataset=pd.read_csv('texttsv.tsv',delimiter='\t',encoding = "ISO-8859-1")
#dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)


##categorizing input
x=dataset.iloc[:,0]
y=dataset.iloc[:,1]
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label_y=LabelEncoder()
y_cat1=label_y.fit_transform(y)
onehotencoder=OneHotEncoder(categorical_features=[0])
y_cat=onehotencoder.fit_transform(y_cat1.reshape(-1,1)).toarray()
y_cat=y_cat[:,1:13]







##################################################################################
############ ########        train neural networks ############

ps=PorterStemmer()
corpus=[]
for i in range (0,40000):
    
    test_train=re.sub('[^a-zA-z]',' ',dataset['content'][i])
    test_train=test_train.lower()
    test_train=test_train.split()
    test_train=[ps.stem(word) for word in test_train   ]
    test_train=' '.join(test_train)
    corpus.append(test_train)
cv=CountVectorizer(max_features=30000)
x=cv.fit_transform(corpus).toarray()
#y_cat=y_cat[0:2000]
##################################################################################
####### spiltting
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y_cat, test_size = 0.2, random_state = 0)   




import keras
from keras.models import Sequential
#from.keras.models import Dense
from keras.layers.core import Dense
from keras.models import load_model
from keras.layers import Dropout



clasy=Sequential()
clasy.add(Dense(output_dim = 8000, init = 'uniform', activation = 'relu', input_dim=4000 ))
clasy.add(Dropout(rate = 0.3))
#clasy.add(Dense(output_dim = 600, init = 'uniform', activation = 'relu'))
#clasy.add(Dropout(rate = 0.1))
#clasy.add(Dense(output_dim = 3000, init = 'uniform', activation = 'relu'))
#clasy.add(Dropout(rate = 0.3))
clasy.add(Dense(output_dim = 8000, init = 'uniform', activation = 'relu'))
#clasy.add(Dense(output_dim = 4800, init = 'uniform', activation = 'relu'))
clasy.add(Dropout(rate = 0.3))

#clasy.add(Dense(output_dim = 12000, init = 'uniform', activation = 'relu'))

clasy.add(Dense(output_dim = 12, init = 'uniform', activation = 'sigmoid'))
clasy.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#clasy.fit(x,y_cat,batch_size=10,nb_epoch=15)

clasy.fit(X_train,Y_train,batch_size=10,nb_epoch=1)


#clasy=load_model("2000_3000_3000_12.h5")
#clasy.save('5k-4k-5k-12.h5')



y_pred=clasy.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(Y_test, y_pred.round())

#accuracy_score(np.array(Y_test, y_pred), np.ones((2, 2)))


########################### predict ing ###############################################


##nltk processing
ps=PorterStemmer()
combined_all=[0,0,0,0,0,0,0,0,0,0,0,0]

for m in range (0,x_size):
    corpus=[]
    for i in range (0,40000):
        test_train=re.sub('[^a-zA-z]',' ',dataset['content'][i])
        test_train=test_train.lower()
        test_train=test_train.split()
        test_train=[ps.stem(word) for word in test_train if not word in set(stopwords.words('english'))]
        test_train=' '.join(test_train)
        corpus.append(test_train)
    

    
#testing_prepossing
    test_pre=re.sub('[^a-zA-z]',' ',x_test[m])
    test_pre=test_pre.lower()
    test_pre=test_pre.split()
    test_pre=[ps.stem(word) for word in test_pre if not word in set(stopwords.words('english'))]
    test_pre=' '.join(test_pre)
    corpus.append(test_pre)   
 
#tokenizing
    

    cv=CountVectorizer(max_features=2000)
    x=cv.fit_transform(corpus).toarray()
#y=dataset.iloc[:,1].values

    x_pred_val=[]
    x_pred_val=x[40000:40001,0:2000]
    x=x[0:40000,0:2000]
#co rpus=x[0:40000,0:30000]
   
    

##predict
    y_pred=clasy.predict(x_pred_val)
   
   
    prob_pred=[]
    prob_pred1=[]
    for k in range (0,12):
        prob_pred=y_pred[0][k]
        prob_pred=prob_pred * 100
        prob_pred=float("{0:.2f}".format(prob_pred))
        prob_pred1.append(prob_pred)

    pred_val=[]
    pred_val=["boredom","empty","enthusiam","fun","happiness","hate","love","netural","relief","sadness","suprise","worry"]

#### to print it to i console #####

    #combined = np.vstack((pred_val, prob_pred1)).T
    #print('yout text was = ',x_test[m])

    #for a in range (0,12):
        #print(pred_val[a],'=',prob_pred1[a],'%')
        
    for q in range (0,12):
        combined_all[q]=combined_all[q] +  prob_pred1[q]
        
        
#################  printing to file #############
######## delete outputfilr if existed #########

    with open('outputfile.txt', 'a') as f:
        print(x_test[m],'\n',file=f)
        for r in range (0,12): 
            print(pred_val[r],'=',prob_pred1[r],'%', file=f)
        print('\n',file=f)
    f.close()


with open('outputfile.txt', 'a') as f:
    print('over all emotion','\n',file=f)   
    for w in range (0,12):  
        combined_all[w]=combined_all[w]/x_size
        combined_all[w]=float("{0:.2f}".format(  combined_all[w]))

        print(pred_val[w],'=',combined_all[w],'%', file=f)
f.close()

##################################################################################



