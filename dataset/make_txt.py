import os

def mktxt():
  path = "/home/zcf/classifier_training/dataset/"
  idx = 7000
  txt = open(path+"armors.txt","w")
  while os.path.exists(path+"armors/{}.jpg".format(idx)):
    txt.write(path+"armors/{}.jpg\n".format(idx))
    idx+=1
  txt.close()

mktxt()
