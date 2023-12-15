#import all dependencies
#import data and code
#run the train.py file only, on training data only
#set paths in train.py file
#test this out without data
#data extraction is also required
#

FROM python:3
# RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install gdown
RUN pip3 install pillow json os
RUN pip3 install opencv-python scikit-learn numpy pillow matplotlib tqdm zipfile
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# /media/E/FYP_everything/code/RCNN/test/train.py
# # whatever like
WORKDIR /media/table_tennis_FYP 
# directory to save checkpoints
RUN mkdir checkpoints

# https://drive.google.com/file/d/1fsd3-I42bqgrKeIICixyebi-r-3gc1sQ/view?usp=share_link
# link to sample zip file
# https://drive.google.com/drive/folders/1HpQihZcBNzx4vzfeM1ld88SIkufwLa61?usp=share_link

# link to dataset path
RUN gdown 1KfDG-RT7CkDuJ5r06BmSbUvyJ6OZDbzy
# link to code file
RUN gdown --folder 1eMrgFrMBh9Mi3YZab6mq4W4XhYmntl2M
COPY . .
CMD ["python3","./code/dataloader.py","./code/train_.py"]
# CMD ["ls"]

