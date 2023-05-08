##  Original Paper Citation:
M. Gaur, A. Alambo, J. P. Sain, U. Kursuncu, K. Thirunarayan, R. Kavuluru, A. Sheth, R. Welton, and J. Pathak, 
“Knowledge-aware assessment of severity of suicide risk for early intervention,” The World Wide Web Conference, 2019. 

## Original Paper Code Repo: 
- https://github.com/jpsain/Suicide-Severity

## Make sure your Python 3 version is at least 3.9.7. The main Python dependencies are the following:
- nltk
- keras
- numpy
- pandas
- gensim
- tensorflow
- sklearn
- DateTime
- statistics
- spacy

## Download dependencies by running the following command:
- pip3 install -r "requirements.txt"

NOTE: If you are using an M1/M2 chip Macbook pro, please download Tensorflow by following this guide: https://developer.apple.com/metal/tensorflow-plugin/

## To run preprocessing code (external_features.py) and create the data/External_Features.csv file (this is optional because we already provided this file), you need to run the following commands:
- python3 -m spacy download en_core_web_sm
- python3 external_features.py

## Get data/numberbatch-en.txt 
Go to https://github.com/commonsense/conceptnet-numberbatch and download numberbatch-en-19.08.txt.gz (English-only). Save the txt.gz file in the "data" folder. Then run the following command to convert it to numberbatch-en.txt: 
- python3 unzipper.py

## Get AFINN-en-165.txt (optional since we provide it already)
Go to https://github.com/fnielsen/afinn/tree/master/afinn/data

## Get LabMT (optional since we provide it already)
Go to https://rdrr.io/cran/qdapDictionaries/man/labMT.html

## Run the evaluation code (suicide_severity_classification.py) with the following command:
- python3 suicide_severity_classification.py ["three" | "four" | "five"]

NOTE: The last argument is "three", "four", or "five" which determine the number of labels:

## Results:
![Screen Shot 2023-05-04 at 1 09 54 PM](https://user-images.githubusercontent.com/12843675/236338458-9bbdfa63-3b14-49c5-872c-6d3a2c3a33ed.png)

![Screen Shot 2023-05-04 at 1 10 04 PM](https://user-images.githubusercontent.com/12843675/236338478-ca5cc800-f7af-49fe-acfb-0c05dac171fa.png)

![Screen Shot 2023-05-04 at 1 10 14 PM](https://user-images.githubusercontent.com/12843675/236338487-810cf2a4-2e60-43ea-b67b-3cc00df13590.png)


