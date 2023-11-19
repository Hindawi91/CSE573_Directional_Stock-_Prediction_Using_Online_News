# Directional Stock Prediction using online news
This is project built as a requirement for course CSE 573 Semantic Web Mining at ASU. The aim is to build directional stock prediction system which predict stock market price movement using online news. It is built using BERT and Bidirectional LSTM. 

## Dataset
The model is trained on a news corpus consisting over 80000 articles related to Amazon and Apple news. The dataset was collected between January 2018 to February 2019. The dataset also contains stock chart price of AMZN and AAPL stocks. We build a labelled dataset utilizing change in stock price at different intervals of time.

## How to setup the project
1. Clone the project.
2. Open a terminal, and change directory to CODE folder. 
3. To download data, In the command line do the following:
    <ul>
      <li> run "cd ./data" to move to the data folder </li>
      <li> run bash get_data.sh </li>
      <li> Unzip the data.zip and all the sub-folders </li>
      <li> Once you're done, you must have two sub-folders inside "data" which are:
             <ul><li> "CHARTS" (which has csv files) </li>
            <li> "News" (which has subfolders. Inside each subfolder is a lot of JSON files)</li></ul>
      </li>
    </ul>
4. Pre-process Data:
    <ul>
        <li> run "cd 1_data_pre_processing" to move to the folder </li>
        <li> go through the "1_Process_News.ipynb" Jupiter file to process the "news" data </li>
        <li> go through the "2_Process_Charts.ipynb" Jupiter file to process the "CHARTS" data </li>
        <li> Once you're done, you must have .pkl files generated for the next step
        </li>
      </ul>

5. Features Extraction:
     <ul>
          <li> run "cd 2_features_extraction" to move to the folder </li>
          <li> run the .py files to generate features for both "Amzn" and "Aapl" using different features everytime ("bert","word2vec","tf-idf") </li>
          <li> Once you're done, you must have 6 ".pkl" features files generated for the next step </li>
        </ul>
6. Model Training and Evaluation:
     <ul>
          <li> run "cd 3_train_models" to move to the folder </li>
          <li> run the .py files to train 6 different models for both "Amzn" and "Aapl" using different features everytime</li>
          <li>The code trains the following models for each experiment: </li>
       <ul> <li> Logistic Regression </li>
            <li> Random Forrests </li>
         <li> Adaboost </li>
         <li> Support Vector Machines </li>
         <li> Knn </li>
         <li> Voting Ensemble of the previous 5 models </li>
       </ul>
       <li> Once you're done, you must have initial results generated in the form of ".csv" files and confusion matrices ".png" files in the "Results" folder</li>
        </ul>
        
7. Generate Results:
        <ul>
          <li> run "cd Results" to move to the folder </li>
          <li> run the two ".ipynb" files to generate results and determine the best news outlets </li>
          <li> results will be generated as figures in the same folder </li>
        </ul>
