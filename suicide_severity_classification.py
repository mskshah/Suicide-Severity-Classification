"""
Original Paper Citation:
M. Gaur, A. Alambo, J. P. Sain, U. Kursuncu, K. Thirunarayan, R. Kavuluru, A. Sheth, R. Welton, and J. Pathak, 
“Knowledge-aware assessment of severity of suicide risk for early intervention,” The World Wide Web Conference, 2019. 

Original Paper Code Repo: https://github.com/jpsain/Suicide-Severity
"""
import time
import sys
from enum import Enum
import csv
import string
import nltk
from nltk import word_tokenize
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from keras.utils.np_utils import to_categorical
import datetime
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPool2D
from keras.layers import Conv2D
from keras.layers import Concatenate
from keras.optimizers import adam
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

nltk.download('punkt')

class ExperimentType(Enum):
    FIVE_LABEL = 1
    FOUR_LABEL = 2
    THREE_PLUS_ONE_LABEL = 3

class CNNConfiguration:
    no_filters = 100
    kernels = [3, 4, 5]
    channel = 1
    c_stride = None
    pad = 'same'
    ip_shape = None
    c_activ = 'relu'
    drop_rate = 0.3
    dense_1_unit = 128
    dense_2_unit = 128
    dense_activ = 'relu'
    op_unit = None
    op_activ = 'softmax'
    l_rate = 0.001
    loss = 'categorical_crossentropy'
    batch = 4
    epoch = 50
    verbose = 1

    @classmethod
    def update(cls, c_stride, ip_shape, op_unit):
        cls.c_stride = c_stride
        cls.ip_shape = ip_shape
        cls.op_unit = op_unit

    def __init__(self, c_stride, ip_shape, op_unit):
        self.update(c_stride, ip_shape, op_unit)


class SeverityClassification:
    def __init__(self, experiment_type = ExperimentType):
        self.experiment_type = experiment_type
        self.punctuations = list(string.punctuation)
        self.ip_txt_file = "data/500_Reddit_users_posts_labels.csv"                 
        self.ip_feat_file = "data/External_Features.csv"                           
        self.w2v_file = {'file': 'data/numberbatch-en.txt', 'is_binary': False}

        # File used to save the model outputs with 314 features
        if self.experiment_type == ExperimentType.FIVE_LABEL:
            self.op_file = "data/Result_Classification_5.tsv"
            
        elif self.experiment_type == ExperimentType.FOUR_LABEL:
            self.op_file = "data/Result_Classification_4.tsv"

        elif self.experiment_type == ExperimentType.THREE_PLUS_ONE_LABEL:
            self.op_file = "data/Result_Classification_3+1.tsv"

        if self.experiment_type == ExperimentType.FIVE_LABEL or self.experiment_type == ExperimentType.FOUR_LABEL:
            self.severity_classes = {'Supportive': 0, 'Indicator': 1, 'Ideation': 2, 'Behavior': 3, 'Attempt': 4}

        elif self.experiment_type == ExperimentType.THREE_PLUS_ONE_LABEL:
            self.severity_classes = {'Supportive': 0, 'Indicator': 0, 'Ideation': 1, 'Behavior': 2, 'Attempt': 3}

        # Hyper Parameters
        self.embedding_dimensions = 300
        self.max_sentence_len = 1500
        self.str_padd = '@PADD'
        self.cross_val_k = 5

        self.cnnConfig = CNNConfiguration(c_stride=(1, self.embedding_dimensions),
                                          ip_shape=(self.max_sentence_len, self.embedding_dimensions, 1),
                                          op_unit=5 if self.experiment_type == ExperimentType.FIVE_LABEL else 4)

        self.intermediate_layer = 'flat_drop'

    def vectorize_data(self, lst_input):
        x_data = []
        padd = self.str_padd
        wv_size = self.embedding_dimensions

        w2v_model = KeyedVectors.load_word2vec_format(self.w2v_file['file'], binary=self.w2v_file['is_binary'])

        vocab = w2v_model.key_to_index
        padding_zeros = np.zeros(wv_size, dtype=np.float32)

        for sent in lst_input:
            emb = []
            for token in sent:
                if token.lower() == padd:
                    emb.append(list(padding_zeros))

                elif token.lower() in vocab.keys():
                    emb.append(w2v_model[token.lower()].astype(float).tolist())

                else:
                    emb.append(list(padding_zeros))

            x_data.append(emb)

        del w2v_model, vocab

        return np.array(x_data)
    
    def read_input_file(self, input_file, exclude_class=None):
        x_data, y_data = [], []

        def should_process_row(row, exclude_class):
            if exclude_class is not None and not row[2] == exclude_class:
                return True
            return exclude_class is None

        if input_file:
            with open(input_file, "r") as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                next(csv_reader)

                for row in csv_reader:
                    if should_process_row(row, exclude_class):
                        sent = row[1]
                        printable = set(string.printable)

                        lst_tokens = []
                        for item in word_tokenize(sent):
                            if item not in self.punctuations and item not in printable:
                                lst_tokens.append(item.lower().strip("".join(self.punctuations)))

                        if len(lst_tokens) > self.max_sentence_len:
                            lst_tokens = lst_tokens[:self.max_sentence_len]

                        elif len(lst_tokens) < self.max_sentence_len:
                            for _ in range(len(lst_tokens), self.max_sentence_len):
                                lst_tokens.append(self.str_padd)

                        if self.experiment_type == ExperimentType.FOUR_LABEL:
                            y_data.append(self.severity_classes[row[2].strip()] - 1)

                        else:
                            y_data.append(self.severity_classes[row[2].strip()])
                        
                        x_data.append(lst_tokens)

        return x_data, y_data

    def read_external_features(self, raw_data, raw_features):
        user_ids, features = [], []

        with open(raw_data) as file:
            for line in file:
                split = line.strip().split(',')
                user_ids.append(split[0])

        with open(raw_features) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')

            next(csv_reader)
            for row in csv_reader:
                scores = [float(value) for value in row[1:]]
                features.append(scores)

        return np.array(features)
    
    def read_data(self, ip_file):
        if self.experiment_type == ExperimentType.FOUR_LABEL:
            x_data, y_data = self.read_input_file(ip_file, exclude_class="Supportive")

        else:
            x_data, y_data = self.read_input_file(ip_file)

        x_data = self.vectorize_data(x_data)
        x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], x_data.shape[2], 1)
        x_data, y_data = np.array(x_data), np.array(y_data)
        return x_data, y_data
    
    
    def get_cnn_model(self):
        l_ip = Input(shape=(self.max_sentence_len, self.embedding_dimensions, 1), dtype='float32')
        lst_convfeat = []
        for filter in self.cnnConfig.kernels:
            l_conv = Conv2D(
                filters=self.cnnConfig.no_filters, 
                kernel_size=(filter, self.embedding_dimensions), 
                strides=self.cnnConfig.c_stride,
                padding=self.cnnConfig.pad, 
                data_format='channels_last', 
                input_shape=self.cnnConfig.ip_shape,
                activation=self.cnnConfig.c_activ
            )(l_ip)

            l_pool = MaxPool2D(pool_size=(self.max_sentence_len, 1))(l_conv)
            lst_convfeat.append(l_pool)

        l_concat = Concatenate(axis=1)(lst_convfeat)
        l_flat = Flatten()(l_concat)
        l_drop = Dropout(
            rate=self.cnnConfig.drop_rate, 
            name='flat_drop'
        )(l_flat)

        l_op = Dense(
            units=self.cnnConfig.op_unit, 
            activation=self.cnnConfig.op_activ, 
            name='cnn_op'
        )(l_drop)

        final_model = Model(l_ip, l_op)
        
        final_model.compile(
            optimizer=adam.Adam(learning_rate=self.cnnConfig.l_rate), 
            loss=self.cnnConfig.loss, 
            metrics=['accuracy']
        )

        return final_model
    
    def mlp_model(self, ip_dim):
        mlp_model = Sequential()

        if self.experiment_type == ExperimentType.THREE_PLUS_ONE_LABEL:
            mlp_model.add(
                Dense(
                    units=self.cnnConfig.op_unit, 
                    activation=self.cnnConfig.op_activ, 
                    name='classif_op'
                )
            )
        else:
            mlp_model.add(
                Dense(
                    units=self.cnnConfig.op_unit, 
                    activation=self.cnnConfig.op_activ, 
                    name='classif_op',
                    input_dim=ip_dim
                )
            )

        mlp_model.compile(
            optimizer=adam.Adam(learning_rate=self.cnnConfig.l_rate), loss=self.cnnConfig.loss, 
            metrics=['accuracy']
        )

        return mlp_model
    
    # Compute Precision, Recall, and F1-score
    def get_prf1(self, y_true, y_pred):
        tp, fp, fn = 0.0, 0.0, 0.0

        for i in range(len(y_pred)):
            if y_pred[i] == y_true[i]:
                tp += 1
            elif y_pred[i] > y_true[i]:
                fp += 1
            else:
                fn += 1
        
        if fn == 0:
            fn  = 1.0
        
        if tp == 0:
            tp = 1.0

        if fp == 0:
            fp = 1.0
        
        R = tp / (tp + fn)
        P = tp / (tp + fp)
        F = 2 * P * R / (P + R)
        
        print ('\nPrecision: {0}\t Recall: {1}\t F1-Score: {2}'.format(P, R, F))
        return {'P': P, 'R': R, 'F': F}
    
    def oe_score(self, y_true, y_pred):
        oe_no = 0
        nt = len(y_pred)

        for i in range(nt):
            if abs(y_pred[i] - y_true[i]) > 1:
                oe_no += 1
        
        OE = oe_no / nt
        
        print('OE:{}'.format(OE))
        return {'OE': OE}

    def scores(self, ypred,ytest):
        ypred = np.argmax(ypred, axis=-1)
        ytest = np.argmax(ytest, axis=-1)
        score = self.get_prf1(ytest, ypred)
        oe = self.oe_score(ytest, ypred)
        return(score, oe)
    
    def run(self):
        # Start time:
        start_time=time.time()

        with open(self.op_file, 'w') as of:

            x_data, y_data = self.read_data(self.ip_txt_file)

            ext_feature = self.read_external_features(self.ip_txt_file, self.ip_feat_file)

            cv_count = 0
            k_score, oescore = [], []
        
            kscore_svmln, oescore_svmln = [], []

            if self.experiment_type == ExperimentType.FIVE_LABEL or self.experiment_type == ExperimentType.FOUR_LABEL:
                k_score_ffnn, oescore_ffnn = [], []
                kscore_svmrbf, oescore_svmrbf = [], []
                kscore_rf, oescore_rf = [], []
            
            # Stratified cross-validation
            skf = StratifiedKFold(n_splits = self.cross_val_k)
            skf.get_n_splits(x_data, y_data)

            # Run the model for each splits
            for train_index, test_index in skf.split(x_data, y_data):
                cv_count += 1
                print ('\nRunning Stratified Cross Validation: {0}/{1}...'.format(cv_count, self.cross_val_k))

                x_train, x_test = x_data[train_index], x_data[test_index]
                y_train, y_test = y_data[train_index], y_data[test_index]

                # Convert the class labels into categorical
                y_train, y_test = to_categorical(y_train), to_categorical(y_test)

                # Reshape the data for CNN
                x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
                x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

                # External features for this particular split
                train_ext_feat, test_ext_feat = ext_feature[train_index], ext_feature[test_index]

                # CNN model for training on the embedded text input
                cnn_model = self.get_cnn_model()

                # Train the model
                cnn_model.fit(
                    x=x_train, 
                    y=y_train, 
                    batch_size=self.cnnConfig.batch, 
                    epochs=self.cnnConfig.epoch, 
                    verbose=self.cnnConfig.verbose
                )
                
                # Trained model for extracting features from intermediate layer
                model_feat_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.get_layer(self.intermediate_layer).output)

                # Get CNN gerated features
                train_cnn_feat = model_feat_extractor.predict(x_train)
                test_cnn_feat = model_feat_extractor.predict(x_test)

                # Merge the CNN generated features with the external features
                x_train_features = []
                for index, cnn_feature in enumerate(train_cnn_feat):
                    tmp_feat = list(cnn_feature)

                    # When running model without these 14 features, comment the following lines. So these 14 features won't be added to the input data
                    tmp_feat.extend(list(train_ext_feat[index]))
                    x_train_features.append(np.array(tmp_feat))

                x_test_features = []
                for index, cnn_feature in enumerate(test_cnn_feat):
                    tmp_feat = list(cnn_feature)

                    # When running model without these 14 features, comment the following lines. So these 14 features won't be added to the input data
                    tmp_feat.extend(list(test_ext_feat[index]))
                    x_test_features.append(np.array(tmp_feat))

                # Convert the list into numpy array
                x_train_features = np.array(x_train_features)
                x_test_features = np.array(x_test_features)

                del train_cnn_feat, test_cnn_feat

                # Get the MLP model for final classification
                mlp_model = self.mlp_model(ip_dim = len(x_train_features[0]))
                tc = time.time()

                # Train the MLP model
                mlp_model.fit(x=x_train_features, y=y_train, batch_size=self.cnnConfig.batch, epochs=self.cnnConfig.epoch, verbose=self.cnnConfig.verbose)
                            
                print ('\nTime elapsed in training CNN: ', str(datetime.timedelta(seconds=time.time() - tc)))

                print ('\nEvaluating on Test data...\n')
                
                # Print Loss and Accuracy
                model_metrics = mlp_model.evaluate(x_test_features, y_test)

                for i in range(len(model_metrics)):
                    print (mlp_model.metrics_names[i], ': ', model_metrics[i])

                y_pred = mlp_model.predict(x_test_features)
                
                score,oe = self.scores(y_pred,y_test)
                k_score.append(score)
                oescore.append(oe)
                y_pred = np.argmax(y_pred, axis=-1)
                y_test_mlp = np.argmax(y_test, axis=-1)

                # Scikit-learn classification report (P, R, F1, Support)
                report = classification_report(y_test_mlp, y_pred)
                print (report)

                of.write('Cross_Val:\n')
                for i in range(len(y_pred)):
                    of.write('\t'.join([str(y_test_mlp[i]), str(y_pred[i])]) + '\n')

                # FFNN
                if self.experiment_type == ExperimentType.FIVE_LABEL or self.experiment_type == ExperimentType.FOUR_LABEL:
                    if self.experiment_type == ExperimentType.FIVE_LABEL:
                        ffnn_model = Sequential([
                            # Flatten(input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])),
                            Dense(128, activation='relu'),
                            Dense(128, activation='relu'),
                            Dense(5, activation='softmax'),
                        ])
                    else:
                        ffnn_model = Sequential([
                            # Flatten(input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])),
                            Dense(128, activation = 'relu'),
                            Dense(128, activation = 'relu'),
                            Dense(4, activation = 'softmax'),
                        ])

                    optimiser = adam.Adam(learning_rate=0.001)
                    ffnn_model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['accuracy'])

                    ffnn_model.fit(x_train_features, y_train, batch_size=4, epochs=50)

                    y_pred_ffnn = ffnn_model.predict(x_test_features)
                    
                    score_ffnn, oe_ffnn=self.scores(y_pred_ffnn, y_test)
                    k_score_ffnn.append(score_ffnn)
                    oescore_ffnn.append(oe_ffnn)
            
                ytrain_new = [key for ss in y_train for key,val in enumerate(ss) if val==1]
                ytest_new = [key for ss in y_test for key,val in enumerate(ss) if val==1]

                # SVM - Linear
                from sklearn.svm import SVC
                model_svm_ln = SVC(kernel='linear', probability=True)
                model_svm_ln.fit(x_train_features, np.array(ytrain_new))
                y_pred_svmln=model_svm_ln.predict(x_test_features)
                score_svmln = self.get_prf1(ytest_new, y_pred_svmln.tolist())
                oe_svmln = self.oe_score(ytest_new, y_pred_svmln.tolist())
                kscore_svmln.append(score_svmln)
                oescore_svmln.append(oe_svmln)

                if self.experiment_type == ExperimentType.FIVE_LABEL or self.experiment_type == ExperimentType.FOUR_LABEL:
                    # SVM - RBF
                    model_svm_rbf = SVC(kernel='rbf')
                    model_svm_rbf.fit(x_train_features, np.array(ytrain_new))
                    y_pred_svmrbf = model_svm_rbf.predict(x_test_features)
                    score_svmrbf = self.get_prf1(ytest_new, y_pred_svmrbf.tolist())
                    oe_svmrbf = self.oe_score(ytest_new, y_pred_svmrbf.tolist())
                    kscore_svmrbf.append(score_svmrbf)
                    oescore_svmrbf.append(oe_svmrbf)

                    # RF
                    from sklearn.ensemble import RandomForestClassifier
                    model_rf = RandomForestClassifier()
                    model_rf.fit(x_train_features, np.array(ytrain_new))
                    y_pred_rf=model_rf.predict(x_test_features)
                    score_rf = self.get_prf1(ytest_new, y_pred_rf.tolist())
                    oe_rf = self.oe_score(ytest_new, y_pred_rf.tolist())
                    kscore_rf.append(score_rf)
                    oescore_rf.append(oe_rf)

                del x_train, y_train


            print('============= MLP model: 300 Features from CNN model + 14 External Features\n')
            print (k_score)        
            print("\n", oescore)

            avgP = np.average([score['P'] for score in k_score])
            avgR = np.average([score['R'] for score in k_score])
            avgF = np.average([score['F'] for score in k_score])
            avgOE = np.average([score['OE'] for score in oescore])

            print ('\nAfter Stratified Cross Validation Average Precision: {0}\t Recall: {1}\t F1-Score: {2}\t OE-Score:{3}'.format(avgP, avgR, avgF,avgOE))
            
            print('============= SVM Linear: 300 Features from CNN model + 14 External Features\n')

            print (kscore_svmln)        
            print("\n",oescore_svmln)

            avgP_svmln = np.average([score['P'] for score in kscore_svmln])
            avgR_svmln = np.average([score['R'] for score in kscore_svmln])
            avgF_svmln = np.average([score['F'] for score in kscore_svmln])
            avgOEs_svmln = np.average([score['OE'] for score in oescore_svmln])

            print ('\nAfter Stratified Cross Validation Average Precision: {0}\t Recall: {1}\t F1-Score: {2}\t OE-Score:{3}'.format(avgP_svmln,avgR_svmln,avgF_svmln,avgOEs_svmln ))

            if self.experiment_type == ExperimentType.FIVE_LABEL or self.experiment_type == ExperimentType.FOUR_LABEL:
                print('============= FFNN model: 300 Features from CNN model + 14 External Features\n')
                print (k_score)        
                print("\n",oescore)

                avgP_ffnn = np.average([score['P'] for score in k_score_ffnn])
                avgR_ffnn = np.average([score['R'] for score in k_score_ffnn])
                avgF_ffnn = np.average([score['F'] for score in k_score_ffnn])
                avgOE_ffnn = np.average([score['OE'] for score in oescore_ffnn])

                print ('\nAfter Stratified Cross Validation Average Precision: {0}\t Recall: {1}\t F1-Score: {2}\t OE-Score:{3}'.format(avgP_ffnn, avgR_ffnn, avgF_ffnn,avgOE_ffnn))

                print('============= SVM RBF: 300 Features from CNN model + 14 External Features\n')

                print (kscore_svmrbf)        
                print("\n",oescore_svmrbf)

                avgP_svmrbf = np.average([score['P'] for score in kscore_svmrbf])
                avgR_svmrbf = np.average([score['R'] for score in kscore_svmrbf])
                avgF_svmrbf = np.average([score['F'] for score in kscore_svmrbf])
                avgOEs_svmrbf = np.average([score['OE'] for score in oescore_svmrbf])

                print ('\nAfter Stratified Cross Validation Average Precision: {0}\t Recall: {1}\t F1-Score: {2}\t OE-Score:{3}'.format(avgP_svmrbf,avgR_svmrbf,avgF_svmrbf,avgOEs_svmrbf ))

                print('============= RF: 300 Features from CNN model + 14 External Features\n')

                print (kscore_rf)        
                print("\n",oescore_rf)

                avgP_rf = np.average([score['P'] for score in kscore_rf])
                avgR_rf = np.average([score['R'] for score in kscore_rf])
                avgF_rf = np.average([score['F'] for score in kscore_rf])
                avgOEs_rf = np.average([score['OE'] for score in oescore_rf])

                print ('\nAfter Stratified Cross Validation Average Precision: {0}\t Recall: {1}\t F1-Score: {2}\t OE-Score:{3}'.format(avgP_rf,avgR_rf,avgF_rf,avgOEs_rf ))

            end_time = time.time()
            print("Time taken to complete the task:{} hours".format((end_time-start_time) / 3600))

if len(sys.argv) <= 1:
    print("Insufficient arguments passed.")

elif sys.argv[1] == "five":
    exp = SeverityClassification(ExperimentType.FIVE_LABEL)
    exp.run()

elif sys.argv[1] == "four":
    exp = SeverityClassification(ExperimentType.FOUR_LABEL)
    exp.run()

elif sys.argv[1] == "three":
    exp = SeverityClassification(ExperimentType.THREE_PLUS_ONE_LABEL)
    exp.run()

else:
    print("Invalid argument passed.")