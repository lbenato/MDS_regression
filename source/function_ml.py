import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from keras import layers

import glob
import time
import importlib

import os
import tempfile

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def data_preprocessing_training(signal, background, parametrisation = 0, train_network = 1,set_epochs = 0, set_batches =0, save_model = 1, save_plots = 1, dnn_output = 0 ):
    if train_network == 1:
        if set_epochs == 1:
            print("Enter the number of epochs for training:")
            EPOCHS = int(input())
        else:
            EPOCHS = 100
    
    if set_batches == 1:
        print("Enter the batch size for training:")
        BATCH_SIZE = int(input())
    else:
        BATCH_SIZE = 2048
    
     #figure out a way to give save_model == 0 and give a default path
    save_path = input(" Enter the path for saving the model. Please use '.keras' extension")
    
        
    smaller_branchlist = ["Jets/Jets.mass", 
    "Jets/Jets.energy",
    "Jets/Jets.pt", 
    "Jets/Jets.eta", 
    "Jets/Jets.flightDist3d", 
    "Jets/Jets.flightDist3dError", 
    "Jets/Jets.flightDist2d", 
    "Jets/Jets.flightDist2dError", 
    "Jets/Jets.nSV", 
    "Jets/Jets.nVertexTracks", 
    "Jets/Jets.SV_mass", 
    "Jets/Jets.nTracksSV", 
    "Jets/Jets.deepJet_probc_probg_probuds", 
    "Jets/Jets.deepJet_probb_probbb_problepb"]
    
    
    if parametrisation == 1:
        smaller_branchlist = ["Jets/Jets.mass", 
        "Jets/Jets.energy",
        "Jets/Jets.pt", 
        "Jets/Jets.eta", 
        "Jets/Jets.flightDist3d", 
        "Jets/Jets.flightDist3dError", 
        "Jets/Jets.flightDist2d", 
        "Jets/Jets.flightDist2dError", 
        "Jets/Jets.nSV", 
        "Jets/Jets.nVertexTracks", 
        "Jets/Jets.SV_mass", 
        "Jets/Jets.nTracksSV", 
        "Jets/Jets.deepJet_probc_probg_probuds", 
        "Jets/Jets.deepJet_probb_probbb_problepb",
        "llp_mass",
        "llp_lifetime"]
        
    # Splitting the loaded_background data
    bkg_train_data, bkg_testval_data = train_test_split( background, train_size=0.6)
    bkg_test_data, bkg_val_data = train_test_split(bkg_testval_data, test_size=0.5)

    # Splitting the loaded_array_signal data
    sig_train_data, sig_testval_data = train_test_split(signal, train_size=0.6)
    sig_test_data, sig_val_data = train_test_split(sig_testval_data, test_size=0.5)

    # assigning labels

    bkg_train_labels = np.array([0.]*len(bkg_train_data))
    bkg_test_labels = np.array([0.]*len(bkg_test_data))
    bkg_val_labels = np.array([0.]*len(bkg_val_data))

    sig_train_labels = np.array([1.]*len(sig_train_data))
    sig_test_labels = np.array([1.]*len(sig_test_data))
    sig_val_labels = np.array([1.]*len(sig_val_data))

    # check the shape
    print("First example:")
    print("background training features", bkg_train_data.shape)
    print("background training labels", bkg_train_labels.shape)
    print("background test features", bkg_test_data.shape)
    print("background test labels", bkg_test_labels.shape)
    print("background val features", bkg_val_data.shape)
    print("background val labels", bkg_val_labels.shape)

    print("signal training features", sig_train_data.shape)
    print("signal training labels", sig_train_labels.shape)
    print("signal test features", sig_test_data.shape)
    print("signal test labels", sig_test_labels.shape)
    print("signal val features", sig_val_data.shape)
    print("signal val labels", sig_val_labels.shape)

    #check data types
    print("Data Types:")
    print("bkg_train_data:", bkg_train_data.dtype)
    print("sig_train_data:", sig_train_data.dtype)
    print("bkg_test_data:", bkg_test_data.dtype)
    print("sig_test_data:", sig_test_data.dtype)
    print("bkg_val_data:", bkg_val_data.dtype)
    print("sig_val_data:", sig_val_data.dtype)


    #print(bkg_train_labels)
    #print(bkg_train_data)

    #print(type(bkg_train_data), type(bkg_train_labels))
    
    # concatenate data

    train_data = np.concatenate([bkg_train_data, sig_train_data], axis=0)
    test_data = np.concatenate([bkg_test_data, sig_test_data], axis=0)
    val_data = np.concatenate([bkg_val_data, sig_val_data], axis=0)

    train_labels = np.concatenate([bkg_train_labels, sig_train_labels], axis=0)
    test_labels = np.concatenate([bkg_test_labels, sig_test_labels], axis=0)
    val_labels = np.concatenate([bkg_val_labels, sig_val_labels], axis=0)

    print("training features", train_data.shape)
    print("training labels", train_labels.shape)
    print("test features", test_data.shape)
    print("test labels", test_labels.shape)
    print("val features", val_data.shape)
    print("val labels", val_labels.shape)
    print(train_data)

    # TODO: standard scalar. Fit on train_data and apply to test_data and val_data
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    val_data = scaler.transform(val_data)
    test_data = scaler.transform(test_data)
    
    # shuffle

    X_train, Y_train = sklearn.utils.shuffle(train_data, train_labels, random_state=None) # change to random_state=None for full randomness
    X_test, Y_test = sklearn.utils.shuffle(test_data, test_labels, random_state=0) # change to random_state=None for full randomness
    X_val, Y_val = sklearn.utils.shuffle(val_data, val_labels, random_state=0) # change to random_state=None for full randomness

    print("shuffled training features", X_train)
    print("shuffled training features", X_train.shape)
    print("shuffled training labels", Y_train)
    print("shuffled training labels", Y_train.shape)
    
    n_features = len(smaller_branchlist)
    
    
    #defining the model and metrics
    METRICS = [
          keras.metrics.BinaryCrossentropy(name='cross entropy'),  # same as model's loss
          keras.metrics.MeanSquaredError(name='Brier score'),
          keras.metrics.TruePositives(name='tp'),
          keras.metrics.FalsePositives(name='fp'),
          keras.metrics.TrueNegatives(name='tn'),
          keras.metrics.FalseNegatives(name='fn'), 
          keras.metrics.BinaryAccuracy(name='accuracy'),
          keras.metrics.Precision(name='precision'),
          keras.metrics.Recall(name='recall'),
          keras.metrics.AUC(name='auc'),
          keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
    ]


    def make_model(metrics=METRICS, output_bias=None):
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)
        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)

        model = keras.Sequential([
            keras.layers.Dense(128, activation='LeakyReLU', input_shape=(n_features,), kernel_initializer=initializer, bias_initializer=None),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(128, activation='LeakyReLU', kernel_initializer=initializer, bias_initializer=None),  # Additional dense layer

            keras.layers.Dropout(0.2),

            keras.layers.Dense(128, activation='LeakyReLU', kernel_initializer=initializer, bias_initializer=None),  # Additional dense layer
            keras.layers.Dropout(0.2),
            keras.layers.Dense(128, activation='LeakyReLU', kernel_initializer=initializer, bias_initializer=None),  # Additional dense layer
            keras.layers.Dropout(0.1),
            keras.layers.Dense(8, activation='LeakyReLU', kernel_initializer=initializer, bias_initializer=None),  # Additional dense layer
            keras.layers.Dense(4, activation='LeakyReLU', kernel_initializer=initializer, bias_initializer=None),  # Additional dense layer
            keras.layers.Dense(2, activation='LeakyReLU', kernel_initializer=initializer, bias_initializer=None),  # Additional dense layer
            keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias, kernel_initializer=initializer)

        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=metrics
        )

        return model

    #EPOCHS = 100
    #BATCH_SIZE = 2048

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_prc', 
        verbose=1,
        patience=10,
        mode='max',
        restore_best_weights=True)
    
    model = make_model()
    model.summary()
    
    output = model.predict(X_train[:10])
    print(output)
    print(Y_train[:10])
    
    results = model.evaluate(X_train, Y_train, batch_size=BATCH_SIZE, verbose=0)
    print("Loss: {:0.4f}".format(results[0]))
    
    #saving the model
    #ask the user to input a path for saving the model
    #save_path = input("Enter the path to save the model: ")
    #if not os.path.exists(save_path):
        #os.makedirs(save_path)
    #saved_model_path = os.path.join(save_path, "parametrised_model")
    
    #saved_model = tf.keras.saving.save_model(model, save_path, overwrite=False, save_format='tf')
    model.save(save_path)
    
    
    #Train the model
    if train_network == 1:
        model = make_model()
        baseline_history = model.fit(
        X_train,
        Y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS, #EPOCHSf
        callbacks=[early_stopping],
        validation_data=(X_val, Y_val))
        
        #plot metrics
        
        def plot_metrics(history):
         metrics = ['loss', 'prc', 'precision', 'recall']
         fig, axs = plt.subplots(2, 2, figsize=(10, 8))
         fig.subplots_adjust(hspace=0.3)  # Adjust the vertical spacing between subplots



         for n, metric in enumerate(metrics):
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            name = metric.replace("_"," ").capitalize()
            plt.subplot(2,2,n+1)
            plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
            plt.plot(history.epoch, history.history['val_'+metric],
                    color=colors[0], linestyle="--", label='Val')
            plt.xlabel('Epoch')
            plt.ylabel(name)
            if metric == 'loss':
                plt.ylim([0.1, plt.ylim()[1]])
                plt.yscale("log")
            elif metric == 'auc':
                  plt.ylim([0.8,1])
            else:
                  plt.ylim([0,1])

            plt.legend()
            plt.savefig('training_evaluation.pdf')

        plot_metrics(baseline_history)

    else:
        loaded_model = keras.models.load_model(save_path)
        #loaded_model = tf.keras.saving.load_model(saved_model, compile = True, safe_mode = True)
    
    train_predictions_baseline = model.predict(train_data, batch_size=BATCH_SIZE)
    test_predictions_baseline = model.predict(test_data, batch_size=BATCH_SIZE)
    print(len(test_data),len(test_predictions_baseline))
    
    from source.tools import tools
    obj = tools()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    obj.plot_roc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])
    obj.plot_roc("Test Baseline", test_labels, test_predictions_baseline, color=colors[1], linestyle='--')
    plt.legend(loc='lower right')
    plt.savefig('roc.pdf')

    sklearn.metrics.roc_auc_score(test_labels, test_predictions_baseline)
    sklearn.metrics.roc_auc_score(train_labels, train_predictions_baseline)
    
    def plot_cm(labels, predictions, p=0.5):
        cm = confusion_matrix(labels, predictions > p)
        plt.figure(figsize=(5,5))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title('Confusion matrix @{:.2f}'.format(p))
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.savefig('confusion_matrix.pdf')
        
    #confusion matrix
    baseline_results = model.evaluate(test_data, test_labels,
                                      batch_size=BATCH_SIZE, verbose=0)
    for name, value in zip(model.metrics_names, baseline_results):
        print(name, ': ', value)
        print()

    plot_cm(test_labels, test_predictions_baseline)
    
    ##DNN output plots
    if dnn_output == 1:
        loaded_sig_concat = np.load('sig_concat.npy')
        loaded_bkg_concat = np.load('bkg_concat.npy')

        
        from source.tools import tools
        scaler = StandardScaler()

        #selecting llp mass and lifetimes 
        mass_lifetime_pairs = [
            (15,  1),
            (15, 10),
            (40, 10),
            (40, 1),
            (55, 1),
            (55, 10)
        ]
        for mass, lifetime in mass_lifetime_pairs:
            print(f"LLP_Mass: {mass}, LLP_Lifetime: {lifetime}")
            obj = tools()
            selected_signal_mass = obj.select(loaded_sig_concat, 'llp_mass' , mass,'==')
            selected_signal_mass_lifetime = obj.select(selected_signal_mass, 'llp_lifetime', lifetime, '==')

            selected_bkg_mass = obj.select(loaded_bkg_concat, 'llp_mass' , mass,'==')
            selected_bkg_mass_lifetime = obj.select(selected_bkg_mass, 'llp_lifetime', lifetime, '==')

            smaller_branchlist = ["Jets/Jets.mass", 
            "Jets/Jets.energy",
            "Jets/Jets.pt", 
            "Jets/Jets.eta", 
            "Jets/Jets.flightDist3d", 
            "Jets/Jets.flightDist3dError", 
            "Jets/Jets.flightDist2d", 
            "Jets/Jets.flightDist2dError", 
            "Jets/Jets.nSV", 
            "Jets/Jets.nVertexTracks", 
            "Jets/Jets.SV_mass", 
            "Jets/Jets.nTracksSV", 
            "Jets/Jets.deepJet_probc_probg_probuds", 
            "Jets/Jets.deepJet_probb_probbb_problepb",
            "llp_mass",
            "llp_lifetime"]



            selected_columns_array_signal = selected_signal_mass_lifetime[smaller_branchlist]
            selected_columns_array_background = selected_bkg_mass_lifetime[smaller_branchlist]

            loaded_background_df = np.lib.recfunctions.structured_to_unstructured(selected_columns_array_background)
            loaded_array_signal_df = np.lib.recfunctions.structured_to_unstructured(selected_columns_array_signal)




            sig_train_data, sig_testval_data = train_test_split(loaded_array_signal_df, train_size=0.6)
            sig_test_data, sig_val_data = train_test_split(sig_testval_data, test_size=0.5)

            bkg_train_data, bkg_testval_data = train_test_split(loaded_background_df, train_size=0.6)
            bkg_test_data, bkg_val_data = train_test_split(bkg_testval_data, test_size=0.5)

            sig_train_data_scaled = scaler.fit_transform(sig_train_data)
            bkg_train_data_scaled = scaler.transform(bkg_train_data)

            sig_test_data_scaled = scaler.transform(sig_test_data)
            bkg_test_data_scaled = scaler.transform(bkg_test_data)

            train_predictions_signal = model.predict(sig_train_data_scaled, batch_size=BATCH_SIZE)
            train_predictions_background = model.predict(bkg_train_data_scaled, batch_size=BATCH_SIZE)

            # Predictions on testing data
            test_predictions_signal = model.predict(sig_test_data_scaled, batch_size=BATCH_SIZE)
            test_predictions_background = model.predict(bkg_test_data_scaled, batch_size=BATCH_SIZE)

            # Instantiate the tools object
            obj = tools()

        # Plot histograms for training data
        obj.plotHist(train_predictions_signal, train_predictions_background, bins=50, interval=[0, 1], logy=True, logx=False,
                     labels=["signal train", "background train"], xlabel='DNN output', density=False, ax=None,
                     moveOverUnderFlow=True, verbosity=0)
        plt.savefig('dnn_output_sig_train_vs_bkg_train.pdf')

        # Plot histograms for testing data
        obj.plotHist(test_predictions_signal, test_predictions_background, bins=50, interval=[0, 1], logy=True, logx=False,
                     labels=["signal test", "background test"], xlabel='DNN output', density=False, ax=None,
                     moveOverUnderFlow=True, verbosity=0)
        plt.savefig('dnn_output_sig_test_vs_bkg_test.pdf')

        # Print some part of train_predictions_background
        print(train_predictions_background[:-100])

        # Plot histograms for baseline data (train and test)
        obj.plotHist(train_predictions_baseline, test_predictions_baseline, bins=50, interval=None, logy=True, logx=False,
                     labels=["train", "test"], xlabel='DNN output', density=False, ax=None,
                     moveOverUnderFlow=True, verbosity=0)
        plt.savefig('dnn_output_train_test_baseline.pdf')

