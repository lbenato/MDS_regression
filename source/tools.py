import awkward as ak
import glob
import time

import os
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np





class tools:
    def select(self,data, column_name, value, operator, verbosity = False, absolute=False):
    
        column_values = data[column_name]

        operator_dict = {
        "==": np.equal,
        ">": np.greater,
        ">=": np.greater_equal,
        "<":np.less,
        "<=":np.less_equal,
        "!=":np.not_equal
        }
        if operator not in operator_dict:
            raise ValueError(f"Invalid operator: {operator}")

        compare_fn = operator_dict[operator]
        mask = compare_fn(column_values, value)
        selected_values = data[mask]


        if verbosity:
            print(f"Selected values for '{column_name}' {operator} {value}:")
            print('values before selection', data)
            print('number of entries before selection', len(data))
            print('values after selection', selected_values)
            print('number of entries after selection',len(selected_values))
            print('mask', mask)
            print('shape before', data.shape)
            print('shape after', selected_values.shape)

     #should add absolute   
        if absolute:
            column_values = np.absolute(column_values)
            mask = compare_fn(column_values, value)

        return selected_values

    
    
    
    #function for plotting histograms
    def plotHist(self, data_sig, data_bkg, bins = 10, interval=None, logy=False, logx=False, labels=None,
             xlabel=None, ylabel=None, density=False, ax=None, moveOverUnderFlow=True, verbosity = 0, doShow=True):
    
        if not ax: fig, ax = plt.subplots()

        if verbosity:
            print("")
            print("starting plotting")
            print("signal")
            print(len(data_sig), min(data_sig), max(data_sig))
            print("background")
            print(len(data_bkg), min(data_bkg), max(data_bkg))

        # handle over/under flows
        the_sig = data_sig
        the_bkg = data_bkg
        if moveOverUnderFlow and interval!=None:
            the_sig = np.maximum(the_sig, interval[0]) # move all values smaller than lower interval value to this value
            the_sig = np.minimum(the_sig, interval[1]) # move all values larger than upper interval value to this value
            the_bkg = np.maximum(the_bkg, interval[0]) # move all values smaller than lower interval value to this value
            the_bkg = np.minimum(the_bkg, interval[1]) # move all values larger than upper interval value to this value

            if verbosity:
                print("After moving under/overflow plotting")
                print("to range", interval)
                print("signal")
                print(len(the_sig), min(the_sig), max(the_sig))
                print("background")
                print(len(the_bkg),  min(the_bkg), max(the_bkg))        

        # creating histogram from data
        # We only need this for the bin_edges.
        # We will use the one from the signal and also apply this to the background
        # We will not use the histograms themselves for now. There are ways to plot this using the histograms, but i need to figure out how
        #hist_data_sig, hist_edges_sig = np.histogram(the_sig, bins=bins, density=density, range=interval)
        # hist_data_bkg, hist_edges_bkg = np.histogram(the_bkg, bins=hist_edges_sig, density=density)


        # handle the labels
        label_sig = None
        label_bkg = None
        if isinstance(labels, tuple) or isinstance(labels, list):
            if len(labels)==2:
                label_sig = labels[0]
                label_bkg = labels[1]
        elif isinstance(labels, str):
            label_sig = labels
            label_bkg = labels    

        # plotting the histogram
        # use same bin edges for signal and background
        hist_data_sig, hist_edges_sig, ooo = ax.hist(the_sig, bins=bins, range=interval, density=density, label=label_sig, alpha=0.9, color="r", histtype="step") 

        if verbosity:
            print()
            print("signal histo")
            print(hist_data_sig, type(hist_data_sig))
            print("signal edges")
            print(hist_edges_sig)


        ax.hist(the_bkg, bins=hist_edges_sig, range=interval, density=density, label=label_bkg, alpha=0.9, color="b", histtype="step") 
        if logy: ax.set_yscale("log")
        #if logx: ax.set_xscale("log")
        
        if ylabel==None:
            ax.set_ylabel("jets")
            if(density): ax.set_ylabel("jets / total jets")
        else:
            ax.set_ylabel(ylabel)
            if(density): ax.set_ylabel(ylabel+" normalized")
        
        if(xlabel): ax.set_xlabel(xlabel)

        # add legend
        if labels!=None:
            legend = ax.legend()
        
        if doShow:
            plt.show()
            
        return fig
        
