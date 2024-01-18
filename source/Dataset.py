import h5py
import vector
import uproot
import pandas as pd
import numpy as np
import awkward as ak
import glob
import time
import os
from numpy.lib import recfunctions



# THis is not a good idea!
# It is not always the case that you want to store there!
# directory = 'beegfs/LLP_ML/'
#os.makedirs(directory, exist_ok=True)

class Dataset:
    def __init__(self, name="dataset"):
        # Constructor
        # For now this only takes a name such as "signal" but later it might be useful to also construct datasets from existing datasets
        self.name=name
        # Some other class variables
        # Many of those could also be defined and discarded within the function they are used, but it might be useful to store them
        self.wildcard = ""
        self.list_of_root_files=[]
        self.branchnames = []
        self.n_entries_initial=0
        self.ak_array_raw = None
        self.ak_array_unrolled = None
        self.np_load_array = []
        self.np_array_structured = None
        self.np_array_unstructured = None

        
    def create_ak_array_from_tree(self, wildcard, tree_name, branchnames=[], verbosity=0):
        # read in the the trees with uproot and create awkward arrays
        # then store the arrays 
        self.branchnames=branchnames
        self.wildcard=wildcard
        self.list_of_root_files = glob.glob(self.wildcard)
        
       
        file_and_tree_expressions = [ rootfile+":"+tree_name for rootfile in self.list_of_root_files]
        if verbosity:
            print("converting the following root files and trees")
            for f in file_and_tree_expressions:
                print(f)
                
        print("start loading")
        # TODO: add timer to measure time of operation
        ak_array = uproot.concatenate(file_and_tree_expressions, branchnames)
        self.n_entries_initial=len(ak_array)
        print("converted trees to awkward array")
        print("found", self.n_entries_initial, "events")

        # The array self.ak_array has the complicated structure from the tree. One entry per event with multiple entries for each jet
        # We want to convert this to a flat array with one entry per jet
        # For this we will use a different method -> see below
        # For now we will return the array so that we can pass it to the next function
        # We could also store it in a class variable but maybe this needs to much memory. Need to test this
        return ak_array
        
    def get_raw_ak_array(self):
        return self.ak_array_raw
    
    def unroll_raw_array_jets(self, ak_array, max_events=-1, verbosity=0):
        # function that creates a new awkward array with one entry per jet
        # then stores the array in a class variable self.ak_array_unrolled
        print("Turning event array into jet array")
        list_of_columns = [c for c in ak_array[0]]
        list_of_entry_dicts = []
        total_jets = 0
        n_events_seen=0
        for event in ak_array:
            n_jets = len(event["Jets/Jets.pt"])
            total_jets += n_jets
            for ijet in range(n_jets):
            # create temporary dictionary to hold jet data
                data_dict = {col: event[col] if not col.startswith("Jets/") else event[col][ijet]
                         for col in list_of_columns}
                if verbosity>1:
                    print(data_dict)
                list_of_entry_dicts.append(data_dict)
            n_events_seen+=1
            if max_events>0 and n_events_seen>=max_events:
                break
                
        full_array = ak.Array(list_of_entry_dicts)
        self.ak_array_unrolled = full_array
        
        print("Processes a total of", total_jets, "jets")
        print("Final array contains", len(full_array), "entries")
        print("stored the final array also as class variable ak_array_unrolled")

        return full_array
    
    def unroll_raw_array_cluster(self, ak_array, max_events=-1, verbosity=0):
        # function that creates a new awkward array with one entry per cluster
        # then stores the array in a class variable self.ak_array_unrolled
        print("Turning event array into cluster array")
        list_of_columns = [c for c in ak_array[0].fields]
        list_of_entry_dicts = []
        total_cl = 0
        n_events_seen=0
        for event in ak_array:
            n_cl = len(event["cscRechitCluster_match_gLLP"])
            total_cl += n_cl
            for icl in range(n_cl):
            # create temporary dictionary to hold cluster data
                data_dict = {col: event[col] if not col.startswith("cscRechitCluster") else event[col][icl]
                         for col in list_of_columns}
                if verbosity>1:
                    print(data_dict)
                list_of_entry_dicts.append(data_dict)
            n_events_seen+=1
            if(n_events_seen%1000==0):
                print("Processing event n. ", n_events_seen)
            if max_events>0 and n_events_seen>=max_events:
                break
                
        full_array = ak.Array(list_of_entry_dicts)
        self.ak_array_unrolled = full_array
        
        print("Processes a total of", total_cl, "clusters")
        print("Final array contains", len(full_array), "entries")
        print("stored the final array also as class variable ak_array_unrolled")

        return full_array

    def unroll_raw_array_hits(self, ak_array, max_events=-1, verbosity=0):
        # function that creates a new awkward array with one entry per cluster
        # then stores the array in a class variable self.ak_array_unrolled
        print("Turning event array into hits array")
        list_of_columns = [c for c in ak_array[0].fields]
        list_of_entry_dicts = []
        total_cl = 0
        n_events_seen=0
        for event in ak_array:
            n_cl = len(event["cscClusteredRechitsIndex"])
            total_cl += n_cl
            for icl in range(n_cl):
            # create temporary dictionary to hold cluster data
                data_dict = {col: event[col] if not col.startswith("cscClusteredRechits") else event[col][icl]
                         for col in list_of_columns}
                if verbosity>1:
                    print(data_dict)
                list_of_entry_dicts.append(data_dict)
            n_events_seen+=1
            if(n_events_seen%1000==0):
                print("Processing event n. ", n_events_seen)
            if max_events>0 and n_events_seen>=max_events:
                break
                
        full_array = ak.Array(list_of_entry_dicts)
        self.ak_array_unrolled = full_array
        
        print("Processes a total of", total_cl, "rechits")
        print("Final array contains", len(full_array), "entries")
        print("stored the final array also as class variable ak_array_unrolled")

        return full_array
    
    def vectorize_and_pad_raw_array_jets(self, ak_array, max_jets=10,  verbosity=0):
        # function that creates a new awkward array with one column entry per jet for each event
        # and pads jet entrys with 0, if number of jets < max_jets
        # then stores the array in a class variable self.ak_array_unrolled
        """
        ak_array: raw array to be vectorized
        max_jets: maximum number of jets per event, default is 10
        """
        
        list_of_columns = [c for c in ak_array[0].fields]
        padded_arrays = {}
        
        #creates a new array for each column
        for col in list_of_columns:
            the_column = ak_array[col]
            
            if not "Jets/Jets." in col:
                padded_arrays[col] = the_column
                continue
                
            pad_arr = ak.pad_none(the_column, max_jets, clip=True)
            
            #creates padded column for each jet of each "Jets/Jets." key
            for ijet in range(max_jets):
                new_col_name = col+"_"+str(ijet)
                padded_arrays[new_col_name] = ak.fill_none(pad_arr[:,ijet], 0.0)
                
        data_dict = ak.zip(padded_arrays)
        full_array = ak.Array(data_dict)
        self.ak_array_unrolled = full_array
        
        print("Final array contains", len(full_array), "entries")
        print("stored the final array also as class variable ak_array_unrolled")
        return data_dict

    def vectorize_and_pad_raw_array_cluster(self, ak_array, cluster_type, max_cl=10,  verbosity=0):
        # function that creates a new awkward array with one column entry per cluster for each event
        # and pads cluster entrys with 0, if number of clusters < max_cl
        # then stores the array in a class variable self.ak_array_unrolled
        """
        ak_array: raw array to be vectorized
        max_cl: maximum number of clusters per event, default is 10
        """
        
        list_of_columns = [c for c in ak_array[0].fields]
        padded_arrays = {}
        
        #creates a new array for each column
        for col in list_of_columns:
            the_column = ak_array[col]
            
            if not cluster_type in col:
                padded_arrays[col] = the_column
                continue
                
            pad_arr = ak.pad_none(the_column, max_cl, clip=True)
            
            #creates padded column for each jet of each "csc"/"dt" key
            for icl in range(max_cl):
                new_col_name = col+"_"+str(icl)
                padded_arrays[new_col_name] = ak.fill_none(pad_arr[:,icl], 0.0)
                
        data_dict = ak.zip(padded_arrays)
        full_array = ak.Array(data_dict)
        self.ak_array_unrolled = full_array
        
        print("Final array contains", len(full_array), "entries")
        print("stored the final array also as class variable ak_array_unrolled")
        return data_dict

    def vectorize_and_pad_raw_array_hits(self, ak_array, cluster_type, max_cl=1000,  verbosity=0):
        # function that creates a new awkward array with one column entry per cluster for each event
        # and pads cluster entrys with 0, if number of clusters < max_cl
        # then stores the array in a class variable self.ak_array_unrolled
        """
        ak_array: raw array to be vectorized
        max_cl: maximum number of clusters per event, default is 10
        """
        
        list_of_columns = [c for c in ak_array[0].fields]
        padded_arrays = {}
        
        #creates a new array for each column
        for col in list_of_columns:
            the_column = ak_array[col]
            
            if not cluster_type in col:
                padded_arrays[col] = the_column
                continue
                
            pad_arr = ak.pad_none(the_column, max_cl, clip=True)
            
            #creates padded column for each jet of each "csc"/"dt" key
            for icl in range(max_cl):
                new_col_name = col+"_"+str(icl)
                padded_arrays[new_col_name] = ak.fill_none(pad_arr[:,icl], 0.0)
                
        data_dict = ak.zip(padded_arrays)
        full_array = ak.Array(data_dict)
        self.ak_array_unrolled = full_array
        
        print("Final array contains", len(full_array), "entries")
        print("stored the final array also as class variable ak_array_unrolled")
        return data_dict

    def unroll_clusters_vectorize_and_pad_hits(self, ak_array, cluster_type, hits_type, default_values_rechits, max_hits=1000, max_events=-1, verbosity=0):
        # function that creates a new awkward array with one entry per cluster
        # then creates a new awkward array with one column entry per hits for each cluster
        # and pads hits entries with the appropriate default values, if number of hits < max_hits
        # then stores the array in a class variable self.ak_array_unrolled
        list_of_columns = [c for c in ak_array[0].fields]
        padded_arrays_hits = {}
    
        #here we need to separate cluster variables (unrolled) and rechits (vectorized/padded)
        list_of_cl_columns = []
        list_of_hits_columns = []
        for col in list_of_columns:
            if "cscClusteredRechits" in col:
                list_of_hits_columns.append(col)
            else:
                list_of_cl_columns.append(col)
                
        list_of_entry_dicts = []
        total_cl = 0
        n_events_seen=0
        for event in ak_array:
            n_cl = len(event["cscRechitCluster_match_gLLP"])
            total_cl += n_cl
            hits_event = event[list_of_hits_columns]
            cl_event = event[list_of_cl_columns]
            for icl in range(n_cl):
                # create temporary dictionary to hold cluster data
                data_dict_cl = {col: cl_event[col] if not col.startswith(cluster_type) else cl_event[col][icl]
                             for col in list_of_cl_columns}
                if verbosity>1:
                    print(data_dict_cl)
                m_hits = event["cscClusteredRechitsIndex"]==icl
                for col in list_of_hits_columns:
                    filtered_hits = ak.Array(hits_event[col][m_hits])
                    pad_arr = ak.pad_none(filtered_hits, max_hits, axis=0, clip=True)
                    for ih in range(max_hits):
                        new_col_name = col+"_"+str(ih)
                        addendum = ak.fill_none([ pad_arr[ih] ], default_values_rechits[col])[0]
                        data_dict_cl[new_col_name] = addendum
            
                list_of_entry_dicts.append(data_dict_cl)
            
            n_events_seen+=1
            if(n_events_seen%1000==0):
                print("Processing event n. ", n_events_seen)
            if max_events>0 and n_events_seen>=max_events:
                break
    
    
        full_array = ak.Array(list_of_entry_dicts)
        self.ak_array_unrolled = full_array
        return full_array
        
        print("Final array contains", len(full_array), "entries")
        print("stored the final array also as class variable ak_array_unrolled")
        
    
    def convert_ak_to_numpy(self, ak_array=None):
        # convert the unrolled ak array to a structured numpy array
        # then store this numpy array in a class variable
              
        # If an array is passed as arguments it will use this.
        # If not it will check if an array is stored as class variable and use this
        # Else it will return None
        if isinstance(ak_array, ak.Array):
            np_array=ak.to_numpy(self.ak_array_unrolled)
            self.np_array_structured = np_array
            print("Converted the ak_array.")
            print("Also stored it in self.np_array_structured")
            return np_array
        elif isinstance(self.ak_array_unrolled, ak.Array):
            np_array=ak.to_numpy(self.ak_array_unrolled)
            self.np_array_structured = np_array
            print("Converted the ak_array.")
            print("Also stored it in self.np_array_structured")
            return np_array
        else:
              print("ERROR: Did not have an array to convert or array was wrong type! Need ak.Array")
              return None
        
    def get_structured_numpy(self):
        # return the structred numpy array
        return self.np_array_structured
              
    def get_unstructured_numpy(self, np_array_structured=None):
        # Convert the structured array to an unstructured one
        # Again allow passing or using of stored structured array
        if isinstance(np_array_structured, np.ndarray):
            self.np_array_unstructured = np.lib.recfunctions.structured_to_unstructured(np_array_structured)
            return self.np_array_unstructured
        elif isinstance(self.np_array_structured, np.ndarray):
            self.np_array_unstructured = np.lib.recfunctions.structured_to_unstructured(self.np_array_structured)
            return self.np_array_unstructured
        else:
            print("Do not have an array to convert or array was of wrong type! Need numpy.ndarray!")
            return None
              
    def save(self, path, input_array=None, mode="structured"):
        # save numpy array to h5 file
        # We will check for several cases:
        # - An array is passed as argument or not
        # If not, do we want to save the structured or unstructured array? It is better to save the structured one, by the way
        the_array = None
        if isinstance(input_array, np.ndarray):
            the_array = input_array
        elif mode=="structured" and isinstance(self.np_array_structured, np.ndarray):
            the_array = self.np_array_structured
        elif mode=="unstructured" and isinstance(self.np_array_unstructured, np.ndarray):
            the_array = self.np_array_unstructured
        else:
            print("Error: Have no array or array was of wrong type or do not know the mode!")
            print("Need numpy numpy.ndarray")
            print("Doing nothing!")
        if isinstance(the_array, np.ndarray):
            # check if path exists if not create directory. Taken idea from above
            # first split path into directory and filename
            # see also https://docs.python.org/3/library/os.path.html
            split_path = os.path.split(path)
            # now if head part is not empty check if it exists and if not create the directory
            if split_path[0] != "":
                if not os.path.exists(split_path[0]):
                    os.makedirs(split_path[0], exist_ok=True)
            # now it should work to write the file        
            with h5py.File(path, 'w') as f:
                written_dataset = f.create_dataset('data', data=the_array)
            print(f)    
         
    def load(self, path): 
        # load numpy array from h5 file and store it in class variable
        with h5py.File(path, 'r') as f:
            np_array = np.array(f['data'])
            print("Loaded array")
            # we also want to store the array in the approproate class variable
            # but first we need to figure out if it is structured or unstructured
            # we can do this by examining the array.dtype.names property
            if np_array.dtype.names!=None:
                # is structured!
                print("Array is structured. Saved in self.np_array_structured")
                self.np_array_structured = np_array                
            else:
                print("Array is structured. Saved in self.np_array_structured")
                self.np_array_unstructured = np_array
            return np_array
                
    def load_df(self, path): 
        # load pandas df from h5 file and store it in class variable
        store = pd.HDFStore(path)
        #if store.keys() ==[]:
        #    print("Warning, empty dataframe, skipping...")
        #    continue
        #else:
        print('Opening file ', path)
        df = store.select("df",start=0,stop=-1)
        count_events = df.shape[0]#store.get_storer('df').shape
        store.close()
        return df     
        
         
        
