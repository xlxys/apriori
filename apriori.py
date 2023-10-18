
import numpy as np
import pandas as pd
import itertools 
from sklearn.preprocessing import KBinsDiscretizer
from scipy.sparse import csr_matrix

def load_data(path):
    
    data = pd.read_csv(path)

    return data

def process(data):

    # convert data to numpy array
    data = data.to_numpy()

    return data

def process_noneBianry(data):

    # Split nominal and numerical data
    nominal_data = data.select_dtypes(include=['object'])
    numerical_data = data.select_dtypes(include=['int', 'float'])
    
    # One-hot encode nominal data
    nominal_data = pd.get_dummies(nominal_data)

    # Discretize numerical data
    discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    discretized_data = pd.DataFrame(discretizer.fit_transform(numerical_data), columns=numerical_data.columns)

    # Concatenate nominal and numerical data
    processed_data = pd.concat([nominal_data, discretized_data], axis=1)

    # Convert data to binary format
    processed_data = np.where(processed_data > 0, 1, 0)

    return processed_data


def csr(data):
    
    transactions = []
    for row in data:
        row = np.where(row > 0)[0]
        transactions.append(row)

    return transactions



def generate_candidate_itemsets(frequent_itemsets, k):
    
    if k == 2:
        candidate_itemsets = []
        for i in range(len(frequent_itemsets)):
            for j in range(i+1, len(frequent_itemsets)):
                itemset_1 = [frequent_itemsets[i]]
                # print (itemset_1)
                itemset_2 = [frequent_itemsets[j]]
                # print (itemset_2)
                if itemset_1[:k-2] == itemset_2[:k-2]:
                    candidate_itemset = itemset_1 + [itemset_2[-1]]
                    candidate_itemset.sort()
                    candidate_itemsets.append(set(candidate_itemset))
                
    else:
        candidate_itemsets = []
        for i in range(len(frequent_itemsets)):
            for j in range(i+1, len(frequent_itemsets)):
                itemset_1 = frequent_itemsets[i]
                itemset_2 = frequent_itemsets[j]
                candidate_itemset = itemset_1.union(itemset_2)
                if len(candidate_itemset) == k and candidate_itemset not in candidate_itemsets:
                    candidate_itemsets.append(candidate_itemset)

    return candidate_itemsets

def get_frequency(transactions):
    item_counts = {}
    for transaction in transactions:
        for item in transaction:
            if item in item_counts:
                item_counts[item] += 1
            else:
                item_counts[item] = 1
    
    return item_counts


def closed(itemsets, frequency):
    closed_itemsets = []
    for itemset in range(len(itemsets)):
        itemsetlen = len(itemsets[itemset])

        for item in range (len(itemsets[itemset])):
            if itemset+1 < len(itemsets):
                
                for supersets in range(len(itemsets[itemset+1])):
                    if itemsets[itemset][item] in itemsets[itemset+1][supersets]:
                        # print(frequency[item])
                        # print("-------------",frequency[itemsetlen+supersets])
                        if frequency[item] != frequency[itemsetlen+supersets] and itemsets[itemset][item] not in closed_itemsets:
                            closed_itemsets.append(itemsets[itemset][item])

    return closed_itemsets


def association_rules(itemsets):
    # itemsets = itemsets[1:]
    association_rules = []
    for itemset in itemsets:
        for i in range(len(itemset)):

            for lhs in itertools.combinations(itemset, i):
                lhs = set(lhs)
               
                rhs = itemset - lhs
                if lhs and rhs:
                    association_rules.append((lhs, rhs))
    return association_rules


def confidenceAndLift(itemsets,frequent, support, min_confidence):
    rules = association_rules(itemsets)
    rulescl = []
    for rule in rules:
        # lhs_count = 0
        # rhs_count = 0
        # b_count = 0

        lhs = rule[0]
        rhs = rule[1]
        if len(lhs) == 1:
            lhs = list(lhs)[0]
        if len(rhs) == 1:
            rhs = list(rhs)[0]
        

        lhs_index = frequent.index(lhs)
        rhs_index = frequent.index(rhs)

        if isinstance(lhs, set):
            if isinstance(rhs, set):
                b = lhs.union(rhs)
            else:
                b = lhs.union(set([rhs]))
        else:
            if isinstance(rhs, set):
                b = set([lhs]).union(rhs)
            else:
                b = set([lhs]).union(set([rhs]))


        b_index = frequent.index(b)

        lhs_count = support[lhs_index]
        rhs_count = support[rhs_index]
        b_count = support[b_index]



        #TODO: calculate confidence and lift
        confidence = b_count / lhs_count
        lift = b_count / (rhs_count * lhs_count)
        if confidence >= min_confidence:
            rulescl.append((lhs, rhs, confidence, lift))
    return rulescl


def flatten_list(lst):
    flattened = []
    for element in lst:
        if isinstance(element, list):
            flattened.extend(flatten_list(element))
        else:
            flattened.append(element)
    return flattened
        
        


def apriori(transactions, min_support, min_confidence):

    # Counting the frequency of each item in the dataset
    item_counts = get_frequency(transactions)

    # Removing infrequent items
    item_counts = {k: v for k, v in item_counts.items() if v >= min_support}

    # Extracting frequent items and their counts
    frequent_items = list(item_counts.keys())
    frequent_items_counts = list(item_counts.values())

    
    # Finding frequent itemsets
    frequent_itemsets = [frequent_items]
    # print (frequent_itemsets) 

    k = 2
    while True:
        # Generating candidate itemsets
        
        # print(frequent_itemsets[-1])
        candidate_itemsets = generate_candidate_itemsets(frequent_itemsets[-1], k)
        # print(candidate_itemsets)

        # Counting the frequency of each candidate itemset in the dataset
        candidate_itemsets_counts = [0] * len(candidate_itemsets)
        for transaction in transactions:
            for i, itemset in enumerate(candidate_itemsets):
                if set(itemset).issubset(set(transaction)):
                    candidate_itemsets_counts[i] += 1
        # print(candidate_itemsets_counts)

        # Removing infrequent candidate itemsets
        frequent_itemsets_k = []
        frequent_itemsets_counts_k = []
        for i, itemset in enumerate(candidate_itemsets):
            if candidate_itemsets_counts[i] >= min_support:
                frequent_itemsets_k.append(itemset)
                frequent_itemsets_counts_k.append(candidate_itemsets_counts[i])
        
        # print(frequent_itemsets_k)
        if frequent_itemsets_k == []:
            break
        
        frequent_itemsets.append(frequent_itemsets_k)
        frequent_items_counts.append(frequent_itemsets_counts_k)
        # print (frequent_itemsets)

        k += 1

    
    
    # Flattening the list of frequent itemsets and their counts
    frequent_items_counts = flatten_list(frequent_items_counts)

    close = closed(frequent_itemsets, frequent_items_counts)
    # rules = association_rules(flatten_list(frequent_itemsets[1:]))
    rules = confidenceAndLift(flatten_list(frequent_itemsets[1:]),flatten_list(frequent_itemsets), frequent_items_counts, min_confidence)
    
   
    return frequent_itemsets, frequent_items_counts, close, rules


import tkinter as tk
from tkinter import filedialog  
from tkinter import ttk


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        frequent_itemsets = []
        frequent_items_counts = []
        close = []

        self.title("Apriori Algorithm")
        self.geometry("800x600")

        # create a button to select a file
        self.select_button = tk.Button(self, text="Select File", command=self.load_csv)
        self.select_button.pack(pady=20, padx=20)

        # create a button to run the algorithm
        self.run_button = tk.Button(self, text="Run Apriori", command=self.run_apriori)
        self.run_button.pack(pady=20, padx=20)

        # Create a frame for the left side
        self.left_frame = tk.Frame(self, bg="gray", width=400, height=600)
        self.left_frame.pack(side="left")

        # Create a frame for the right side
        self.right_frame = tk.Frame(self, bg="white", width=400, height=600)
        self.right_frame.pack(side="right")

        # create a label to display the CSV file
        self.file_label = tk.Label(self.left_frame, text="")
        self.file_label.pack(fill="both", expand=1)

      
    
    def run_apriori(self):

        

        for widget in self.right_frame.winfo_children():
            widget.destroy()

        global frequent_itemsets, frequent_items_counts, close
        frequent_itemsets, frequent_items_counts, close, rules = apriori(self.transactions, self.support_scale.get(), self.confidence_scale.get())
        # print(frequent_itemsets)
        # print(frequent_items_counts)

        frequent_itemsets = flatten_list(frequent_itemsets)
        frequent_items_counts = flatten_list(frequent_items_counts)

        # print(frequent_itemsets)
        # print(frequent_items_counts)

        self.table = ttk.Treeview(self.right_frame, columns=('frequent','support','closed'), show="headings")
        self.table.heading('frequent', text="frequent itemsets")
        self.table.heading('support', text="support")
        self.table.heading('closed', text="closed")
        self.table.pack(fill="both", expand=1)
        
        for i in range(len(frequent_itemsets)):
            if frequent_itemsets[i] in close:
                self.table.insert(parent='', index='end', values=(frequent_itemsets[i], frequent_items_counts[i], "yes"))
            else:
                self.table.insert(parent='', index='end', values=(frequent_itemsets[i], frequent_items_counts[i], "no"))

        
        self.table2 = ttk.Treeview(self.right_frame, columns=('rules','confidence','lift'), show="headings")
        self.table2.heading('rules', text="rules")
        self.table2.heading('confidence', text="confidence")
        self.table2.heading('lift', text="lift")
        self.table2.pack(fill="both", expand=1)

        for i in range(len(rules)):
            self.table2.insert(parent='', index='end', values=(str(rules[i][0])+"-->"+str(rules[i][1]) , rules[i][2], rules[i][3]))




    def load_csv(self):
        # prompt the user to select a file
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        
        if file_path:

            df = load_data(file_path)

            self.data = process(df)

            self.transactions = csr(self.data)

    

            # create a label and scale for the minimum support
            self.support_label = tk.Label(self.left_frame, text="Minimum Support")
            self.support_label.pack(fill="both", expand=1)
            self.support_scale = tk.Scale(self.left_frame, from_=1, to=len(self.transactions), resolution=1, orient=tk.HORIZONTAL)
            self.support_scale.pack(fill="both", expand=1)

            # create a label and scale for the minimum confidence
            self.confidence_label = tk.Label(self.left_frame, text="Minimum Confidence")
            self.confidence_label.pack(fill="both", expand=1)
            self.confidence_scale = tk.Scale(self.left_frame, from_=0, to=1, resolution=0.1, orient=tk.HORIZONTAL)
            self.confidence_scale.pack(fill="both", expand=1)


            # # create a table widget
            # self.table = tk.Label(self.left_frame, text=df.to_string(index=False), font="none 12")
            # self.table.pack(fill="both", expand=1)


if __name__ == "__main__":
    app = App()
    app.mainloop()
