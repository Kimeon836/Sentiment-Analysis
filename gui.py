# Importing all neccessary libraries
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfile
from sa import SA
import csv
import os

# Initializing tkinter window
root = tk.Tk()
root.title("Sentiment Analysis")
root.geometry("900x600")

# Creating object of SA class from sa.py 
AI = SA()
# Loading module
AI.load_saved_model("./models/my_model_2")

# Can train model like this
# AI.train_model("./models/my_ai")

# Getting index of results csv file to avoid overwriting previous files
curr_idx = 0
for i in os.listdir("./results"):
    if i.startswith("results"):
        idx = int(i[8:-4])
        curr_idx = (idx + 1) if curr_idx <= idx else curr_idx

# Making csv file with two columns, here all the reviews will be stored along with its sentiment
with open(f"./results/results_{curr_idx}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Reviews", "Type"])
        
# Clears the input box at first click to remove default text
def clear_entrybox_onclick(event):
   entry_box.configure(state=tk.NORMAL)
   entry_box.delete(0, tk.END)
   entry_box.unbind('<Button-1>', clicked)

# It updates the stats, adds the new entry in panel and also saves it in results.csv
def manage_review(review, type_):
    if len(review) > 93: temp = review[:91] + "..."
    else: temp = review

    table.insert(parent='',index=0,text='', values=(temp, type_))
    
    tot_p = int(tot.get()[15:])
    pos_p = (float(pos.get()[17:-1])/100)*tot_p
    neg_p = (float(neg.get()[17:-1])/100)*tot_p
    neu_p = (float(neu.get()[16:-1])/100)*tot_p
    tot_p += 1

    if type_.lower() in pos.get().lower():
        pos_p += 1
    elif type_.lower() in neu.get().lower():
        neu_p += 1
    else:
        neg_p += 1

    neu_p = (neu_p * 100)/tot_p
    neu.set(f"Neutral Review: {round(neu_p, 3)}%")
    neg_p = (neg_p * 100)/tot_p
    neg.set(f"Negative Review: {round(neg_p, 3)}%")
    pos_p = (pos_p * 100)/tot_p

    pos.set(f"Positive Review: {round(pos_p, 3)}%")

    tot.set(f"Total Reviews: {tot_p}")

    with open(f"./results/results_{curr_idx}.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow([review, type_])

# Triggered when user clicks submit, uses sa module to to predict review and then calls manage_review() function
def submit(event=None):
    review = entry_var.get()
    _, type_ = AI.predict(review)
    manage_review(review, type_)
    entry_var.set("")

# Triggered when user clicks open file btn, takes the file path from user and uses sa model function to predict the reviews
def from_file():
    file_path = askopenfile(mode='r', filetypes=[('text files', '*txt'), ('csv files', '*csv')])
    if file_path == None:
        return None
    file_path = file_path.name
    db = AI.predict_from_file(file_path=file_path)
    
    for i in db:
        review, _, type_ = i
        manage_review(review, type_)
        
# Declaring and initializing all variables for stats
pos = tk.StringVar()
neg = tk.StringVar()
neu = tk.StringVar()
tot = tk.StringVar()
pos.set("Positive Review: 0%")
neu.set("Neutral Review: 0%")
neg.set("Negative Review: 0%")
tot.set("Total Reviews: 0")

# Entry widget to take user input
entry_var = tk.StringVar()
entry_box = tk.Entry(root, textvariable = entry_var, font=('calibre', 10, 'normal'), width=80)
entry_var.set('Enter review here')
entry_box.grid(row=0,column=0, padx=10, pady=10)

# Binding mouse button with entry widget
clicked = entry_box.bind('<Button-1>', clear_entrybox_onclick)
entry_box.bind('<Return>', submit)

# Submit button
submit_button = tk.Button(root, text ="Submit", command = submit)
submit_button.grid(row=0,column=1)

# Open file button
file_submission = tk.Button(root, text ="Open file", command = from_file)
file_submission.grid(row=0,column=2, padx=10)

# All stats
stats_pos = tk.Label(textvariable=pos)
stats_neu = tk.Label(textvariable=neu)
stats_neg = tk.Label(textvariable=neg)
stats_total = tk.Label(textvariable=tot)

# Posititing stats
stats_pos.grid(row=1, column=0, sticky="w", padx=25, pady=30)
stats_neu.grid(row=1, column=1, sticky="w")
stats_neg.grid(row=1, column=2, sticky="w")
stats_total.grid(row=1, column=3, sticky="w")
stats_neu.place(x=260, y=74)
stats_neg.place(x=500, y=74)
stats_total.place(x=700, y=74)

# Adding scrollbar to panel where all history will be shown
scroll_bar_y = tk.Scrollbar(root)
scroll_bar_y.grid(row=2,column=3, sticky='ns', padx=0)

# Panel where new reviews will be logged
table = ttk.Treeview(root, yscrollcommand=scroll_bar_y.set, height=20)
scroll_bar_y.configure(command=table.yview)

# Setting up panel by adding column names
table.column("#0", width=0,  stretch=tk.NO)
table['columns'] = ('Reviews', 'Type')
table.column("Reviews",anchor=tk.CENTER,stretch=tk.NO, width=600)
table.column("Type",anchor=tk.CENTER,stretch=tk.NO,width=250)

# Position columns
table.heading("#0",text="",anchor=tk.CENTER)
table.heading("Reviews",text="Review",anchor=tk.CENTER)
table.heading("Type",text="Type",anchor=tk.CENTER)
table.grid(row=2, column=0, columnspan=3, padx=10)

# Runs the block when program is directly invoked
if __name__ == "__main__":
    root.mainloop()