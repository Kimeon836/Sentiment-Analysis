"""A Demo Program for Review 2 to showcase the functions in sa.py"""
from sa import SA

if __name__ == "__main__":
    my_ai = SA() # Declaring and Initating object

    #my_ai.train_model("./models/my_model_5") # saving it as my_model_5
    my_ai.load_saved_model("./models/my_model_2") # loading a saved model

    temp = my_ai.predict_from_file("./new.txt") # Predicting values from txt file
    # Some code to print out reviews
    print()
    print(*[f"Review: {i[0][:100]} ------ {i[2]}\n" for i in temp])
    print("\n\n\n")

    # Calling function to know model's details
    my_ai.model_details()
    print("\n\n")

    # Some code to take user input and predict the review
    while True:
        b, c = my_ai.predict(input("Give a review: "))
        print(c)
    