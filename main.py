from sa import SA

if __name__ == "__main__":
    my_ai = SA()
    my_ai.load_saved_model("./models/my_model_2")

    temp = my_ai.predict_from_file("./new.txt")
    print()
    print(*[f"Review: {i[0][:100]} ------ {i[2]}\n" for i in temp])
    print("\n\n\n")

    my_ai.model_details()
    print("\n\n")

    while True:
        b, c = my_ai.predict(input("Give a review: "))
        print(c)
    