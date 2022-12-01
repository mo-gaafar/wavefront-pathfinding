import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io


def load_from_file(filename):
    """
    #TODO: Function to load a map from a matlab file.

    Params:
        filename: string of filename
    Returns:
        map: 2d numpy array of 0s and 1s
    """
    mat = scipy.io.loadmat(filename)
    return mat['map']


def display_map(map, trajectory=[]):
    """
    #TODO: Function to display a map and the calculated trajectory.
    (use a more colourful scheme to differentiate between trajectory and map)

    Params:
        map: 2d numpy array of 0s and 1s
    """

    # add code here

    pass


def print_output(value_map, trajectory):
    """
    #TODO: Function to print the output in the required format.
    """

    # add code here

    pass


def wavefront_map(map, goal_row, goal_col):
    """
    #TODO: Function to calculate the wavefront map from a given goal location.
    """

    # add code here
    value_map = []

    return value_map


def planner(map, start_row, start_col):
    """
    #TODO: Plans a path from start to goal on a given 2d map, uses wavefront map.

    Params:
        map: 2d numpy array of 0s and 1s
        start_row: row index of start
        start_col: column index of start

    Returns:
        value_map: 2d numpy array of values
        trajectory: list of tuples of (row, col) indices
    """
    value_map = []
    trajectory = []

    # add code here :)

    # find the goal location using a search algorithm (search for 2)

    # check for a valid goal location (cant be on an obstacle)

    # if valid goal location, calculate the trajectory somehow

    # else print an error message

    # return the value map and trajectory in required format

    return value_map, trajectory


def generate_random_map():
    """
    #TODO: Function to generate a random map. for testing purposes. 
    """

    # add code here :)

    return map


def main_loop():
    """
    TODO: Main loop to get user input and call the required functions.
    """

    print("Starting user input loop...")
    while True:
        print("Select an option:")
        print("1. Load map from file")
        print("2. Generate random map")
        print("3. Exit")
        option = input("Enter option: ")

        if option == "1":
            filename = input("Enter filename: ")
            map = load_from_file(filename)
            start_row = int(input("Enter start row: "))
            start_col = int(input("Enter start col: "))
            value_map, trajectory = planner(map, start_row, start_col)
            display_map(value_map, trajectory)
            print_output(value_map, trajectory)

        elif option == "2":
            map = generate_random_map()
            start_row = int(input("Enter start row: "))
            start_col = int(input("Enter start col: "))
            value_map, trajectory = planner(map, start_row, start_col)
            display_map(value_map, trajectory)
            print_output(value_map, trajectory)

        elif option == "3":
            break

        else:
            print("Invalid option. Try again.")

        print("")
    pass


main_loop()
