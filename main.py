import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io
import PIL
from PIL import ImageDraw
from map import initial_map

def load_from_file(filename):
    """
    Function to load a map from a matlab file.

    Params:
        filename: string of filename
    Returns:
        map: 2d numpy array of 0s and 1s and 2 (goal)
    """
    mat = scipy.io.loadmat(filename)
    # print(mat)
    return np.array(mat['map'])


def display_map(map=[], trajectory=[]):
    """
    #TODO: Function to display a map and the calculated trajectory.
    (use a more colourful scheme to differentiate between trajectory and map)

    Params:
        map: 2d numpy array of 0s and 1s
        trajectory: list of tuples of (row, col) indices
    """

    # Make color map matrix with same size as map
    color_map = np.zeros((map.shape[0], map.shape[1], 3))
    for row in range(map.shape[0]):
        for col in range(map.shape[1]):
            if map[row, col] == 1:
                color_map[row, col] = [145/255, 145/255, 145/255]
            elif map[row, col] == 2:
                color_map[row, col] = [0, 1, 0]
            else:
                color_map[row, col] = [1, 1, 1]
    
    for row, col in trajectory:
        color_map[row, col] = [1, 0, 0]

    table = plt.table(cellText=map,cellColours=color_map, loc=(0,0), cellLoc='center')
    plt.axis('off')
    # plt.show()
    plt.savefig('my_plot.png')

def display_map_using_pillow(map, trajectory=[]):
    # Draw the map
    img = PIL.Image.new('RGB', (map.shape[1], map.shape[0]), color = 'white')
    pixels = img.load()

    for i in range(img.size[0]):    
        for j in range(img.size[1]):    
            if map[j,i] == 1:
                pixels[i,j] = (145, 145, 145)
            elif map[j,i] == 2:
                pixels[i,j] = (0, 255, 0)
            else:
                pixels[i,j] = (255, 255, 255)

    # Draw the trajectory
    for row, col in trajectory:
        pixels[col, row] = (255, 0, 0)

    # Enlarge the image
    img = img.resize((img.size[0]*10, img.size[1]*10), PIL.Image.NEAREST)
    
    img.save('my_plot.png')

def print_output(value_map=np.array([]), trajectory=[]):
    """
    Function to print the output in the required format.

    Params:
        value_map: 2d numpy array of values
        trajectory: list of tuples of (row, col) indices

    """

    # creates a formatted string of the value map and trajectory
    # prints it to the console and saves it to a file

    # loop through the value map and adding to string
    str_maze = ""
    str_maze += "value_map = \n \n"
    for row in range(value_map.shape[0]):
        for item in range(value_map.shape[1]):
            # left aligned formatting
            str_maze += "{:4}".format(value_map[row, item])

        str_maze += "\n"

    str_maze += "\n\n"

    # looping through the trajectory and adding it to the string
    str_trajectory = ""
    str_trajectory += "trajectory = \n \n"

    for row in trajectory:
        for item in row:
            # left aligned formatting
            str_trajectory += "{:4}".format(item)
        str_trajectory += "\n"

    str_out = str_maze + str_trajectory

    # checks if the value map is very large and only prints it in file if it is
    if (np.size(value_map) > 1000):
        print("Output too large to print to console. Saved to output.txt")
        print("")
        with open("output.txt", "w") as text_file:
            text_file.write(str_out)

        # prints the trajectory to the console regardless..
        print(str_trajectory)

    else:
        print(str_out)

        print("Output saved to output.txt")


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

    # PLACEHOLDERS, delete and replace with code
    value_map = map
    trajectory = []

    # add code here :)

    # find the goal location (search for 2)

    # check for a valid goal location (cant be on an obstacle)

    # if valid goal location, calculate the trajectory somehow

    # else print an error message

    # return the value map and trajectory in required format

    return value_map, trajectory


def generate_random_map():
    """
    #TODO: Function to generate a random map. for testing purposes. 
    """
    map = np.array([])

    # add code here :)

    return map


def main_loop():
    """
    Main loop to get user input and call the required functions.
    """

    print("Starting user input loop...")
    while True:
        print("Select an option:")
        print("1. Load map from file")
        print("2. Generate random map")
        print("0. Exit")
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

        elif option == "0":
            break

        else:
            print("Invalid option. Try again.")

        print("")
    pass


def debug_loop():
    matrix = load_from_file("maze.mat")
    print_output(value_map=matrix, trajectory=[])
    trajectory = [[8,2],[8,3]]
    display_map_using_pillow(np.array(matrix),trajectory)

main_loop()
# debug_loop()
