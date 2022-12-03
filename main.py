import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io
import PIL
from PIL import ImageDraw

'''
Developed by:
    Aly Khaled
    Mariam Aly
    Maryam Moataz
    Mohamed Nasser
'''

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


def display_map(map=[], trajectory=[], start_row=0, start_col=0):
    """
    Function to display a map and the calculated trajectory.

    Note:
        Saves the image to my_plot.png

    Params:
        map: 2d numpy array of 0s and 1s
        trajectory: list of tuples of (row, col) indices
    """

    map = np.array(map)

    if map.size == 0:
        return

    if (np.size(map) > 600):
        # print("Map too large to display. Saved to my_plot.png")
        # display map without numbering
        display_map_using_pillow(map, trajectory, start_row, start_col)
        return

    # Make color map matrix with same size as map
    color_map = np.zeros((map.shape[0], map.shape[1], 3))
    for row in range(map.shape[0]):
        for col in range(map.shape[1]):
            if map[row, col] == 1:  # if the pixel is an obstacle colour it grey
                color_map[row, col] = [145/255, 145/255, 145/255]
            elif map[row, col] == 2:  # if the pixel is the goal colour it green
                color_map[row, col] = [0, 1, 0]
            else:  # if the pixel is a free space colour it white
                color_map[row, col] = [1, 1, 1]

    # Draw the trajectory
    for row, col in trajectory:
        color_map[row, col] = [1, 0, 0] if map[row, col] != 2 else [0, 1, 0]

    color_map[start_row, start_col] = [0, 0, 1]
    # Plot the map as a table
    table = plt.table(cellText=map, cellColours=color_map,
                      loc=(0, 0), cellLoc='center')
    plt.axis('off')
    plt.savefig('my_plot.png')
    plt.show()

    print("Map saved to my_plot.png")


def display_map_using_pillow(map, trajectory=[], start_row=0, start_col=0):
    """
    Function to display a map and the calculated trajectory as an image instead of matplotlib.

    Note:
        Saves the image to my_plot.png
    Params:
        map: 2d numpy array of the map
        trajectory: list of tuples of (row, col) indices
    """

    # Draw the map
    img = PIL.Image.new('RGB', (map.shape[1], map.shape[0]), color='white')
    pixels = img.load()

    for i in range(img.size[0]):
        for j in range(img.size[1]):
            if map[j, i] == 1:  # if the pixel is an obstacle colour it grey
                pixels[i, j] = (145, 145, 145)
            elif map[j, i] == 2:  # if the pixel is the goal colour it green
                pixels[i, j] = (0, 255, 0)
            else:  # if the pixel is a free space colour it white
                pixels[i, j] = (255, 255, 255)

    # Draw the trajectory
    for row, col in trajectory[:-1]:
        pixels[col, row] = (255, 0, 0)

    pixels[start_col, start_row] = (0, 0, 255)
    # Enlarge the image nearest neighbour
    img = img.resize((img.size[0]*10, img.size[1]*10), PIL.Image.NEAREST)

    # Save the image
    img.save('my_plot.png')

    print("Map saved to my_plot.png")


def print_output(value_map=np.array([]), trajectory=[]):
    """
    Function to print the output in the required format.

    Description:
        creates a formatted string of the value map and trajectory
        prints it to the console and saves it to a text file "output.txt"

    Params:
        value_map: 2d numpy array of values
        trajectory: list of tuples of (row, col) indices

    """

    # loop through the value map and adding to string
    value_map = np.array(value_map)
    trajectory = np.array(trajectory)

    if value_map.size == 0 or trajectory.size == 0:
        print("No value map or trajectory to print")
        return

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


def find_goal_coordinate(map):
    """
    Function to find the goal coordinate in the map.

    Params:
        map: 2d numpy array of 0s and 1s
    Returns:
        goal_coordinate: tuple of (row, col) indices
    """
    map = np.array(map)

    for row in range(map.shape[0]):
        for col in range(map.shape[1]):
            if map[row, col] == 2:
                return (row, col)

    return None


def wavefront_map(map, goal_row, goal_col):
    """
    Function to calculate the wavefront map from a given goal location.

    Params:
        map: 2d numpy array of 0s and 1s
        goal_row: int of goal row index
        goal_col: int of goal col index
    Returns:
        map: 2d numpy array of the wavefront map
    """

    moves = [[0, 1], [0, -1], [1, 0], [-1, 0],
             [1, 1], [1, -1], [-1, 1], [-1, -1]]

    queue = []

    # Start from the goal location (row,col,prevValue)
    queue.append([goal_row, goal_col, 2])

    while queue:
        current_row, current_col, prev_value = queue.pop(0)
        for move in moves:
            new_row = current_row + move[0]
            new_col = current_col + move[1]

            # Check if the new location is not 1 or already filled with a value (not 0) and is within the map
            if new_row < 0 or new_row >= len(map) or new_col < 0 or new_col >= len(map[0]) or map[new_row][new_col] == 1 or map[new_row][new_col] != 0:
                continue

            # Add the new location to the queue
            queue.append([new_row, new_col, prev_value+1])
            # Fill the new location with the value of the previous location + 1
            map[new_row][new_col] = prev_value+1

    return map


def planner(map, start_row, start_col):
    """
    Plans a path from start to goal on a given 2d map, uses wavefront map and backtracking functions.

    Params:
        map: 2d numpy array of 0s and 1s
        start_row: row index of start
        start_col: column index of start

    Returns:
        value_map: 2d numpy array of values
        trajectory: list of tuples of (row, col) indices
    """

    value_map = [[]]
    trajectory = [[]]

    #! shift the starting location by 1 to make it one indexed as in requirements pdf
    start_row += 1
    start_col += 1

    # check for a valid start location (cant be on an obstacle, out of bounds, or the goal)
    if start_row < 0 or start_row >= len(map) or start_col < 0 or start_col >= len(map[0]):
        print("Invalid start location")
        return value_map, trajectory
    if map[start_row][start_col] == 1:
        print("Invalid start location")
        return value_map, trajectory
    if map[start_row][start_col] == 2:
        print("Invalid start location")
        return value_map, trajectory

    # find the goal location (search for 2)
    row, col = find_goal_coordinate(map)

    # if valid goal location, calculate the trajectory and value map

    if row != None and col != None:
        # calculate the wavefront map
        value_map = wavefront_map(map, row, col)

        # find the trajectory
        trajectory = backtracking(value_map, start_row, start_col)

    else:
        print("No goal location found")

    return value_map, trajectory


def backtracking(map, start_row, start_col):
    '''
    Function to calculate the shortest path to goal on wavefront matrix from a given start location.
    '''

    # relative neighbor moves are ordered by priority
    # moves = [[-1, 0], [0, 1], [1, 0], [0, -1], [-1, 1], [1, 1], [1, -1], [-1, -1]]

    # reversed priority moves
    moves = [[-1, -1], [1, -1], [1, 1], [-1, 1],
             [0,  -1], [1,  0], [0, 1], [-1, 0]]

    # priority: up, right, down, left, upper right, lower right, lower left, upper left. respectively

    # image coordinate system is flipped from the cartesian coordinate system

    # (x,y)
    # ------------------------------
    # - (0,0) (0,1) (0,2) (0,3) (0,4)
    # - (1,0) (1,1) (1,2) (1,3) (1,4)
    # - (2,0) (2,1) (2,2) (2,3) (2,4)
    # - (3,0) (3,1) (3,2) (3,3) (3,4)
    # - (4,0) (4,1) (4,2) (4,3) (4,4)

    trajectory = []
    # check if the starting location has no solution
    if map[start_row][start_col] == 0:
        print("No Solution")
        return trajectory
    current_row, current_col = start_row, start_col

    # add the starting location to the trajectory
    trajectory.append((current_row, current_col))
    # get the value of the starting location
    current_value = map[current_row][current_col]

    while current_value != 2:  # while the current location is not the goal location
        # get the value of the current location
        current_value = map[current_row][current_col]
        minRow = current_row
        minCol = current_col
        for move in moves:  # for each neighbor
            new_row = current_row + move[0]  # get the new row
            new_col = current_col + move[1]  # get the new col

            if new_row < 0 or new_row >= len(map) or new_col < 0 or new_col >= len(map[0]) or map[new_row][new_col] == 1:
                continue  # if the new location is out of bounds or an obstacle, skip it

            # if the current value is greater than or equal to the new value assign the new value to the current value
            if current_value >= map[new_row][new_col]:
                current_value = map[new_row][new_col]
                minRow, minCol = new_row, new_col  # assign the new location to the min location

        # make the current location be the location with the lowest value
        current_row, current_col = minRow, minCol
        # add the current location to the trajectory
        trajectory.append((current_row, current_col))

    return trajectory


def generate_random_map(rows, cols):
    """
    Function to generate a random map. for testing purposes. 
    """
    map = np.array([])
    map = np.random.randint(0, 2, (rows, cols))

    # set the goal location
    map[rows-1][cols-1] = 2

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
        print("3. Use hardcoded map")
        print("0. Exit")
        option = input("Enter option: ")

        if option == "1":

            # get the file name
            filename = input("Enter filename: ")

            # load the map from the file
            map = load_from_file(filename)

            # allocate 16 bits to avoid overflow
            map = np.array(map)
            map = map.astype(np.uint16)

            print(map.shape)
            # get user input for start location
            start_row = int(input("Enter start row: "))
            start_col = int(input("Enter start col: "))

            # call the planning function
            value_map, trajectory = planner(map, start_row, start_col)

            # print the output
            print_output(value_map, trajectory)
            display_map(value_map, trajectory, start_row, start_col)

        elif option == "2":
            # get the size of random map
            rows = int(input("Enter map rows size: "))
            cols = int(input("Enter map columns size: "))

            map = generate_random_map(rows, cols)

            # allocate 16 bits to avoid overflow
            map = np.array(map)
            map = map.astype(np.uint16)

            # get the start location
            start_row = int(input("Enter start row: "))
            start_col = int(input("Enter start col: "))

            # call the planner
            value_map, trajectory = planner(map, start_row, start_col)

            # keep trying to find a valid start location until a valid one is found
            while trajectory == [[]] or trajectory == []:
                map = generate_random_map(rows, cols)
                map = np.array(map)
                map = map.astype(np.uint16)
                value_map, trajectory = planner(map, start_row, start_col)

            # print the output
            print_output(value_map, trajectory)
            display_map(value_map, trajectory, start_row+1, start_col+1)

        elif option == "3":
            # use hardcoded map for testing
            map =  [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 1, 1, 0 ,0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 ,1, 0, 0, 0, 1],
                [1 ,0, 0, 0, 0, 0, 0, 0 ,0, 0 ,0 ,0, 0, 0, 1, 1, 0, 0 ,0, 1],
                [1, 0 ,0 ,0 ,1, 1, 1, 1, 1, 0 ,0 ,0 ,0, 0, 1 ,1, 0, 0, 0, 1],
                [1, 0 ,0, 0, 1, 1, 1, 1, 1, 0 ,0 ,0 ,0, 0, 0 ,0, 0, 0 ,0, 1],
                [1, 0, 0, 0, 0, 0, 0, 1, 1, 0 ,0 ,0, 0 ,0, 0 ,0, 0, 0, 0, 1],
                [1, 0 ,0, 0 ,0, 0 ,0 ,1 ,1, 0 ,0 ,0 ,0 ,0, 0 ,0, 0 ,0 ,0, 1],
                [1, 0, 0, 0 ,0, 0, 0, 0, 0, 0 ,0, 0 ,0 ,0 ,1 ,0, 0, 0 ,0, 1],
                [1, 0 ,0, 0 ,0, 0, 0 ,0, 0 ,0, 0, 0, 0, 1 ,1 ,0, 0 ,0, 0 ,1],
                [1, 0, 0, 0 ,0, 0, 0, 0, 0 ,0 ,0, 0 ,1 ,1, 1 ,0, 0, 0, 0 ,1],
                [1 ,1 ,1, 1 ,1 ,1, 1, 1, 1 ,1 ,1 ,1 ,1, 1 ,1 ,1, 1, 1, 1 ,1]
            ]

            # allocate 16 bits to avoid overflow
            map = np.array(map)
            map = map.astype(np.uint16)

            # get the start location
            start_row = int(input("Enter start row: "))
            start_col = int(input("Enter start col: "))

            # call the planner
            value_map, trajectory = planner(map, start_row, start_col)

            # print the output
            print_output(value_map, trajectory)
            display_map(value_map, trajectory, start_row, start_col)

        elif option == "0":
            break

        else:
            print("Invalid option. Try again.")

        print("")
    pass


main_loop()
