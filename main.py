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
    map = np.array(map)

    if (np.size(map) > 200):
        # print("Map too large to display. Saved to my_plot.png")
        # display map without numbering
        display_map_using_pillow(map, trajectory)
        return

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

    table = plt.table(cellText=map, cellColours=color_map,
                      loc=(0, 0), cellLoc='center')
    plt.axis('off')
    # plt.show()
    plt.savefig('my_plot.png')


def display_map_using_pillow(map, trajectory=[]):
    # Draw the map
    img = PIL.Image.new('RGB', (map.shape[1], map.shape[0]), color='white')
    pixels = img.load()

    for i in range(img.size[0]):
        for j in range(img.size[1]):
            if map[j, i] == 1:
                pixels[i, j] = (145, 145, 145)
            elif map[j, i] == 2:
                pixels[i, j] = (0, 255, 0)
            else:
                pixels[i, j] = (255, 255, 255)

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
    value_map = np.array(value_map)
    trajectory = np.array(trajectory)

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


def helper(map, prevValue, currentRow, currentCol):

    # if map[currentRow][currentCol] == 1:
    #     return

    # if map[currentRow][currentCol] != 2:
    #     map[currentRow][currentCol] = prevValue

    # moves = [[0, 1], [0, -1], [1, 0], [-1, 0],
    #          [1, 1], [1, -1], [-1, 1], [-1, -1]]

    # queue = []
    # queue.append
    pass


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
            if map[new_row][new_col] == 1 or map[new_row][new_col] != 0 or new_row < 0 or new_row >= len(map) or new_col < 0 or new_col >= len(map[0]):
                continue

            # Add the new location to the queue
            queue.append([new_row, new_col, prev_value+1])
            # Fill the new location with the value of the previous location + 1
            map[new_row][new_col] = prev_value+1

    return map


def planner(map, start_row, start_col):
    """
    Plans a path from start to goal on a given 2d map, uses wavefront map.

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

    # find the goal location (search for 2)
    row, col = find_goal_coordinate(map)

    #! shift the starting location by 1 to make it one indexed as in requirements pdf
    start_row += 1
    start_col += 1

    # check for a valid goal location (cant be on an obstacle)

    value_map = wavefront_map(map, row, col)

    trajectory = backtracking(value_map, start_row, start_col)

    # if valid goal location, calculate the trajectory somehow

    # else print an error message

    # return the value map and trajectory in required format

    return value_map, trajectory


def backtracking(map, start_row, start_col):
    '''
    Function to calculate the shortest path from a given start location.
    '''

    #! add to pdf report

    # relative neighbor moves are ordered by priority
    # moves = [[-1, 0], [0, 1], [1, 0], [0, -1], [-1, 1], [1, 1], [1, -1], [-1, -1]]
    moves = [[-1, -1], [1, -1], [1, 1], [-1, 1],
             [0, -1], [1, 0], [0, 1], [-1, 0]]

    # priority: up, right, down, left, upper right, lower right, lower left, upper left. respectively

    # ------------------------------
    # - (0,0) (0,1) (0,2) (0,3) (0,4)c
    # - (1,0) (1,1) (1,2) (1,3) (1,4)
    # - (2,0) (2,1) (2,2) (2,3) (2,4)
    # - (3,0) (3,1) (3,2) (3,3) (3,4)
    # - (4,0) (4,1) (4,2) (4,3) (4,4)
    #! inshallah while loop will work

    current_row, current_col = start_row, start_col

    trajectory = []
    trajectory.append((current_row, current_col))
    current_value = map[current_row][current_col]
    while True:
        if current_value == 2:
            break
        current_value = map[current_row][current_col]
        minRow = current_row
        minCol = current_col
        for move in moves:
            new_row = current_row + move[0]
            new_col = current_col + move[1]

            if new_row < 0 or new_row >= len(map) or new_col < 0 or new_col >= len(map[0]) or map[new_row][new_col] == 1:
                continue

            if current_value >= map[new_row][new_col]:
                current_value = map[new_row][new_col]
                minRow, minCol = new_row, new_col
        current_row, current_col = minRow, minCol
        trajectory.append((current_row, current_col))

    return trajectory

    # get all neighbors and find minimum value

    # neighbors = []
    # idx = 0
    # min_neighbor_val = 9999999999
    # for move in moves:
    #     row = move[0] + current_row
    #     col = move[1] + current_col

    #     if row < 0 or row >= len(map) or col < 0 or col >= len(map[0]):
    #         map_val = 9999999999
    #     else:
    #         map_val = map[row][col]

    #     neighbors.append(map_val)

    #     if min_neighbor_val < neighbors[idx] and neighbors[idx] != 1:
    #         min_neighbor_val = neighbors[idx]

    #     idx += 1

    # # find first occurence of min in list
    # optimimum_min_idx = neighbors.index(min_neighbor_val)
    # print("=====================================")
    # print(optimimum_min_idx)
    # print("=====================================")
    # # update current point and append to trajectory

    # print("=====================================")
    # print(current_row)
    # print("=====================================")
    # current_row += moves[optimimum_min_idx][0]
    # current_col += moves[optimimum_min_idx][1]

    # trajectory.append([current_row, current_col])

    # # terminate after reaching goal
    # if map[current_row][current_col] == 2:
    #     return trajectory


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
    # matrix = load_from_file("maze.mat")
    matrix = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
              [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2, 0, 1],
              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
              [1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
              [1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
              [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
              [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

    trajectory = []
    # matrix = load_from_file("maze.mat")
    matrix = np.array(matrix)
    matrix = matrix.astype(np.uint16)
    # print_output(value_map=matrix, trajectory=trajectory)
    row, col = find_goal_coordinate(matrix)
    value_map, trajectory = planner(matrix, 12, 1)
    #! 13,2 not working so we handle zero indexing here
    # value_map = wavefront_map(matrix, row, col)
    print_output(value_map=value_map, trajectory=trajectory)
    # display_map_using_pillow(np.array(matrix),trajectory)
    display_map_using_pillow(value_map, trajectory)
    # display_map(matrix, trajectory)


# main_loop()
debug_loop()
