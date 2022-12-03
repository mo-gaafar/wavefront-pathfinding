import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io
import PIL
from PIL import ImageDraw
from map import initial_map, nosolution, notAllSol, diagonalSol


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
    Function to display a map and the calculated trajectory.
    (use a more colourful scheme to differentiate between trajectory and map)

    Params:
        map: 2d numpy array of 0s and 1s
        trajectory: list of tuples of (row, col) indices
    """

    map = np.array(map)

    if map.size == 0: 
        return

    if (np.size(map) > 500):
        # print("Map too large to display. Saved to my_plot.png")
        # display map without numbering
        display_map_using_pillow(map, trajectory)
        return

    # Make color map matrix with same size as map
    color_map = np.zeros((map.shape[0], map.shape[1], 3))
    for row in range(map.shape[0]):
        for col in range(map.shape[1]):
            if map[row, col] == 1: # if the pixel is an obstacle colour it grey
                color_map[row, col] = [145/255, 145/255, 145/255]
            elif map[row, col] == 2: # if the pixel is the goal colour it green
                color_map[row, col] = [0, 1, 0]
            else: # if the pixel is a free space colour it white
                color_map[row, col] = [1, 1, 1]

    # Draw the trajectory
    for row, col in trajectory:
        color_map[row, col] = [1, 0, 0] if map[row, col] != 2 else [0, 1, 0]

    # Plot the map as a table
    table = plt.table(cellText=map, cellColours=color_map,
                      loc=(0, 0), cellLoc='center')
    plt.axis('off')
    plt.savefig('my_plot.png')
    plt.show()

    print("Map saved to my_plot.png")


def display_map_using_pillow(map, trajectory=[]):
    """
    Function to display a map and the calculated trajectory.
    (use a more colourful scheme to differentiate between trajectory and map)
    
    Params:
        map: 2d numpy array of the map
        trajectory: list of tuples of (row, col) indices
    """

    # Draw the map
    img = PIL.Image.new('RGB', (map.shape[1], map.shape[0]), color='white')
    pixels = img.load()

    for i in range(img.size[0]):
        for j in range(img.size[1]):
            if map[j, i] == 1: # if the pixel is an obstacle colour it grey
                pixels[i, j] = (145, 145, 145) 
            elif map[j, i] == 2: # if the pixel is the goal colour it green
                pixels[i, j] = (0, 255, 0) 
            else: # if the pixel is a free space colour it white
                pixels[i, j] = (255, 255, 255)

    # Draw the trajectory
    for row, col in trajectory[:-1]:
        pixels[col, row] =  (255, 0, 0)
    


    # Enlarge the image nearest neighbour
    img = img.resize((img.size[0]*10, img.size[1]*10), PIL.Image.NEAREST)

    # Save the image
    img.save('my_plot.png')

    print("Map saved to my_plot.png")


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
            if  new_row < 0 or new_row >= len(map) or new_col < 0 or new_col >= len(map[0]) or map[new_row][new_col] == 1 or map[new_row][new_col] != 0:
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
    Function to calculate the shortest path from a given start location.
    '''

    #! add to pdf report

    # relative neighbor moves are ordered by priority
    # moves = [[-1, 0], [0, 1], [1, 0], [0, -1], [-1, 1], [1, 1], [1, -1], [-1, -1]]
    moves = [[-1, -1], [1, -1], [1, 1], [-1, 1],
             [0,  -1], [1,  0], [0, 1], [-1, 0]]

    # priority: up, right, down, left, upper right, lower right, lower left, upper left. respectively

    # ------------------------------
    # - (0,0) (0,1) (0,2) (0,3) (0,4)
    # - (1,0) (1,1) (1,2) (1,3) (1,4)
    # - (2,0) (2,1) (2,2) (2,3) (2,4)
    # - (3,0) (3,1) (3,2) (3,3) (3,4)
    # - (4,0) (4,1) (4,2) (4,3) (4,4)
    # ------------------------------
    # handle edge cases before entering while loop (starting on boundary, starting on obstacle, etc.)

    trajectory = []
    if map[start_row][start_col] == 0:
        print("No Solutin")
        return trajectory
    current_row, current_col = start_row, start_col

    trajectory.append((current_row, current_col)) # add the starting location to the trajectory
    current_value = map[current_row][current_col] # get the value of the starting location

    while current_value != 2: # while the current location is not the goal location
        current_value = map[current_row][current_col] # get the value of the current location
        minRow = current_row 
        minCol = current_col 
        for move in moves: # for each neighbor
            new_row = current_row + move[0] # get the new row
            new_col = current_col + move[1] # get the new col

            if new_row < 0 or new_row >= len(map) or new_col < 0 or new_col >= len(map[0]) or map[new_row][new_col] == 1:
                continue # if the new location is out of bounds or an obstacle, skip it

            if current_value >= map[new_row][new_col]: # if the current value is greater than or equal to the new value assign the new value to the current value
                current_value = map[new_row][new_col] 
                minRow, minCol = new_row, new_col # assign the new location to the min location
        
        current_row, current_col = minRow, minCol # make the current location be the location with the lowest value
        trajectory.append((current_row, current_col)) # add the current location to the trajectory

    return trajectory


def generate_random_map():
    """
    #TODO: Function to generate a random map. for testing purposes. 
    """
    map = np.array([])
    rows = 20
    cols = 20
    map = np.random.randint(0, 2, (rows, cols))
    
    # set the goal location
    map[rows-1][cols-1] = 2
    # print_output(value_map=map, trajectory=[(0, 0)])

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
            map = np.array(map)
            map = map.astype(np.uint16)
            start_row = int(input("Enter start row: "))
            start_col = int(input("Enter start col: "))
            value_map, trajectory = planner(map, start_row, start_col)
            print_output(value_map, trajectory)
            display_map(value_map, trajectory)

        elif option == "2":
            map = generate_random_map()
            map = np.array(map)
            map = map.astype(np.uint16)
            start_row = int(input("Enter start row: "))
            start_col = int(input("Enter start col: "))
            value_map, trajectory = planner(map, start_row, start_col)
            while trajectory == [[]] or trajectory == []:
                map = generate_random_map()
                map = np.array(map)
                map = map.astype(np.uint16)
                value_map, trajectory = planner(map, start_row, start_col)
            print_output(value_map, trajectory)
            display_map(value_map, trajectory)

        elif option == "0":
            break

        else:
            print("Invalid option. Try again.")

        print("")
    pass


def debug_loop():
    # matrix = load_from_file("maze.mat")
    matrix = notAllSol
    # matrix = generate_random_map()

    trajectory = []
    # matrix = load_from_file("maze.mat")
    matrix = np.array(matrix)
    matrix = matrix.astype(np.uint16)
    # print_output(value_map=matrix, trajectory=trajectory)
    row, col = find_goal_coordinate(matrix)
    value_map, trajectory = planner(matrix, 0, 9)
    #! 13,2 not working so we handle zero indexing here
    # value_map = wavefront_map(matrix, row, col)
    print_output(value_map=value_map, trajectory=trajectory)
    display_map(value_map, trajectory)


main_loop()
# debug_loop()
