initial_map =  [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
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

result = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
[1, 21, 20, 19, 18, 18, 1, 1, 14 ,14, 14, 14,14, 14, 1, 1, 3, 3, 3, 1],
[1, 21, 20, 19, 18, 17, 1, 1, 14, 13, 13, 13, 13, 13, 1, 1, 3, 2, 3, 1],
[1, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 12, 12, 12, 1, 1, 3, 3, 3, 1],
[1, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 11, 11, 1 ,1, 4, 4, 4, 1],
[1 ,21, 20, 19, 18, 17, 16, 15 ,14, 13 ,12 ,11, 10, 10, 1, 1, 5, 5 ,5, 1],
[1, 21 ,20 ,19 ,1, 1, 1, 1, 1, 13 ,12 ,11 ,10, 9, 1 ,1, 6, 6, 6, 1],
[1, 21 ,20, 19, 1, 1, 1, 1, 1, 13 ,12 ,11 ,10, 9, 8 ,7, 7, 7 ,7, 1],
[1, 21, 20, 19, 18, 17, 17, 1, 1, 13 ,12 ,11, 10 ,9, 8 ,8, 8, 8, 8, 1],
[1, 21 ,20, 19 ,18, 17 ,16 ,1 ,1, 13 ,12 ,11 ,10 ,9, 9 ,9, 9 ,9 ,9, 1],
[1, 21, 20, 19 ,18, 17, 16, 15, 14, 13 ,12, 11 ,10 ,10 ,1 ,10, 10, 10 ,10, 1],
[1, 21 ,20, 19 ,18, 17, 16 ,15, 14 ,13, 12, 11, 11, 1 ,1 ,11, 11 ,11, 11 ,1],
[1, 21, 20, 19 ,18, 17, 16, 15, 14 ,13 ,12, 12 ,1 ,1, 1 ,12, 12, 12, 12 ,1],
[1 ,1 ,1, 1 ,1 ,1, 1, 1, 1 ,1, 1 ,1,1, 1 ,1 ,1, 1, 1, 1 ,1]]

nosolution = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

notAllSol =[[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0],
              [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
              [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
              [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

diagonalSol =[[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]