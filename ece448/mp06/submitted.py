# submitted.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# submitted should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi)

import queue
import heapq

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement bfs function
    # data structure we use
    bfs_queue = queue.Queue()
    path = []
    visit = {}
    # prepare for the start
    start = maze.start
    target = maze.waypoints[0]
    bfs_queue.put(start)
    visit[start] = (-1,-1)

    if (start == target): # if start point is the target point
        path.append(start)
        return path
    while (bfs_queue.qsize() != 0):
        # print(bfs_queue.qsize())
        cur = bfs_queue.get()
        if (cur == target): # reach the target
            while (cur != (-1,-1)):
                path.append(cur)
                cur = visit[cur]
            path.reverse()
            break
        # otherwise, loop through the neighbor
        neighbor = maze.neighbors_all(cur[0],cur[1])
        for point in neighbor:
            if point not in visit.keys():
                visit[point] = cur
                bfs_queue.put(point)
    return path

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement astar_single
    # data structure we use
    astar_queue = queue.PriorityQueue()
    path = []
    visit = {}
    cost = {}
    # prepare for the start
    start = maze.start
    target = maze.waypoints[0]
    visit[start] = (-1,-1)
    cost[start] = 0

    if (start == target): # if start point is the target point
        path.append(start)
        return path
    f_start = (Chebyshev_distance(start,target),start[0],start[1])
    astar_queue.put(f_start)
    while (astar_queue.qsize != 0):
        f_cur = astar_queue.get()
        cur = (f_cur[1],f_cur[2])
        if (cur == target): # reach the target
            while (cur != (-1,-1)):
                path.append(cur)
                cur = visit[cur]
            path.reverse()
            break
        # otherwise, loop through the neighbor
        neighbor = maze.neighbors_all(cur[0],cur[1])
        for point in neighbor:
            if (point not in visit.keys()) or cost[point] > cost[cur] + 1:
                # print(cost[cur])
                cost[point] = cost[cur] + 1
                visit[point] = cur
                f_point = (Chebyshev_distance(point,target) + cost[point],point[0],point[1])
                astar_queue.put(f_point)
    return path

# the A* heuristic function.
def Chebyshev_distance(a,b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

# This function is for Extra Credits, please begin this part after finishing previous two functions
def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # data structure we use
    path = []
    visit = {}
    visit_target = {}
    # prepare for the start
    start = maze.start
    nearest_target = maze.waypoints[0]
    path.append(start)
    for target in maze.waypoints:
        visit_target[target] = 0
        if (Chebyshev_distance(target,start) < Chebyshev_distance(nearest_target,start)):
            nearest_target = target
    visit[start] = (-1,-1)
    while (any(value == 0 for value in visit_target.values())):
        tem_path = astar_helper(maze,start,nearest_target,visit_target)
        # update the data
        visit_target[nearest_target] = 1
        start = nearest_target
        for target in visit_target.keys():
            if visit_target[target] == 0:
                nearest_target = target
                break
        for target in maze.waypoints:
            if (Chebyshev_distance(target,start) < Chebyshev_distance(nearest_target,start) ) and (visit_target[target] == 0):
                nearest_target = target
        path.extend(tem_path[1:])
    return path

def astar_helper(maze,start,target,visit_target):
    # data structure we use
    astar_queue = queue.PriorityQueue()
    path = []
    visit = {}
    cost = {}
    # prepare for the start
    visit[start] = (-1,-1)
    cost[start] = 0

    if (start == target): # if start point is the target point
        path.append(start)
        return path
    f_start = (Chebyshev_distance(start,target) + MST(visit_target,start),start[0],start[1])
    astar_queue.put(f_start)
    while (astar_queue.qsize != 0):
        f_cur = astar_queue.get()
        cur = (f_cur[1],f_cur[2])
        if (cur == target): # reach the target
            while (cur != (-1,-1)):
                path.append(cur)
                cur = visit[cur]
            path.reverse()
            break
        # otherwise, loop through the neighbor
        neighbor = maze.neighbors_all(cur[0],cur[1])
        for point in neighbor:
            if point not in visit.keys():
                # print(cost[cur])
                cost[point] = cost[cur] + 1
                visit[point] = cur
                f_point = (Chebyshev_distance(point,target) + cost[point] + MST(visit_target,target),point[0],point[1])
                astar_queue.put(f_point)
    return path

# given the visit target dictionary and the target, calculate the distance of MST 
def MST(visit_target,start,init = 0):
    dist_table = {}
    temp = visit_target.copy()
    dist = 0
    temp[start] = init
    for target in temp.keys():
        if (temp[target] == 0): # only care the target not visit before
            dist_table[target] = Chebyshev_distance(start,target)
    # loop through all targets
    while (sum(temp.values()) != len(temp)):
        nearest_target = start
        for target in dist_table:
            temp_dist = 2024 # any number large enough
            if temp[target] == 0 and temp_dist > dist_table[target]:
                nearest_target = target
                temp_dist = dist_table[target]
        temp[nearest_target] = 1
        dist += dist_table[nearest_target]
        for target in dist_table: # update the distacne in the table
            if (Chebyshev_distance(nearest_target,target) + dist_table[nearest_target] < dist_table[target]):
                dist_table[target] = Chebyshev_distance(nearest_target,target) + dist_table[nearest_target]

    return dist 