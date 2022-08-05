import sys
import queue
import time
import random

time_limit = 5
random_seed = 5
input_file = ''

node_num = 0
depot_ptr = 0
required_edges_num = 0
non_required_edges_num = 0
vehicle_num = 0
capacity = 0
total_cost_of_required_edges = 0

nodes = []

distance_array = [] # 记忆化

free_list = []

total_routes = []
total_cost = 0


cost_map = {} # arc to cost
demand_map = {} # arc to demand

route_map = {} # index to route
route_idx = 0

load_map = {} # index to load


class Node:
    neighbours = []
    idx = 0
    cost_list = []
    demand_list = []
    capacity = capacity

    def __init__(self):
        self.neighbours = []
        self.idx = 0
        self.cost_list = []
        self.demand_list = []
        self.capacity = capacity


# 读取文件，初始化参数
def initialization():
    consoles = sys.argv
    global input_file, time_limit, random_seed
    global nodes, node_num, depot_ptr, required_edges_num, non_required_edges_num, \
        vehicle_num, capacity, total_cost_of_required_edges

    if len(consoles) < 2:
        print("Lack of file name, can not execute normally.\n")
        exit(1)
    elif len(consoles) < 6:
        input_file = consoles[1]
        print("Lack of parameters\n")
    else:
        input_file = consoles[1]
        time_limit = int(consoles[3])
        random_seed = int(consoles[5])

    with open(input_file, 'r') as file:
        line = file.readline()
        while line:
            str_list = line.split()
            if str_list[0] == 'VERTICES':
                node_num = int(str_list[2])
                for i in range(node_num):
                    nodes.append(Node())
                    nodes[i].idx = i
            elif str_list[0] == 'DEPOT':
                depot_ptr = int(str_list[2]) - 1
            elif str_list[0] == 'REQUIRED':
                required_edges_num = int(str_list[3])
            elif str_list[0] == 'NON-REQUIRED':
                non_required_edges_num = int(str_list[3])
            elif str_list[0] == 'VEHICLES':
                vehicle_num = int(str_list[2])
            elif str_list[0] == 'CAPACITY':
                capacity = int(str_list[2])
            elif str_list[0] == 'TOTAL':
                total_cost_of_required_edges = int(str_list[6])
            elif str_list[0].isdigit():
                # 为了编号方便，idx 统一从零开始
                from_node_idx = int(str_list[0]) - 1
                to_node_idx = int(str_list[1]) - 1
                cost = int(str_list[2])
                demand = int(str_list[3])
                nodes[from_node_idx].neighbours.append(to_node_idx)
                nodes[from_node_idx].cost_list.append(cost)
                nodes[from_node_idx].demand_list.append(demand)
                nodes[to_node_idx].neighbours.append(from_node_idx)
                nodes[to_node_idx].cost_list.append(cost)
                nodes[to_node_idx].demand_list.append(demand)
                cost_map[(from_node_idx, to_node_idx)] = cost
                cost_map[(to_node_idx, from_node_idx)] = cost
                demand_map[(from_node_idx, to_node_idx)] = demand
                demand_map[(to_node_idx, from_node_idx)] = demand

                if demand != 0: # 有任务需求的边，将 正反边 加入 free_list
                    free_list.append((from_node_idx, to_node_idx))
                    free_list.append((to_node_idx,from_node_idx))

            elif str_list[0] == 'END':
                break
            line = file.readline()


# 计算两点之间的最短开销
def dijkstra(src: int, dst: int) -> int:
    distance = [sys.maxsize] * node_num
    distance[src] = 0
    visited = []
    pq = queue.PriorityQueue()
    pq.put([0, src])
    while not pq.empty():

        term = pq.get()
        curr_cost = term[0]
        curr_idx = term[1]
        if curr_idx in visited:
            continue
        visited.append(curr_idx)

        for i in range(len(nodes[curr_idx].neighbours)):
            neigh_idx = nodes[nodes[curr_idx].neighbours[i]].idx
            new_cost = curr_cost + nodes[curr_idx].cost_list[i]

            if distance[neigh_idx] > new_cost:
                distance[neigh_idx] = new_cost
                pq.put([new_cost, neigh_idx])

    return distance[dst]

def path_scanning():
    global total_cost, total_routes, capacity, depot_ptr, free_list, route_idx, route_map
    route = []
    end = depot_ptr

    cost = 0
    arc = (0, 0)
    load = 0
    compatible = False
    while free_list != []:
        edge_cost = sys.maxsize
        for i in free_list:
            if demand_map[i] + load <= capacity:
                compatible = True
                if distance_array[end][i[0]] < edge_cost:
                    arc = i
                    edge_cost = distance_array[end][i[0]]
                elif distance_array[end][i[0]] == edge_cost and distance_array[i[1]][depot_ptr] > distance_array[arc[1]][depot_ptr]:
                    arc = i

        if not compatible:
            total_cost = total_cost + cost + distance_array[arc[1]][depot_ptr]
            total_routes.append(0)
            total_routes.extend(route)
            total_routes.append(0)
            route_map[route_idx] = route

            load_map[route_idx] = load
            route_idx = route_idx + 1
            # print("route = ", route)
            # print("cost =",total_cost)
            load = 0
            route = []
            cost = 0
            end = depot_ptr
            arc = (0,0)
        else:
            free_list.remove(arc)
            free_list.remove((arc[1], arc[0]))

            load = load + demand_map[arc]

            cost = cost + distance_array[end][arc[0]]
            cost = cost + cost_map[arc]
            # print("In adding, cost = ", cost)
            route.append(arc)
            compatible = False
            end = arc[1]
    if route != []: # 加上剩余部分
        total_cost = total_cost + cost + distance_array[arc[1]][depot_ptr]
        # print("route = ", route)
        # print("cost =", total_cost)
        total_routes.append(0)
        total_routes.extend(route)
        total_routes.append(0)
        load_map[route_idx] = load
        route_map[route_idx] = route
        route_idx = route_idx + 1

def flipping():
    global total_cost
    for i in range(len(total_routes)):
        origin_cost = 0
        new_cost = 0
        if total_routes[i] != 0:
            if i > 0:
                if total_routes[i - 1] == 0: # 路径开始元素
                    origin_cost += distance_array[depot_ptr][total_routes[i][0]]
                    new_cost += distance_array[depot_ptr][total_routes[i][1]]


                else:
                    origin_cost += distance_array[total_routes[i - 1][1]][total_routes[i][0]]
                    new_cost += distance_array[total_routes[i - 1][1]][total_routes[i][1]]

            if i < len(total_routes):
                if total_routes[i+1] == 0: # 路径最后一个元素
                    origin_cost += distance_array[total_routes[i][1]][depot_ptr]
                    new_cost += distance_array[total_routes[i][0]][depot_ptr]
                else:
                    origin_cost += distance_array[total_routes[i][1]][total_routes[i + 1][0]]
                    new_cost += distance_array[total_routes[i][0]][total_routes[i + 1][0]]

        if origin_cost > new_cost:
            # print("origin_cost = ", origin_cost)
            # print("new_cost = ", new_cost)
            pair = total_routes[i]
            # print("origin_pair = ", pair)
            new_pair = (pair[1], pair[0])
            total_routes[i]  = new_pair
            total_cost = total_cost - origin_cost + new_cost

def self_insertion() -> int:
    global route_idx, depot_ptr, total_routes, total_cost
    route_random_number = random.randint(0, route_idx - 1)
    route = total_routes[route_random_number]
    segment_number = random.randint(1, len(route) - 2)
    pos_idx = random.randint(1, len(route) - 2) # 需要插入的位置

    if segment_number == 1:
        old_start = depot_ptr
    else:
        old_start = route[segment_number - 1][1]
    if segment_number == len(route) - 2:
        old_end = depot_ptr
    else:
        old_end = route[segment_number + 1][0]

    segment = route[segment_number]

    route.remove(segment)
    route.insert(pos_idx, segment)

    if pos_idx == 1:
        new_start = depot_ptr
    else:
        new_start = route[pos_idx - 1][1]

    if pos_idx == len(route) - 2:
        new_end = depot_ptr
    else:
        new_end = route[pos_idx + 1][0]
    old_cost = distance_array[new_start][new_end] + distance_array[old_start][segment[0]] \
               + distance_array[segment[1]][old_end]
    new_cost = distance_array[old_start][old_end] + distance_array[new_start][segment[0]] \
               + distance_array[segment[1]][new_end]

    return old_cost - new_cost


def cross_single_insertion() -> int:
    global total_routes, capacity, depot_ptr
    r1 = random.randint(0, len(total_routes) - 1) # 选择一条需要提取段的子路径
    route1 = total_routes[r1] # 子路径本径
    from_seg_num = random.randint(1, len(route1) - 2) # 将要取出进行交换的段
    from_seg = route1[from_seg_num] # 段本段
    demand = demand_map[from_seg] # 该段的需求量
    route_candidate = [] # 挑选出可以插入的路线作为候选
    idx_map = {} #记录候选者的全局路径索引
    cnt = 0
    for i in range(len(total_routes)):
        if i != r1 and load_map[i] + demand <= capacity:
            route_candidate.append(total_routes[i])
            idx_map[cnt] = i
            cnt = cnt + 1
    if cnt == 0: # 没有符合的路径
        return 0
    r2 = random.randint(0, len(route_candidate) - 1)
    route2 = route_candidate[r2]
    insertion_pos = random.randint(1, len(route2) - 1) # 将要插入的位置（去首去尾）
    if from_seg_num == 1:
        old_start = depot_ptr
    else:
        old_start = route1[from_seg_num - 1][1]
    if from_seg_num == len(route1) - 2:
        old_end = depot_ptr
    else:
        old_end = route1[from_seg_num + 1][0]

    route2.insert(insertion_pos, from_seg)
    route1.remove(from_seg)

    if insertion_pos == 1:
        new_start = depot_ptr
    else:
        new_start = route2[insertion_pos - 1][1]
    if insertion_pos == len(route2) - 1:
        new_end = depot_ptr
    else:
        new_end = route2[insertion_pos + 1][0]

    old_cost = distance_array[new_start][new_end] + distance_array[old_start][from_seg[0]] \
               + distance_array[from_seg[1]][old_end]
    new_cost = distance_array[old_start][old_end] + distance_array[new_start][from_seg[0]] \
               + distance_array[from_seg[1]][new_end]

    load_map[r1] = load_map[r1] - demand_map[from_seg]
    load_map[idx_map[r2]] = load_map[idx_map[r2]] + demand_map[from_seg]

    return new_cost - old_cost

def cross_double_insertion() -> int:
    global total_routes, route_idx, capacity, depot_ptr
    r1 = random.randint(0, route_idx - 1)
    route1 = total_routes[r1]
    if len(route1) < 4:
        return 0
    from_idx = random.randint(1, len(route1) - 3)
    demand = demand_map[route1[from_idx]] + demand_map[route1[from_idx + 1]]
    candidate = []
    idx_map = {}
    cnt = 0
    for i in range(len(total_routes)):
        if i != r1 and load_map[i] + demand <= capacity:
            candidate.append(total_routes[i])

            idx_map[cnt] = i
            cnt = cnt + 1
    if cnt == 0:
        return 0
    if from_idx == 1:
        old_start = depot_ptr
    else:
        old_start = route1[from_idx - 1][1]
    if from_idx == len(route1) - 3:
        old_end = depot_ptr
    else:
        old_end = route1[from_idx + 2][0]

    r2 = random.randint(1, len(candidate)-1)
    route2 = candidate[r2]
    insert_pos = random.randint(1, len(route2) - 1)

    route2.insert(insert_pos, route1[from_idx])
    route2.insert(insert_pos + 1, route1[from_idx + 1])

    route1.remove(route1[from_idx])
    route1.remove(route1[from_idx + 1])
    if insert_pos == 1:
        new_start = depot_ptr
    else:
        new_start = route2[insert_pos - 1][1]
    if insert_pos == len(route2) - 1:
        new_end = depot_ptr
    else:
        new_end = route2[insert_pos + 1][0]

    old_cost = distance_array[new_start][new_end] + distance_array[old_start][route1[from_idx][0]] \
               + distance_array[route1[from_idx + 1][1]][old_end]
    new_cost = distance_array[old_start][old_end] + distance_array[new_start][route1[from_idx][0]] \
               + distance_array[route1[from_idx + 1][1]][new_end]

    load_map[r1] = load_map[r1] - demand
    load_map[idx_map[r2]] = load_map[idx_map[r2]] + demand

    return new_cost - old_cost

def swap():
    global total_routes, depot_ptr, capacity
    r1 = 0
    r2 = 0
    while r1 == r2:
        r1 = random.randint(0, len(total_routes))
        r2 = random.randint(0, len(total_routes))
    route1 = total_routes[r1]
    route2 = total_routes[r2]
    cnt = 0
    while True:
        pos1 = random.randint(1, len(route1) - 2)
        pos2 = random.randint(1, len(route2) - 2)
        seg1 = route1[pos1]
        seg2 = route2[pos2]
        if pos1 == 1:
            start1 = depot_ptr
        else:
            start1 = route1[pos1 - 1][1]
        if pos1 == len(route1) - 2:
            end1 = depot_ptr
        else:
            end1 = route1[pos1 + 1][0]

        if pos1 == 1:
            start2 = depot_ptr
        else:
            start2 = route2[pos2 - 1][1]
        if pos1 == len(route2) - 2:
            end2 = depot_ptr
        else:
            end2 = route2[pos2 + 1][0]

        demand1 = load_map[r1] - demand_map[seg1] + demand_map[seg2]
        demand2 = load_map[r2] - demand_map[seg2] + demand_map[seg1]
        cnt += 1
        if (demand1 <= capacity and demand2 <= capacity) or cnt >= len(route1) + len(route2):
            break

    if cnt >= len(route1) + len(route2):
        return 0

    route1.remove(seg1)
    route1.insert(r1, seg2)
    route2.remove(seg2)
    route2.insert(r2, seg1)

    load_map[r1] = demand1
    load_map[r2] = demand2

    old_cost = distance_array[start1][seg1[0]] + distance_array[seg1[1]][end1] \
               + distance_array[start2][seg2[0]] + distance_array[seg2[1]][end2]
    new_cost = distance_array[start1][seg2[0]] + distance_array[seg2[1]][end1] \
               + distance_array[start2][seg1[0]] + distance_array[seg1[1]][end2]
    return new_cost - old_cost








def print_info():
    global total_routes, total_cost
    for i in range(len(total_routes)):
        if i == 0:
            print("s 0", end="")
        elif total_routes[i] != 0:
            print(',(%d,%d)'%(total_routes[i][0]+1, total_routes[i][1]+1), end="")
        else:
            print(",%d" %(0), end="")

    print("\nq", total_cost)



if __name__ == "__main__":
    start = time.time() # start time
    initialization()
    # 记忆化
    distance_array = [[0 for i in range(node_num)] for j in range(node_num)]
    for i in range(node_num):
        for j in range(node_num):
            if i != j:
                distance_array[i][j] = dijkstra(i, j) # 存储从 i 到 j 的距离开销

    end = time.time() # end time

    path_scanning()

    flipping()

    print_info()


    print("Using {}s".format(end-start))

