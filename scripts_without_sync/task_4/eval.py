import cv2
import numpy as np


class JunctionPoint:
    def __init__(self, tag=None):
        self.tag = tag
        self.neighbors = []

    def link(self, point: 'JunctionPoint', dist: int):
        self.neighbors.append((point, dist))

    def __str__(self):
        return str(self.tag)


class RoutePath:
    def __init__(self, origin: JunctionPoint):
        self.origin = origin
        self.waypoints = [origin]
        self.total_dist = 0

    @property
    def current_point(self):
        return self.waypoints[-1]

    def append_point(self, point: JunctionPoint, dist: int):
        self.waypoints.append(point)
        self.total_dist += dist

    def duplicate(self) -> 'RoutePath':
        clone = RoutePath(self.origin)
        clone.waypoints = self.waypoints.copy()
        clone.total_dist = self.total_dist
        return clone

    def __str__(self):
        return ' -> '.join(map(str, self.waypoints))


j1_entry_r = JunctionPoint(1)
j1_entry_b = JunctionPoint(1)
j1_entry_l = JunctionPoint(1)

j1_exit_r = JunctionPoint('j1_exit_r')
j1_exit_b = JunctionPoint('j1_exit_b')
j1_exit_l = JunctionPoint('j1_exit_l')

j1_entry_r.link(j1_exit_b, 14)
j1_entry_r.link(j1_exit_l, 13)

j1_entry_l.link(j1_exit_b, 9)
j1_entry_l.link(j1_exit_r, 13)

j1_entry_b.link(j1_exit_r, 9)
j1_entry_b.link(j1_exit_l, 14)


j2_entry_r = JunctionPoint(2)
j2_entry_b = JunctionPoint(2)
j2_entry_t = JunctionPoint(2)

j2_exit_r = JunctionPoint('j2_exit_r')
j2_exit_b = JunctionPoint('j2_exit_b')
j2_exit_t = JunctionPoint('j2_exit_t')

j2_entry_r.link(j2_exit_b, 14)
j2_entry_r.link(j2_exit_t, 9)

j2_entry_b.link(j2_exit_t, 13)
j2_entry_b.link(j2_exit_r, 9)

j2_entry_t.link(j2_exit_b, 13)
j2_entry_t.link(j2_exit_r, 14)


j3_entry_l = JunctionPoint(3)
j3_entry_r = JunctionPoint(3)
j3_entry_b = JunctionPoint(3)
j3_entry_t = JunctionPoint(3)

j3_exit_l = JunctionPoint('j3_exit_l')
j3_exit_r = JunctionPoint('j3_exit_r')
j3_exit_b = JunctionPoint('j3_exit_b')
j3_exit_t = JunctionPoint('j3_exit_t')

j3_entry_l.link(j3_exit_r, 13)
j3_entry_l.link(j3_exit_t, 14)
j3_entry_l.link(j3_exit_b, 9)

j3_entry_r.link(j3_exit_l, 13)
j3_entry_r.link(j3_exit_t, 9)
j3_entry_r.link(j3_exit_b, 14)

j3_entry_b.link(j3_exit_r, 9)
j3_entry_b.link(j3_exit_t, 13)
j3_entry_b.link(j3_exit_l, 14)

j3_entry_t.link(j3_exit_r, 14)
j3_entry_t.link(j3_exit_l, 9)
j3_entry_t.link(j3_exit_b, 13)


j4_entry_l = JunctionPoint(4)
j4_entry_b = JunctionPoint(4)
j4_entry_t = JunctionPoint(4)

j4_exit_l = JunctionPoint('j4_exit_l')
j4_exit_b = JunctionPoint('j4_exit_b')
j4_exit_t = JunctionPoint('j4_exit_t')

j4_entry_l.link(j4_exit_b, 9)
j4_entry_l.link(j4_exit_t, 14)

j4_entry_b.link(j4_exit_t, 13)
j4_entry_b.link(j4_exit_l, 14)

j4_entry_t.link(j4_exit_b, 13)
j4_entry_t.link(j4_exit_l, 9)


j5_entry_l = JunctionPoint(5)
j5_entry_r = JunctionPoint(5)
j5_entry_t = JunctionPoint(5)

j5_exit_l = JunctionPoint('j5_exit_l')
j5_exit_r = JunctionPoint('j5_exit_r')
j5_exit_t = JunctionPoint('j5_exit_t')

j5_entry_l.link(j5_exit_r, 13)
j5_entry_l.link(j5_exit_t, 14)

j5_entry_r.link(j5_exit_l, 13)
j5_entry_r.link(j5_exit_t, 9)

j5_entry_t.link(j5_exit_r, 14)
j5_entry_t.link(j5_exit_l, 9)


j1_exit_l.link(j2_entry_t, 33)
j1_exit_r.link(j4_entry_t, 28)
j1_exit_b.link(j3_entry_t, 10)

j2_exit_t.link(j1_entry_l, 28)
j2_exit_r.link(j3_entry_l, 12)
j2_exit_b.link(j5_entry_l, 33)

j3_exit_l.link(j2_entry_r, 12)
j3_exit_r.link(j4_entry_l, 12)
j3_exit_b.link(j5_entry_t, 10)
j3_exit_t.link(j1_entry_b, 10)

j4_exit_l.link(j3_entry_r, 12)
j4_exit_t.link(j1_entry_r, 33)
j4_exit_b.link(j5_entry_r, 28)

j5_exit_l.link(j2_entry_b, 28)
j5_exit_r.link(j4_entry_b, 33)
j5_exit_t.link(j3_entry_b, 10)

lane_width = 27

upright_lanes = {
    'up_left': [44, 106, 215],
    'up_center': [307, 106, 215],
    'up_right': [570, 106, 215],
    'down_left': [44, 350, 459],
    'down_center': [307, 350, 459],
    'down_right': [570, 350, 459]
}

sideways_lanes = {
    'west_up': [39, 111, 240],
    'west_mid': [283, 111, 240],
    'west_low': [526, 111, 240],
    'east_up': [39, 375, 503],
    'east_mid': [283, 375, 503],
    'east_low': [526, 375, 503]
}


def locate_origin_marker(frame):
    blue_mask = cv2.inRange(frame, (50, 0, 0), (255, 0, 0))
    cols = np.nonzero(np.argmax(blue_mask, axis=0))[0]
    rows = np.nonzero(np.argmax(blue_mask, axis=1))[0]
    
    if len(cols) == 0 or len(rows) == 0:
        return None, None
    
    pos_x = int(sum(cols) / len(cols))
    pos_y = int(sum(rows) / len(rows))
    return pos_x, pos_y


def locate_target_marker(frame):
    red_mask = cv2.inRange(frame, (0, 0, 50), (0, 0, 255))
    cols = np.nonzero(np.argmax(red_mask, axis=0))[0]
    rows = np.nonzero(np.argmax(red_mask, axis=1))[0]
    
    if len(cols) == 0 or len(rows) == 0:
        return None, None
    
    pos_x = int(sum(cols) / len(cols))
    pos_y = int(sum(rows) / len(rows))
    return pos_x, pos_y


def verify_direct_path(origin_lane, target_lane, origin_heading,
                       target_heading, origin_x, origin_y,
                       target_x, target_y, origin_type, target_type):
    if origin_lane == target_lane and origin_heading == target_heading:
        if origin_type == 'upright':
            if origin_heading == 'north':
                if origin_y > target_y:  # origin should be lower
                    return True
            else:
                if origin_y < target_y:  # origin should be higher
                    return True
        else:
            if origin_heading == 'east':
                if origin_x < target_x:
                    return True
            else:
                if origin_x > target_x:
                    return True

    scenarios = []
    scenarios.append(all([origin_lane == 'west_up',
                         origin_heading == 'west',
                         target_lane == 'up_left',
                         target_heading == 'south']))
    scenarios.append(all([origin_lane == 'up_left',
                         origin_heading == 'north',
                         target_lane == 'west_up',
                         target_heading == 'east']))

    scenarios.append(all([origin_lane == 'east_up',
                         origin_heading == 'east',
                         target_lane == 'up_right',
                         target_heading == 'south']))
    scenarios.append(all([origin_lane == 'up_right',
                         origin_heading == 'north',
                         target_lane == 'east_up',
                         target_heading == 'west']))

    scenarios.append(all([origin_lane == 'west_low',
                         origin_heading == 'west',
                         target_lane == 'down_left',
                         target_heading == 'north']))
    scenarios.append(all([origin_lane == 'down_left',
                         origin_heading == 'south',
                         target_lane == 'west_low',
                         target_heading == 'east']))

    scenarios.append(all([origin_lane == 'east_low',
                         origin_heading == 'east',
                         target_lane == 'down_right',
                         target_heading == 'north']))
    scenarios.append(all([origin_lane == 'down_right',
                         origin_heading == 'south',
                         target_lane == 'east_low',
                         target_heading == 'west']))

    if any(scenarios):
        return True

    return False


def find_the_shortest_way(image) -> list:

    origin_x, origin_y = locate_origin_marker(image)
    target_x, target_y = locate_target_marker(image)

    if origin_x is None or target_x is None:
        return []

    cv2.circle(image, (origin_x, origin_y), 20, (0, 255, 0), 1)
    cv2.circle(image, (target_x, target_y), 20, (0, 255, 0), 1)

   

    origin_type = None
    origin_lane = None
    origin_heading = None
    target_type = None
    target_lane = None
    target_heading = None

    for lane_id, (lx, l_start, l_end) in upright_lanes.items():
        if lx - lane_width <= origin_x <= lx + lane_width and l_start <= origin_y <= l_end:
            origin_type = 'upright'
            origin_lane = lane_id
            if origin_x > lx:
                origin_heading = 'north'
            else:
                origin_heading = 'south'

        if lx - lane_width <= target_x <= lx + lane_width and l_start <= target_y <= l_end:
            target_type = 'upright'
            target_lane = lane_id
            if target_x > lx:
                target_heading = 'north'
            else:
                target_heading = 'south'

    for lane_id, (ly, l_start, l_end) in sideways_lanes.items():
        if ly - lane_width <= origin_y <= ly + lane_width and l_start <= origin_x <= l_end:
            origin_type = 'sideways'
            origin_lane = lane_id
            if origin_y > ly:
                origin_heading = 'east'
            else:
                origin_heading = 'west'

        if ly - lane_width <= target_y <= ly + lane_width and l_start <= target_x <= l_end:
            target_type = 'sideways'
            target_lane = lane_id
            if target_y > ly:
                target_heading = 'east'
            else:
                target_heading = 'west'

  
    if verify_direct_path(origin_lane, target_lane, origin_heading,
                         target_heading, origin_x, origin_y, target_x,
                         target_y, origin_type, target_type):
        return []

    origin_node = JunctionPoint("ORIGIN")
    if origin_lane == 'up_left':
        if origin_heading == 'north':
            origin_node.link(j1_entry_l, 23)
        else:
            origin_node.link(j2_entry_t, 5)

    elif origin_lane == 'up_center':
        if origin_heading == 'north':
            origin_node.link(j1_entry_b, 5)
        else:
            origin_node.link(j3_entry_t, 5)

    elif origin_lane == 'up_right':
        if origin_heading == 'north':
            origin_node.link(j1_entry_r, 28)
        else:
            origin_node.link(j4_entry_t, 5)

    elif origin_lane == 'down_left':
        if origin_heading == 'north':
            origin_node.link(j2_entry_b, 5)
        else:
            origin_node.link(j5_entry_l, 28)

    elif origin_lane == 'down_center':
        if origin_heading == 'north':
            origin_node.link(j3_entry_b, 5)
        else:
            origin_node.link(j5_entry_t, 5)

    elif origin_lane == 'down_right':
        if origin_heading == 'north':
            origin_node.link(j4_entry_b, 5)
        else:
            origin_node.link(j5_entry_r, 23)

    elif origin_lane == 'west_up':
        if origin_heading == 'east':
            origin_node.link(j1_entry_l, 5)
        else:
            origin_node.link(j2_entry_t, 28)

    elif origin_lane == 'west_mid':
        if origin_heading == 'east':
            origin_node.link(j3_entry_l, 5)
        else:
            origin_node.link(j2_entry_r, 5)

    elif origin_lane == 'west_low':
        if origin_heading == 'east':
            origin_node.link(j5_entry_l, 5)
        else:
            origin_node.link(j2_entry_b, 23)

    elif origin_lane == 'east_up':
        if origin_heading == 'east':
            origin_node.link(j4_entry_t, 23)
        else:
            origin_node.link(j1_entry_r, 5)

    elif origin_lane == 'east_mid':
        if origin_heading == 'east':
            origin_node.link(j4_entry_l, 5)
        else:
            origin_node.link(j3_entry_r, 5)

    elif origin_lane == 'east_low':
        if origin_heading == 'east':
            origin_node.link(j4_entry_b, 28)
        else:
            origin_node.link(j5_entry_r, 5)

    target_node = JunctionPoint("TARGET")
    if target_lane == 'up_left':
        if target_heading == 'north':
            j2_exit_t.link(target_node, 5)
        else:
            j1_exit_l.link(target_node, 28)

    elif target_lane == 'up_center':
        if target_heading == 'north':
            j3_exit_t.link(target_node, 5)
        else:
            j1_exit_b.link(target_node, 5)

    elif target_lane == 'up_right':
        if target_heading == 'north':
            j4_exit_t.link(target_node, 5)
        else:
            j1_exit_r.link(target_node, 23)

    elif target_lane == 'down_left':
        if target_heading == 'south':
            j2_exit_b.link(target_node, 5)
        else:
            j5_exit_l.link(target_node, 23)

    elif target_lane == 'down_center':
        if target_heading == 'south':
            j3_exit_b.link(target_node, 5)
        else:
            j5_exit_t.link(target_node, 5)

    elif target_lane == 'down_right':
        if target_heading == 'south':
            j4_exit_b.link(target_node, 5)
        else:
            j5_exit_r.link(target_node, 28)

    elif target_lane == 'west_up':
        if target_heading == 'east':
            j2_exit_t.link(target_node, 23)
        else:
            j1_exit_l.link(target_node, 5)

    elif target_lane == 'west_mid':
        if target_heading == 'east':
            j2_exit_r.link(target_node, 5)
        else:
            j3_exit_l.link(target_node, 5)

    elif target_lane == 'west_low':
        if target_heading == 'east':
            j2_exit_b.link(target_node, 28)
        else:
            j5_exit_l.link(target_node, 5)

    elif target_lane == 'east_up':
        if target_heading == 'east':
            j1_exit_r.link(target_node, 5)
        else:
            j4_exit_t.link(target_node, 28)

    elif target_lane == 'east_mid':
        if target_heading == 'east':
            j3_exit_r.link(target_node, 5)
        else:
            j4_exit_l.link(target_node, 5)

    elif target_lane == 'east_low':
        if target_heading == 'east':
            j5_exit_r.link(target_node, 5)
        else:
            j4_exit_b.link(target_node, 23)

    candidate_routes = [RoutePath(origin_node)]
    optimal_route = None
    optimal_length = 0

    searching = True
    while searching:
        next_routes = []
        found_shorter = False
        shorter_candidates = []
        for route in candidate_routes:
            for adjacent_point, segment_len in route.current_point.neighbors:
                if adjacent_point == target_node:
                    if optimal_route:
                        route.append_point(adjacent_point, segment_len)
                           
                        if route.total_dist < optimal_route.total_dist:
                            optimal_route = route
                            optimal_length = route.total_dist
                    else:
                        route.append_point(adjacent_point, segment_len)
                        optimal_route = route
                        optimal_length = route.total_dist
                        found_shorter = True
                else:
                    route_copy = route.duplicate()
                    route_copy.append_point(adjacent_point, segment_len)
                    next_routes.append(route_copy)
                    if optimal_route:
                        if route_copy.total_dist < optimal_length:
                            found_shorter = True
                            shorter_candidates.append(route)

      

        if optimal_route and not found_shorter:
            break
        candidate_routes = next_routes.copy()

    if optimal_route is None:
        return []

    interchange_sequence = []
    for pt in optimal_route.waypoints:
        if pt.tag in (1, 2, 3, 4, 5):
            interchange_sequence.append(pt.tag)

    return interchange_sequence