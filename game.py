import arcade
import time
import math
import sys
import os

GRID_DIM = 30
MOVEMENT_SPEED = GRID_DIM
MAX_SPEED = 8
MIN_SPEED = 1.5
SPEED_GAIN = 1.1
FRICTION_GAIN = .98
BRAKE_GAIN = .8
ANG_GAIN = .07

START_POS = (5, 2)

MAP_PATH = 'maps/map1.txt'



SCREEN_WIDTH = 50 * GRID_DIM
SCREEN_HEIGHT = 30 * GRID_DIM

DT = .001


SCORE_X, SCORE_Y = 25*GRID_DIM, 20*GRID_DIM
SCORE_TXT_SIZE = 120
SHOW_DIST = True
GOAL_DELTA_TIME = .5


class Car(arcade.Sprite):
    def __init__(self, pos):
        self._center_x = GRID_DIM/2 + pos[0] * GRID_DIM
        self._center_y = GRID_DIM/2 + pos[1] * GRID_DIM

        self._height = GRID_DIM
        self._width = GRID_DIM/2

        self.colur = arcade.color.BLACK

        self.speed = 1
        self.ang = 0
        self._collision_radius = GRID_DIM
        self.ang_gain = 0
        self.speed_gain = 0

    def reset_pos(self, pos):
        self._center_x = GRID_DIM/2 + pos[0] * GRID_DIM
        self._center_y = GRID_DIM/2 + pos[1] * GRID_DIM
        self.speed = 1
        self.ang = 0

    def setUp(self, pos):
        self._center_x, self._center_y = pos * GRID_DIM + GRID_DIM/2

    def update(self):
        self.ang += self.ang_gain
        self.speed *= self.speed_gain

        if self.speed < MIN_SPEED:
            self.speed = MIN_SPEED
        elif self.speed > MAX_SPEED:
            self.speed = MAX_SPEED

        self._center_x += math.cos(self.ang) * self.speed
        self._center_y += math.sin(self.ang) * self.speed


    def draw(self):
        arcade.draw_rectangle_filled(self._center_x, self._center_y, self._height, self._width, self.colur, tilt_angle=-self.ang*180/math.pi)
class Wall(arcade.Sprite):
    def __init__(self, x, y):
        self._height = GRID_DIM
        self._width = GRID_DIM
        self.sprite_lists = []
        self._center_x = self._height / 2 + x * GRID_DIM
        self._center_y = self._width / 2 + y * GRID_DIM

        self.colur = arcade.color.GRAY
        self.mode = 'bg'
        self.line_width = 1

    def draw(self):
        arcade.draw_rectangle_filled(self._center_x, self._center_y, self._width, self._height, self.colur)
class Point(arcade.Sprite):
    def __init__(self, x, y, tilt):
        if tilt:
            self._width = GRID_DIM * 3
            self._height = GRID_DIM / 3
        else:
            self._width = GRID_DIM/3
            self._height = GRID_DIM*3

        self.sprite_lists = []
        self._center_x = GRID_DIM / 2 + x * GRID_DIM
        self._center_y = GRID_DIM / 2 + y * GRID_DIM

        self.colur = arcade.color.RED
        self.mode = 'bg'
        self.line_width = 1

    def draw(self):
        arcade.draw_rectangle_filled(self._center_x, self._center_y, self._width, self._height, self.colur)
class GoalLine(arcade.Sprite):
    def __init__(self, x, y, tilt):
        if tilt:
            self._width = GRID_DIM * 3
            self._height = GRID_DIM / 3
        else:
            self._width = GRID_DIM/3
            self._height = GRID_DIM*3

        self.sprite_lists = []
        self._center_x = GRID_DIM / 2 + x * GRID_DIM
        self._center_y = GRID_DIM / 2 + y * GRID_DIM

        self.colur = arcade.color.YELLOW
        self.mode = 'bg'
        self.line_width = 1

    def draw(self):
        arcade.draw_rectangle_filled(self._center_x, self._center_y, self._width, self._height, self.colur)
class Bar(arcade.Sprite):
    def __init__(self, x):
        self._width = GRID_DIM
        self._height = GRID_DIM / 3

        self.sprite_lists = []
        self._center_x = GRID_DIM*2 + x * 2 * GRID_DIM
        self._center_y = GRID_DIM*20

        self.colur = arcade.color.ORANGE
        self.mode = 'bg'
        self.line_width = 1

    def draw(self, h):
        arcade.draw_rectangle_filled(self._center_x, self._center_y+h/2, self._width, h, self.colur)
class Dot(arcade.Sprite):
    def __init__(self, x, y, ang):
        self._center_x = x
        self._center_y = y
        self.x_dot = math.cos(ang)
        self.y_dot = math.sin(ang)
    def update(self):
        self._center_x += self.x_dot*10
        self._center_y += self.y_dot*10
    def draw(self):
        arcade.draw_point(self._center_x, self._center_y, arcade.color.BLACK, 4)

class MyGame(arcade.Window):
    """ Main application class. """
    def __init__(self, width=SCREEN_WIDTH, height=SCREEN_HEIGHT, render=True, AI_mode=False):
        super().__init__(width, height)
        self.timestamp = time.time()
        self.car = None
        self.barSpeed = None
        self.barDist = None
        self.dots = []
        self.distance = []

        self.car_start_pos = START_POS

        self.do_rendering = render
        self.AI_mode = AI_mode

        self.wallList = arcade.SpriteList()
        self.pointList = arcade.SpriteList()
        self.pointListRef = arcade.SpriteList()
        self.goalList = arcade.SpriteList()
        arcade.set_background_color(arcade.color.AMAZON)
        self.score = 0
        self.score_last = 0
        self.goal_timer = time.time()

        self.done = False


    def setup(self):
        # Set up your game here
        self.car = Car(START_POS)
        self.barSpeed = Bar(1)
        self.barDist = Bar(2)

        f = open(MAP_PATH, "r")
        lines = f.readlines()
        for j, line in enumerate(lines):
            for i, indx in enumerate(line):
                if indx == '1':
                    self.wallList.append(Wall(i, len(lines)-1-j))
                elif indx == '2':
                    self.pointListRef.append(Point(i, len(lines)-1-j, False))
                elif indx == '3':
                    self.pointListRef.append(Point(i, len(lines)-1-j, True))
                elif indx == '4':
                    self.goalList.append(GoalLine(i, len(lines)-1-j, False))
                elif indx == '5':
                    self.goalList.append(GoalLine(i, len(lines)-1-j, True))
                    self.car = Car([i, len(lines)-1])
                    self.car_start_pos = [i, len(lines)-1]
        f.close()
        self.pointListRef.append(Point(-10,-10, True))
        self.resetPoints()

    def resetPoints(self):
        for point in self.pointListRef:
            self.pointList.append(point)

    def reset(self):
        self.car.reset_pos(self.car_start_pos)
        if len(self.pointList) > 0:
            for _ in range(len(self.pointList)):
                self.pointList.pop()
        self.resetPoints()
        self.score = 0
        self.score_last = 0
        self.done = False
        self.goal_timer = time.time()

    def on_draw(self):
        if self.do_rendering:
            """ Render the screen. """
            arcade.start_render()
            for gl in self.goalList:
                gl.draw()
            for point in self.pointList:
                point.draw()
            for wall in self.wallList:
                wall.draw()
            self.car.draw()
            self.barSpeed.draw(self.car.speed*25)
            self.barDist.draw(self.distance[2]*0.5)

            if SHOW_DIST:
                for dot in self.dots:
                    dot.draw()
                    arcade.draw_line(dot._center_x, dot._center_y, self.car._center_x, self.car._center_y, arcade.color.BLACK)


            arcade.draw_text("Score: " + str(self.score), start_x=SCORE_X, start_y=SCORE_Y, font_size=SCORE_TXT_SIZE, color=arcade.color.BLACK)

    def update(self, f):
        """ All the logic to move, and the game logic goes here. """
        if not self.AI_mode and time.time() - self.timestamp > DT:
            self.timestamp = time.time()
            self.car.update()


        for wall in self.wallList:
            if self.collition_detection(wall, self.car):
                self.done = True


        for i, point in enumerate(self.pointList):
            if self.collition_detection(point, self.car):#
                self.score += 10
                self.pointList.pop(i)

        for goal in self.goalList:
            if self.collition_detection(goal, self.car) and time.time() - self.goal_timer > GOAL_DELTA_TIME:
                self.goal_timer = time.time()
                self.resetPoints()

        self.dots = []
        self.distance = []
        for ang in  [-90, -45, -15, 0, 15, 45, 90]: # [-45, -15, 0, 15, 45]: # range(-90, 91, 45):
            dist, dot = self.dist(self.car, ang)
            self.dots.append(dot)
            self.distance.append(dist)
        #for ang in [-15, 15]:# range(-15, 21, 30):
        #    dist, dot = self.dist(self.car, ang)
        #    self.dots.append(dot)
        #    self.distance.append(dist)

    def collition_detection(self, a, b):
        return a._center_y + a.height/2 > b._center_y > a._center_y - a.height/2 and \
               a._center_x + a.width/2 > b._center_x > a._center_x - a.width/2

    def dist(self, obj, ang):
        dot = Dot(obj._center_x, obj._center_y, obj.ang + ang*math.pi/180)
        flagg = False
        for _ in range(50):
            for wall in self.wallList:
                flagg = flagg or self.collition_detection(wall, dot)

            if flagg:
                break
            dot.update()
        return math.sqrt(math.pow(obj._center_x - dot._center_x, 2) + math.pow(obj._center_y - dot._center_y, 2)), dot

    def on_key_press(self, key, modifiers):
        """Called whenever a key is pressed. """
        if not self.AI_mode:
            if key == arcade.key.UP:
                self.car.speed_gain = SPEED_GAIN

            elif key == arcade.key.DOWN:
                self.car.speed_gain = BRAKE_GAIN

            elif key == arcade.key.LEFT:
                self.car.ang_gain = ANG_GAIN

            elif key == arcade.key.RIGHT:
                self.car.ang_gain = -ANG_GAIN

    def on_key_release(self, key, modifires):
        if not self.AI_mode:
            if key == arcade.key.LEFT or key == arcade.key.RIGHT:
                self.car.ang_gain = 0
            elif key == arcade.key.DOWN or arcade.key.UP:
                self.car.speed_gain = FRICTION_GAIN

    def take_action(self, action):
        if action == 0: # No key
            self.car.speed_gain = FRICTION_GAIN
            self.car.ang_gain = 0

        elif action == 1: # Up
            self.car.speed_gain = SPEED_GAIN
            self.car.ang_gain = 0

        elif action == 2: # Down
            self.car.speed_gain = BRAKE_GAIN
            self.car.ang_gain = 0

        elif action == 3: # Left
            self.car.speed_gain = FRICTION_GAIN
            self.car.ang_gain = ANG_GAIN

        elif action == 4: # Right
            self.car.speed_gain = FRICTION_GAIN
            self.car.ang_gain = -ANG_GAIN

        elif action == 5: # Up + Left
            self.car.speed_gain = SPEED_GAIN
            self.car.ang_gain = ANG_GAIN

        elif action == 6: # Up + Right
            self.car.speed_gain = SPEED_GAIN
            self.car.ang_gain = -ANG_GAIN

        elif action == 7: # Down + Left
            self.car.speed_gain = BRAKE_GAIN
            self.car.ang_gain = ANG_GAIN

        elif action == 8: # Down + Right
            self.car.speed_gain = BRAKE_GAIN
            self.car.ang_gain = -ANG_GAIN

        self.update(1/60)
        self.car.update()
        #self.car.draw()


        if self.AI_mode:
            obs = self.distance
            obs.append(self.car.speed)
            obs.append(self.car.ang % (math.pi*2))

            score = self.score - self.score_last
            self.score_last = self.score
            return obs, score, self.done


if __name__ == "__main__":
    game = MyGame(render=True, AI_mode=False)
    game.setup()
    arcade.run()