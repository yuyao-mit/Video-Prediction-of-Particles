#!/usr/bin/env python
# coding: utf-8
# Type-D: internal bricks: False; mass: equal; interactions: ; time: 10s; number: 20
from datetime import datetime
import random
import os
import pygame
import pymunk
import pymunk.pygame_util
import numpy as np
import cv2
import subprocess

# Global Variables
SIMULATION_TIME = 11 # Total simulation time in seconds
FINAL_VIDEO_TIME = SIMULATION_TIME - 1  # Exclude the first second
FPS = 100  # Frames per second
TOTAL_BALLS = 20
BALL_RADIUS = 24
WIDTH, HEIGHT = 512, 512
COLORS = {'red': '#FF0000', 'green': '#00FF00', 'yellow': '#FFFF00', 'blue': '#0000FF'}
MASSES = {'red': 4, 'yellow': 4, 'green': 4, 'blue': 4}
BRICK_SIZE = 32
SPRING_STIFFNESS = 30
SPRING_MIN_LENGTH = 20
SPRING_MAX_LENGTH = 200
ROD_LENGTH = 60
ROPE_MAX_LENGTH = 100
MAX_INIT_VELOCITY = 100
BRICK_NUM = 0

INPUT_DIR = "./"
OUTPUT_DIR = "./converted/D"

def create_space_with_random_bricks(BRICK_NUM):
    """Initialize the Pymunk space with walls and random bricks."""
    space = pymunk.Space()
    space.gravity = (0, 0)

    # Create walls
    walls = []
    grid_color = (169, 169, 169)
    for i in range(0, WIDTH, BRICK_SIZE):
        walls.append(pymunk.Poly(space.static_body, [(i, 0), (i + BRICK_SIZE, 0), (i + BRICK_SIZE, BRICK_SIZE), (i, BRICK_SIZE)]))
        walls.append(pymunk.Poly(space.static_body, [(i, HEIGHT - BRICK_SIZE), (i + BRICK_SIZE, HEIGHT - BRICK_SIZE),
                                                     (i + BRICK_SIZE, HEIGHT), (i, HEIGHT)]))
    for j in range(0, HEIGHT, BRICK_SIZE):
        walls.append(pymunk.Poly(space.static_body, [(0, j), (BRICK_SIZE, j), (BRICK_SIZE, j + BRICK_SIZE), (0, j + BRICK_SIZE)]))
        walls.append(pymunk.Poly(space.static_body, [(WIDTH - BRICK_SIZE, j), (WIDTH, j),
                                                     (WIDTH, j + BRICK_SIZE), (WIDTH - BRICK_SIZE, j + BRICK_SIZE)]))

    # Random internal bricks
    occupied_positions = set()
    internal_bricks = []
    for _ in range(BRICK_NUM):
        while True:
            x = random.randint(1, (WIDTH // BRICK_SIZE) - 2) * BRICK_SIZE
            y = random.randint(1, (HEIGHT // BRICK_SIZE) - 2) * BRICK_SIZE
            if (x, y) not in occupied_positions:
                occupied_positions.add((x, y))
                brick = pymunk.Poly(space.static_body, [(x, y), (x + BRICK_SIZE, y),
                                                        (x + BRICK_SIZE, y + BRICK_SIZE), (x, y + BRICK_SIZE)])
                brick.elasticity = 1.0
                internal_bricks.append(brick)
                break

    for wall in walls:
        wall.elasticity = 1.0
        space.add(wall)
    space.add(*internal_bricks)

    return space, walls, internal_bricks, grid_color

def create_balls_with_connections(space, MAX_INIT_VELOCITY):
    """Create balls in pairs with specified connections."""
    balls = []
    connections = []
    for _ in range(TOTAL_BALLS // 2):
        # Create two balls
        color1, color2 = random.choice(list(COLORS.keys())), random.choice(list(COLORS.keys()))
        mass1, mass2 = MASSES[color1], MASSES[color2]

        ball_shapes = []
        for color, mass in [(color1, mass1), (color2, mass2)]:
            moment = pymunk.moment_for_circle(mass, 0, BALL_RADIUS)
            body = pymunk.Body(mass, moment)
            body.position = random.uniform(BALL_RADIUS, WIDTH - BALL_RADIUS), random.uniform(BALL_RADIUS, HEIGHT - BALL_RADIUS)
            body.velocity = random.uniform(-MAX_INIT_VELOCITY, MAX_INIT_VELOCITY), random.uniform(-MAX_INIT_VELOCITY, MAX_INIT_VELOCITY)
            shape = pymunk.Circle(body, BALL_RADIUS)
            shape.elasticity = 1.0
            shape.friction = 0
            shape.color = COLORS[color]
            space.add(body, shape)
            ball_shapes.append(shape)

        # Add the two balls to the main ball list
        balls.extend(ball_shapes)

        # Assign a random connection type
        # connection_type = random.choice(["none", "spring", "rigid_rod", "rope"])
        connection_type = random.choice(["none", "spring", "rigid_rod"])
        if connection_type == "spring":
            rest_length = random.uniform(SPRING_MIN_LENGTH, SPRING_MAX_LENGTH)
            spring = pymunk.DampedSpring(
                ball_shapes[0].body, ball_shapes[1].body, (0, 0), (0, 0), rest_length, SPRING_STIFFNESS, 0
            )
            space.add(spring)
            connections.append(("spring", spring))
        elif connection_type == "rigid_rod":
            rod = pymunk.PinJoint(ball_shapes[0].body, ball_shapes[1].body, (0, 0), (0, 0))
            space.add(rod)
            connections.append(("rigid_rod", rod))
        elif connection_type == "rope":
            rope = pymunk.SlideJoint(
                ball_shapes[0].body, ball_shapes[1].body, (0, 0), (0, 0), 0, ROPE_MAX_LENGTH
            )
            space.add(rope)
            connections.append(("rope", rope))

    return balls, connections

def simulate_and_record_with_connections(space, walls, internal_bricks, grid_color, balls, connections):
    """Simulate the environment with connections and record a high-resolution video."""
    pygame.init()
    screen = pygame.Surface((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    draw_options = pymunk.pygame_util.DrawOptions(screen)

    high_res_width, high_res_height = WIDTH * 2, HEIGHT * 2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(VIDEO_FILENAME, fourcc, FPS, (high_res_width, high_res_height))

    steps = int(SIMULATION_TIME * FPS)
    start_frame = FPS

    for frame in range(steps):
        screen.fill((255, 255, 255))

        for wall in walls + internal_bricks:
            pygame.draw.polygon(screen, grid_color, [(int(p[0]), int(p[1])) for p in wall.get_vertices()])
            pygame.draw.polygon(screen, (0, 0, 0), [(int(p[0]), int(p[1])) for p in wall.get_vertices()], width=1)

        for ball in balls:
            pygame.draw.circle(screen, ball.color, (int(ball.body.position.x), int(ball.body.position.y)), BALL_RADIUS)
            pygame.draw.circle(screen, (0, 0, 0), (int(ball.body.position.x), int(ball.body.position.y)), BALL_RADIUS, width=1)

        # Visualize connections
        for conn_type, conn in connections:
            pos1 = (int(conn.a.position.x), int(conn.a.position.y))
            pos2 = (int(conn.b.position.x), int(conn.b.position.y))

            if conn_type == "spring":
                draw_dashed_line(screen, pos1, pos2, color=(0, 0, 0), dash_length=5)
            elif conn_type == "rigid_rod":
                pygame.draw.line(screen, (0, 0, 0), pos1, pos2, width=4)
            elif conn_type == "rope":
                pygame.draw.line(screen, (0, 0, 0), pos1, pos2, width=2)

        if frame >= start_frame:
            frame_data = pygame.surfarray.array3d(screen)
            frame_data = np.transpose(frame_data, (1, 0, 2))
            high_res_frame = cv2.resize(frame_data, (high_res_width, high_res_height), interpolation=cv2.INTER_LINEAR)
            video.write(cv2.cvtColor(high_res_frame, cv2.COLOR_RGB2BGR))

        space.step(1 / FPS)
        clock.tick(FPS)

    video.release()
    pygame.quit()
    print(f"Simulation complete. Video saved as {VIDEO_FILENAME}.")

def draw_dashed_line(surface, start_pos, end_pos, color, dash_length=5):
    """Draw a dashed line between two points."""
    x1, y1 = start_pos
    x2, y2 = end_pos
    total_length = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    num_dashes = int(total_length // dash_length)

    for i in range(num_dashes + 1):
        t1 = i / num_dashes
        t2 = (i + 0.5) / num_dashes
        if t2 > 1.0:
            t2 = 1.0
        segment_start = (
            int(x1 + t1 * (x2 - x1)),
            int(y1 + t1 * (y2 - y1))
        )
        segment_end = (
            int(x1 + t2 * (x2 - x1)),
            int(y1 + t2 * (y2 - y1))
        )
        pygame.draw.line(surface, color, segment_start, segment_end, width=1)

def convert_videos(input_dir, output_dir):
    """Convert video files to a specified format."""
    os.makedirs(output_dir, exist_ok=True)
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".mp4"):
            input_path = os.path.join(input_dir, file_name)
            base_name, ext = os.path.splitext(file_name)
            output_file_name = f"{base_name}_converted{ext}"
            output_path = os.path.join(output_dir, output_file_name)
            command = ["ffmpeg", "-i", input_path, "-vcodec", "libx264", "-acodec", "aac", output_path]
            print(f"Converting {file_name} to {output_file_name}...")
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    print(f"All files have been converted and saved in {output_dir}.")

if __name__ == "__main__":
    # Create a unique filename for the video
    VIDEO_FILENAME = f"type_D_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"

    # Initialize the Pymunk simulation space with walls and bricks
    simulation_space, simulation_walls, internal_bricks, grid_wall_color = create_space_with_random_bricks(BRICK_NUM)

    # Create balls and their connections
    balls, connections = create_balls_with_connections(simulation_space, MAX_INIT_VELOCITY)

    # Simulate the environment and record the video
    simulate_and_record_with_connections(simulation_space, simulation_walls, internal_bricks, grid_wall_color, balls, connections)

    # Convert the generated video files
    convert_videos(INPUT_DIR, OUTPUT_DIR)

