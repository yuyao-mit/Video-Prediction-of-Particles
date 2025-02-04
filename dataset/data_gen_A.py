#!/usr/bin/env python
# coding: utf-8

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
SIMULATION_TIME = 4  # Total simulation time in seconds
FINAL_VIDEO_TIME = SIMULATION_TIME - 1  # Exclude the first second
FPS = 100  # Frames per second
TOTAL_BALLS = 12
BALL_RADIUS = 30
WIDTH, HEIGHT = 512, 512
COLORS = {'red': '#FF0000', 'green': '#00FF00', 'yellow': '#FFFF00', 'blue': '#0000FF'}
MASSES = {'red': 1, 'yellow': 2, 'green': 4, 'blue': 8}
BRICK_SIZE = 32
INPUT_DIR = "/home/gridsan/yyao/Research_Projects/Particle_Simulation/graph_video_prediction/dataset/data_gen"
OUTPUT_DIR = "/home/gridsan/yyao/Research_Projects/Particle_Simulation/graph_video_prediction/dataset/raw"


def create_space_with_random_bricks():
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
    for _ in range(10):
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


def create_balls(space):
    """Create balls with random positions and velocities."""
    balls = []
    for _ in range(TOTAL_BALLS):
        color = random.choice(list(COLORS.keys()))
        mass = MASSES[color]
        moment = pymunk.moment_for_circle(mass, 0, BALL_RADIUS)
        body = pymunk.Body(mass, moment)
        body.position = random.uniform(BALL_RADIUS, WIDTH - BALL_RADIUS), random.uniform(BALL_RADIUS, HEIGHT - BALL_RADIUS)
        body.velocity = random.uniform(-150, 150), random.uniform(-150, 150)
        shape = pymunk.Circle(body, BALL_RADIUS)
        shape.elasticity = 1.0
        shape.friction = 0
        shape.color = COLORS[color]
        space.add(body, shape)
        balls.append(shape)
    return balls


def simulate_and_record_with_internal_bricks(space, walls, internal_bricks, grid_color, balls):
    """Simulate the environment and record a high-resolution video."""
    pygame.init()
    screen = pygame.Surface((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    draw_options = pymunk.pygame_util.DrawOptions(screen)

    # Video writer
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
    for _ in range(1):
        VIDEO_FILENAME = f"type_A_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        simulation_space, simulation_walls, internal_bricks, grid_wall_color = create_space_with_random_bricks()
        balls = create_balls(simulation_space)
        simulate_and_record_with_internal_bricks(simulation_space, simulation_walls, internal_bricks, grid_wall_color, balls)

    convert_videos(INPUT_DIR, OUTPUT_DIR)
