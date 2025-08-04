#!/usr/bin/env python3
import pygame
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class JoystickGUI(Node):
    def __init__(self):
        super().__init__('cmd_vel_joystick')
        self.publisher = self.create_publisher(Twist, '/unitree_go2/cmd_vel', 10)

    def send(self, x, y, z):
        msg = Twist()
        msg.linear.x =  y
        msg.linear.y = -x
        msg.angular.z = z
        self.publisher.publish(msg)

def main():
    rclpy.init()
    node = JoystickGUI()

    pygame.init()
    screen = pygame.display.set_mode((400, 400))
    pygame.display.set_caption("Go2 Joystick Controller")
    font = pygame.font.SysFont(None, 24)
    clock = pygame.time.Clock()

    center = (200, 200)

    running = True
    while running and rclpy.ok():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        mouse = pygame.mouse.get_pressed()
        pos = pygame.mouse.get_pos()

        vx, vy, omega = 0.0, 0.0, 0.0

        if mouse[0]:  # clic gauche
            dx = max(-100, min(100, pos[0] - center[0]))
            dy = max(-100, min(100, pos[1] - center[1]))
            vx =  dx / 100.0
            vy = -dy / 100.0

        if keys[pygame.K_a]:
            omega = 1.0
        elif keys[pygame.K_e]:
            omega = -1.0

        node.send(vx, vy, omega)

        # UI
        screen.fill((30, 30, 30))
        pygame.draw.circle(screen, (50, 50, 50), center, 100)
        pygame.draw.circle(screen, (0, 255, 200), (int(center[0] + vx * 100), int(center[1] - vy * 100)), 20)
        txt = font.render(f"vx: {vy:.2f}  vy: {vx:.2f}  ω: {omega:.2f}", True, (255, 255, 255))
        screen.blit(txt, (10, 10))
        pygame.display.flip()

        clock.tick(30)

    pygame.quit()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
