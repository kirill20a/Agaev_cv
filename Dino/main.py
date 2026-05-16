"""
T-Rex Dino Game Bot - v11.7 (Hybrid)
"""
import time
import numpy as np
import cv2
from mss import mss
import pyautogui
from pathlib import Path
from collections import deque

class DinoGameBot:
    def __init__(self, region=None):
        if region is None:
            self.region = {'top': 350, 'left': 50, 'width': 1000, 'height': 200}
        else:
            self.region = region
        
        self.sct = mss()
        
        self.dino_left = 0
        self.dino_right = 90
        self.dino_top = 70
        self.dino_bottom = 150
        
        self.cactus_zone = {'x': 90, 'y': 65, 'w': 600, 'h': 95}
        self.bird_zone = {'x': 90, 'y': 25, 'w': 600, 'h': 130}
        
        self.binary_threshold = 83
        self.min_cactus_w = 8
        self.min_cactus_h = 15
        
        self.dist_small = 95
        self.dist_large = 80
        self.dist_group_2small = 75
        self.dist_group_2large = 75
        self.dist_group_3small = 75
        self.dist_group_4mixed = 75
        self.dist_bird_jump = 95
        self.dist_bird_duck = 100
        self.critical_gap = 60
        
        # Единственное новое: если кактус ближе X - игнорируем ВСЕХ птиц
        self.cactus_danger_dist = 100  
        
        self.game_speed = 1.0
        
        self.jumps = 0
        self.ducks = 0
        self.start_time = time.time()
        self.last_action = 0
        
        self.is_ducking = False
        self.duck_start_time = 0
        self.duck_end_time = 0
        self.last_bird_seen_time = 0
        self.post_duck_delay = 0.15
        
        self.fps_times = deque(maxlen=60)
        
        self.template_dir = Path("templates")
        self.duck_template = None
        self.jump_bird_template = None
        self.bird_match_threshold = 0.55
        
        self.load_templates()

    def load_templates(self):
        bird_low_path = self.template_dir / "bird_low.png"
        if bird_low_path.exists():
            self.duck_template = cv2.imread(str(bird_low_path), cv2.IMREAD_GRAYSCALE)
        
        cactus_b_path = self.template_dir / "cactus_b.png"
        if cactus_b_path.exists():
            self.jump_bird_template = cv2.imread(str(cactus_b_path), cv2.IMREAD_GRAYSCALE)

    def capture(self):
        try:
            img = self.sct.grab(self.region)
            return np.array(img, dtype=np.uint8)[:, :, :3]
        except:
            return None

    def classify_cactus(self, width, height):
        if width < 22:
            return 'CACT_1S', self.dist_small
        elif width < 32:
            return 'CACT_1L', self.dist_large
        elif width < 42:
            return 'CACT_2S', self.dist_group_2small
        elif width < 56:
            if height > 55:
                return 'CACT_2L', self.dist_group_2large
            else:
                return 'CACT_3S', self.dist_group_3small
        elif width < 65:
            return 'CACT_3S', self.dist_group_3small
        else:
            return 'CACT_4M', self.dist_group_4mixed

    def detect(self, frame):
        if frame is None:
            return None
        
        h, w = frame.shape[:2]
        gray = np.dot(frame[..., :3], [0.114, 0.587, 0.299]).astype(np.uint8)
        _, binary = cv2.threshold(gray, self.binary_threshold, 255, cv2.THRESH_BINARY_INV)
        
        obstacles = []
        bird_detected = False
        has_duck_bird = False
        nearest_cactus_dist = 999  # Для проверки danger zone
        
        cx, cy = self.cactus_zone['x'], self.cactus_zone['y']
        cw = min(self.cactus_zone['w'], w - cx)
        ch = min(self.cactus_zone['h'], h - cy)
        
        if cw > 0 and ch > 0:
            roi = binary[cy:cy+ch, cx:cx+cw]
            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                x, y, cw_i, ch_i = cv2.boundingRect(cnt)
                
                if cw_i < self.min_cactus_w or ch_i < self.min_cactus_h:
                    continue
                if cw_i > ch_i * 3:
                    continue
                
                abs_x = cx + x
                dist = abs_x - self.dino_right
                label, req = self.classify_cactus(cw_i, ch_i)
                
                if dist < nearest_cactus_dist:
                    nearest_cactus_dist = dist
                
                obstacles.append({
                    'type': 'cactus',
                    'label': label,
                    'dist': dist,
                    'req': req,
                    'action': 'jump',
                    'priority': 2,
                    'x': abs_x, 'y': cy + y,
                    'w': cw_i, 'h': ch_i
                })
        
        # ВАЖНО: если кактус в опасной зоне - полностью пропускаем поиск птиц
        cactus_danger = nearest_cactus_dist < self.cactus_danger_dist
        
        if not cactus_danger:
            bx, by = self.bird_zone['x'], self.bird_zone['y']
            bw = min(self.bird_zone['w'], w - bx)
            bh = min(self.bird_zone['h'], h - by)
            
            if bw > 0 and bh > 0:
                bird_roi = gray[by:by+bh, bx:bx+bw]
                
                if self.duck_template is not None:
                    if self.duck_template.shape[0] <= bh and self.duck_template.shape[1] <= bw:
                        result = cv2.matchTemplate(bird_roi, self.duck_template, cv2.TM_CCOEFF_NORMED)
                        locations = np.where(result >= self.bird_match_threshold)
                        
                        if len(locations[0]) > 0:
                            has_duck_bird = True
                            bird_detected = True
                            self.last_bird_seen_time = time.time()
                        
                        for pt in zip(*locations[::-1]):
                            bird_x = bx + pt[0]
                            bird_y = by + pt[1]
                            dist = bird_x - self.dino_right
                            
                            obstacles.append({
                                'type': 'bird',
                                'label': 'BIRD_DUCK',
                                'dist': dist,
                                'req': self.dist_bird_duck,
                                'action': 'duck',
                                'priority': 0,
                                'x': bird_x, 'y': bird_y,
                                'w': self.duck_template.shape[1],
                                'h': self.duck_template.shape[0]
                            })
                
                if self.jump_bird_template is not None and not has_duck_bird:
                    if self.jump_bird_template.shape[0] <= bh and self.jump_bird_template.shape[1] <= bw:
                        result = cv2.matchTemplate(bird_roi, self.jump_bird_template, cv2.TM_CCOEFF_NORMED)
                        locations = np.where(result >= self.bird_match_threshold)
                        
                        for pt in zip(*locations[::-1]):
                            bird_x = bx + pt[0]
                            bird_y = by + pt[1]
                            dist = bird_x - self.dino_right
                            
                            obstacles.append({
                                'type': 'bird',
                                'label': 'BIRD_JUMP',
                                'dist': dist,
                                'req': self.dist_bird_jump,
                                'action': 'jump',
                                'priority': 1,
                                'x': bird_x, 'y': bird_y,
                                'w': self.jump_bird_template.shape[1],
                                'h': self.jump_bird_template.shape[0]
                            })
        
        if obstacles:
            obstacles.sort(key=lambda o: (o['priority'], o['dist']))
            
            if len(obstacles) > 1:
                gap = obstacles[1]['dist'] - obstacles[0]['dist']
                if gap < self.critical_gap:
                    obstacles[0]['req'] = max(obstacles[0]['req'], self.dist_group_4mixed)
            
            nearest = obstacles[0]
            now = time.time()
            
            if nearest['action'] == 'jump':
                if self.is_ducking and now < self.duck_end_time:
                    should_act = False
                elif not self.is_ducking and self.duck_end_time > 0:
                    if now < self.duck_end_time + self.post_duck_delay:
                        should_act = False
                    else:
                        should_act = nearest['dist'] <= nearest['req']
                else:
                    should_act = nearest['dist'] <= nearest['req']
            else:
                should_act = nearest['dist'] <= nearest['req']
            
            return {
                'should_act': should_act,
                'nearest': nearest,
                'obstacles': obstacles,
                'cactus_danger': cactus_danger,
                'debug': self.visualize(frame, obstacles, should_act, cactus_danger)
            }
        
        return {
            'should_act': False,
            'nearest': None,
            'obstacles': [],
            'cactus_danger': cactus_danger,
            'debug': self.visualize(frame, [], False, cactus_danger)
        }

    def visualize(self, frame, obstacles, should_act, cactus_danger=False):
        debug = frame.copy()
        h, w = debug.shape[:2]
        
        cv2.rectangle(debug, (self.dino_left, self.dino_top), 
                     (self.dino_right, self.dino_bottom), (255, 0, 0), 2)
        
        # Линия опасной зоны
        if self.cactus_danger_dist > 0:
            danger_x = self.dino_right + self.cactus_danger_dist
            line_color = (0, 0, 255) if cactus_danger else (100, 100, 100)
            cv2.line(debug, (danger_x, 0), (danger_x, h), line_color, 1)
            if cactus_danger:
                cv2.putText(debug, "BIRDS BLOCKED", (danger_x + 2, 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        for obs in obstacles:
            if obs['type'] == 'bird':
                color = (0, 255, 255) if obs['action'] == 'duck' else (0, 255, 0)
            else:
                colors = {
                    'CACT_1S': (200, 200, 0),
                    'CACT_1L': (0, 200, 200),
                    'CACT_2S': (0, 150, 255),
                    'CACT_2L': (0, 100, 255),
                    'CACT_3S': (0, 50, 255),
                    'CACT_4M': (0, 0, 255),
                }
                color = colors.get(obs['label'], (255, 100, 0))
            
            cv2.rectangle(debug, (obs['x'], obs['y']), 
                         (obs['x'] + obs['w'], obs['y'] + obs['h']), color, 2)
        
        if obstacles:
            line_x = self.dino_right + obstacles[0]['req']
            color = (0, 0, 255) if should_act else (0, 255, 255)
            cv2.line(debug, (line_x, 0), (line_x, h), color, 2)
        
        fps = self.get_fps()
        score = (self.jumps + self.ducks) * 100

        
        return debug               

    def get_fps(self):
        now = time.time()
        self.fps_times.append(now)
        if len(self.fps_times) > 1:
            dt = self.fps_times[-1] - self.fps_times[0]
            return (len(self.fps_times) - 1) / dt if dt > 0 else 0
        return 0

    def update_speed(self):
        elapsed = time.time() - self.start_time
        
        if elapsed < 10: time_speed = 1.1
        elif elapsed < 22: time_speed = 2.1
        elif elapsed < 32: time_speed = 3.5
        elif elapsed < 40: time_speed = 4.2
        elif elapsed < 49: time_speed = 4.9
        elif elapsed < 60: time_speed = 5.5
        elif elapsed < 69: time_speed = 6.2
        elif elapsed < 78: time_speed = 8.0
        elif elapsed < 100: time_speed = 8.8
        elif elapsed < 130: time_speed = 9.1
        elif elapsed < 180: time_speed = 9.5
        else: time_speed = 9.65
        
        self.game_speed = self.game_speed * 0.85 + time_speed * 0.265
        factor = 1 + max(0, self.game_speed - 1) * 0.2

        self.dist_small = int(95 * factor)
        self.dist_large = int(90 * factor)
        self.dist_group_2small = int(80 * factor)
        self.dist_group_2large = int(80 * factor)
        self.dist_group_3small = int(75 * factor)
        self.dist_group_4mixed = int(75 * factor)
        self.dist_bird_duck = int(100 * factor)
        self.dist_bird_jump = int(95 * factor)

    def act(self, action):
        now = time.time()
        
        if action == 'jump':
            if self.is_ducking and now < self.duck_end_time:
                return False
            if not self.is_ducking and self.duck_end_time > 0:
                if now < self.duck_end_time + self.post_duck_delay:
                    return False
        
        if now - self.last_action < 0.05:
            return False
        
        if action == 'duck':
            if not self.is_ducking:
                pyautogui.keyDown('down')
                self.is_ducking = True
                self.duck_start_time = now
            
            self.duck_end_time = now + 0.5
            
            if self.last_bird_seen_time > now - 0.1:
                self.duck_end_time = max(self.duck_end_time, now + 0.3)
            
            self.ducks += 1
            
        elif action == 'jump':
            if self.is_ducking:
                pyautogui.keyUp('down')
                self.is_ducking = False
                time.sleep(0.01)
            
            pyautogui.press('space')
            self.jumps += 1
        
        self.last_action = now
        return True

    def update_keys(self):
        now = time.time()
        
        if self.is_ducking:
            if now >= self.duck_end_time:
                if self.last_bird_seen_time < now - 0.3:
                    pyautogui.keyUp('down')
                    self.is_ducking = False

    def run(self):
        print(f"Dino Bot v11.7 - Hybrid | Danger zone: {self.cactus_danger_dist}px (0=off)")
        input()
        
        pyautogui.press('space')
        time.sleep(0.3)
        self.start_time = time.time()
        
        frame_count = 0
        
        try:
            while True:
                frame = self.capture()
                if frame is None:
                    continue
                
                detection = self.detect(frame)
                if detection is None:
                    continue
                
                self.update_speed()
                self.update_keys()
                
                if detection['should_act'] and detection['nearest']:
                    self.act(detection['nearest']['action'])
                
                if detection['debug'] is not None and frame_count % 2 == 0:
                    cv2.imshow('Dino Bot', detection['debug'])
                
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                
                frame_count += 1
                if frame_count % 100 == 0:
                    time.sleep(0.0001)
                
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()

    def cleanup(self):
        elapsed = time.time() - self.start_time
        score = (self.jumps + self.ducks) * 100
        
        print(f"Score: ~{score} | Time: {elapsed:.0f}s | Jumps: {self.jumps} | Ducks: {self.ducks}")
        
        cv2.destroyAllWindows()
        pyautogui.keyUp('down')

def main():
    bot = DinoGameBot({'top': 350, 'left': 50, 'width': 1000, 'height': 200})
    bot.run()

if __name__ == "__main__":
    main()