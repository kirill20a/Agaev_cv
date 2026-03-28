import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from pathlib import Path
from scipy.optimize import linear_sum_assignment


def area(labeled, label):
    return (labeled == label).sum()

def centroid(labeled, label=1):
    y, x = np.where(labeled == label)
    return np.mean(y), np.mean(x)

def neighbours4(y, x):
    return [(y, x+1), (y+1, x), (y, x-1), (y-1, x)] 

def neighboursX(y, x):
    return [(y-1, x+1), (y+1, x+1), (y+1, x-1), (y-1, x-1)]  

def neighbours8(y, x):
    return neighbours4(y, x) + neighboursX(y, x)

def get_bounds(labeled, label, connectivity=neighbours4):
    pos = np.where(labeled == label)
    bounds = []
    for y, x in zip(*pos):
        for yn, xn in connectivity(y, x):
            if yn < 0 or yn >= labeled.shape[0]:  
                bounds.append((y, x))
                break
            elif xn < 0 or xn >= labeled.shape[1]: 
                bounds.append((y, x))
                break
            elif labeled[yn, xn] != label:
                bounds.append((y, x))
                break
    return bounds

def find_objects(labeled):
    """Находит все объекты и возвращает их центры и метки"""
    objects = []
    for label_id in range(1, np.max(labeled) + 1):
        cy, cx = centroid(labeled, label_id)
        objects.append({
            'id': label_id,
            'centroid': (cx, cy),
            'area': area(labeled, label_id)
        })
    return objects

def match_objects(prev_objects, curr_objects, max_dist=50):
    """Сопоставляет объекты между кадрами"""
    if not prev_objects:
        return [], list(range(len(curr_objects)))
    
    if not curr_objects:
        return list(range(len(prev_objects))), []

    distances = np.zeros((len(prev_objects), len(curr_objects)))
    for i, prev in enumerate(prev_objects):
        for j, curr in enumerate(curr_objects):
            dx = prev['centroid'][0] - curr['centroid'][0]
            dy = prev['centroid'][1] - curr['centroid'][1]
            distances[i, j] = np.sqrt(dx*dx + dy*dy)
    
    row_ind, col_ind = linear_sum_assignment(distances)

    matched_prev = []
    matched_curr = []
    used_curr = set()
    used_prev = set()
    
    for i, j in zip(row_ind, col_ind):
        if distances[i, j] <= max_dist:
            matched_prev.append(i)
            matched_curr.append(j)
            used_prev.add(i)
            used_curr.add(j)
    

    unmatched_prev = [i for i in range(len(prev_objects)) if i not in used_prev]
    unmatched_curr = [j for j in range(len(curr_objects)) if j not in used_curr]
    
    return matched_prev, matched_curr, unmatched_prev, unmatched_curr

def load_images(folder_path):
    """Загрузка всех .npy файлов в порядке возрастания"""
    files = sorted(Path(folder_path).glob("*.npy"),
                   key=lambda x: int(x.stem.split('_')[1]) if '_' in x.stem else int(x.stem))
    
    print(f"Найдено файлов: {len(files)}")
    images = []
    for f in files:
        img = np.load(f)
        if np.max(img) > 1:
            img = (img > 0).astype(int)
        images.append(img)
    
    return images

def track_objects(images):
    """Отслеживание объектов по кадрам"""
    trajectories = {}
    prev_objects = []
    object_counter = 0
    
    for frame, img in enumerate(images):
        print(f"Кадр {frame}...")
        

        labeled = label(img)
        curr_objects = find_objects(labeled)
        
        if frame == 0:

            for obj in curr_objects:
                trajectories[object_counter] = [(frame, obj['centroid'][0], obj['centroid'][1])]
                obj['track_id'] = object_counter
                object_counter += 1
        else:

            matched_prev, matched_curr, unmatched_prev, unmatched_curr = match_objects(prev_objects, curr_objects)
            

            for prev_idx, curr_idx in zip(matched_prev, matched_curr):
                track_id = prev_objects[prev_idx]['track_id']
                x, y = curr_objects[curr_idx]['centroid']
                trajectories[track_id].append((frame, x, y))

                curr_objects[curr_idx]['track_id'] = track_id
            

            for curr_idx in unmatched_curr:
                x, y = curr_objects[curr_idx]['centroid']
                trajectories[object_counter] = [(frame, x, y)]
                curr_objects[curr_idx]['track_id'] = object_counter
                object_counter += 1
            

        prev_objects = curr_objects
    
    return trajectories

def plot_trajectories(trajectories):
    """Построение графика траекторий"""
    plt.figure(figsize=(12, 10))
    

    colors = plt.cm.tab20(np.linspace(0, 1, len(trajectories)))
    
    for i, (obj_id, path) in enumerate(trajectories.items()):
        path = np.array(path)
        x = path[:, 1]  
        y = path[:, 2]  
        
        
        plt.plot(x, y, '-', color=colors[i], linewidth=2, label=f'Object {obj_id}')
        
        
        plt.scatter(x, y, color=colors[i], s=30, zorder=5)
        
        
        if len(x) > 0:
            plt.scatter(x[0], y[0], color=colors[i], s=150, marker='o', 
                       edgecolors='black', linewidth=2, zorder=6)
        
        
        if len(x) > 0:
            plt.scatter(x[-1], y[-1], color=colors[i], s=150, marker='s', 
                       edgecolors='black', linewidth=2, zorder=6)
        
        print(f"Объект {obj_id}: {len(path)} точек")
    
    plt.xlabel('X координата', fontsize=12)
    plt.ylabel('Y координата', fontsize=12)
    plt.title('Траектории движения объектов', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.subplots_adjust(right=0.85)
    plt.savefig('trajectories.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    folder = "images"
    
    if not Path(folder).exists():
        print(f"Ошибка: папка '{folder}' не найдена!")
        return
    
    images = load_images(folder)
    print(f"Загружено {len(images)} кадров\n")
    
    trajectories = track_objects(images)
    print(f"\nНайдено {len(trajectories)} объектов\n")
    
    plot_trajectories(trajectories)

if __name__ == "__main__":
    main()