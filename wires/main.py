import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.morphology import opening

for i in range(1, 7):
    filename = f"wires_from_para/wires{i}.npy"
    image = np.load(filename)
    
    struct = np.ones((3, 1))
    process = opening(image, struct)
    
    labeled_image = label(image)
    labeled_process = label(process)
    
    print(f"\nФайл: wires{i}.npy")
    print(f"Original: {np.max(labeled_image)}")
    print(f"Processed: {np.max(labeled_process)}")
    
    for wire_num in range(1, np.max(labeled_image) + 1):
        wire_mask = (labeled_image == wire_num)
        wire_parts = labeled_process[wire_mask]
        unique_parts = np.unique(wire_parts[wire_parts > 0])
        num_parts = len(unique_parts)
        
        if num_parts == 0:
            otvet = "нет"
        elif num_parts == 1:
            otvet = "целый"
        else:
            otvet = f"порван на {num_parts} частей"
        
        print(f"  Провод {wire_num}, {otvet}")
    
    if i == 6:#чтобы показать картинку изменить число (1-6)
        plt.subplot(121)
        plt.imshow(image)
        plt.title(f'Original wires{i}')
        plt.subplot(122)
        plt.imshow(process)
        plt.title(f'Processed wires{i}')
        plt.show()
