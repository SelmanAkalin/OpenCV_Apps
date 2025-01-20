import matplotlib.pyplot as plt
import cv2

# Resmi oku
image_path = 'c:/Users/LENOVO/Desktop/OPENCVAPP/indir.png'
original_image = cv2.imread(image_path)

if original_image is None:
    print("Error: Unable to load the image.")
else:
    # BGR'den RGB'ye dönüştür
    rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Resmi göster
    plt.imshow(rgb_image)
    plt.title("Original Image")
    plt.axis("off")  # Eksenleri gizle
    plt.show()
