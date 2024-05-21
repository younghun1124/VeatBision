import cv2
import numpy as np

def draw_circle(window_name, radius, duration):
    # Create a black image window
    img = np.zeros((500, 500, 3), np.uint8)
    cv2.namedWindow(window_name)

    # Calculate the center of the image
    center_x = img.shape[1] // 2
    center_y = img.shape[0] // 2
    center_x = img.shape[1] // 2
    center_y = img.shape[0] // 2

    # Start with a small radius
    current_radius = 10

    # Get the current time
    start_time = cv2.getTickCount()

    while True:
        # Calculate the elapsed time
        elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()

        # Increase the radius based on the elapsed time
        current_radius = int(radius * elapsed_time / duration)

        # Draw the filled circle on the image
        cv2.circle(img, (center_x, center_y), current_radius, (0, 255, 0), -1)

        # Draw the border circle on the image
        
        cv2.circle(img, (center_x, center_y), radius, (0, 0, 255), 2)

        # Show the image
        cv2.imshow(window_name, img)

        # Break the loop if the duration has passed
        if elapsed_time >= duration:
            break

        # Wait for a key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close the window
    cv2.destroyAllWindows()

# Set the radius and duration
radius = 100
duration = 5  # in seconds


# Call the function to draw the circle
draw_circle("Growing Circle", radius, duration)
