import cv2
import csv
from ultralytics import YOLO
from paddleocr import PaddleOCR
import logging
logging.getLogger('ultralytics').setLevel(logging.WARNING)

# Initialize YOLO models
coco_model = YOLO('yolov8n.pt', verbose=False)  # General object detection model
license_plate_detector = YOLO('License_plate_ph.pt', verbose=False)  # License plate detection model

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

def process_video(video_source, output_csv='test.csv'):  # Accepts video file as input
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}.")
        return

    # Prepare the CSV file
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Frame', 'License Plate Text', 'Confidence', 'Coordinates'])

        frame_count = 0
        processed_plates = set()  # Set to store already detected license plates

        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or error in capturing frame.")
                break

            # Process only every 3rd frame to reduce load
            if frame_count % 3 != 0:
                frame_count += 1
                continue

            frame_count += 1

            # Detect vehicles with the general-purpose YOLO model (coco_model)
            coco_results = coco_model.predict(frame)

            for result in coco_results:
                boxes = result.boxes
                if boxes is None:
                    continue

                # Filter for vehicles based on class IDs (e.g., car, bus, etc.)
                for box in boxes:
                    cls = int(box.cls)
                    if cls in [2, 3, 5, 7]:  # Vehicle class IDs in COCO (car, bus, motorcycle, truck)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        vehicle_roi = frame[y1:y2, x1:x2]

                        # Detect license plates within the vehicle ROI using the license plate detector
                        lp_results = license_plate_detector.predict(vehicle_roi)

                        for lp_result in lp_results:
                            lp_boxes = lp_result.boxes
                            if lp_boxes is None:
                                continue

                            # Process each detected license plate
                            for lp_box in lp_boxes:
                                lp_x1, lp_y1, lp_x2, lp_y2 = map(int, lp_box.xyxy[0])
                                license_plate_roi = vehicle_roi[lp_y1:lp_y2, lp_x1:lp_x2]

                                # Perform OCR on the license plate region
                                ocr_results = ocr.ocr(license_plate_roi, cls=True)
                                detected_text = ""
                                if ocr_results:
                                    for line in ocr_results:
                                        for word_info in line:
                                            if word_info and len(word_info) > 1:
                                                detected_text += word_info[1][0] + " "

                                # Clean the detected text and remove spaces
                                cleaned_text = detected_text.replace(" ", "").strip()

                                # Check if the license plate has been processed before
                                if cleaned_text and cleaned_text not in processed_plates:
                                    processed_plates.add(cleaned_text)

                                    # Write the result to the CSV file
                                    csvwriter.writerow([frame_count, cleaned_text, float(lp_box.conf), f"{lp_x1},{lp_y1},{lp_x2},{lp_y2}"])
                                    print(f"Detected and saved: {cleaned_text}")

            # Resize frame for better display (optional)
            resized_frame = cv2.resize(frame, (640, 480))
            cv2.imshow("Video Frame", resized_frame)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Provide the path to your MP4 file
    video_file = 'my.mp4'  # Replace with your video file name
    process_video(video_file, 'test.csv')
