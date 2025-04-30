import os
import argparse

def check_feedback(feedback_dir, threshold):
    image_count = len([f for f in os.listdir(os.path.join(feedback_dir, 'images')) if f.endswith('.png')])
    label_count = len([f for f in os.listdir(os.path.join(feedback_dir, 'labels')) if f.endswith('.txt')])

    print(f"Feedback Images: {image_count}, Feedback Labels: {label_count}")

    if image_count > threshold or label_count > threshold:
        with open("feedback_check_done.txt", "w") as f:
            f.write("Feedback threshold exceeded, retraining triggered.\n")
        return True
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check feedback data and trigger retraining if needed.")
    parser.add_argument('--feedback-dir', type=str, required=True, help="Directory containing feedback data.")
    parser.add_argument('--threshold', type=int, required=True, help="Threshold for triggering retraining.")

    args = parser.parse_args()

    check_feedback(args.feedback_dir, args.threshold)
