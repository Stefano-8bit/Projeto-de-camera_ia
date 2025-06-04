import cv2
import time
import numpy as np

class PersonTimer:
    def __init__(self, max_disappeared=30):
        self.next_id = 0
        self.boxes = {}
        self.start_times = {}
        self.disappeared = {}
        self.finished = []
        self.max_disappeared = max_disappeared

    def update(self, detections):
        if len(self.boxes) == 0:
            for box in detections:
                self._add(box)
            return

        existing_ids = list(self.boxes.keys())
        existing_centroids = np.array(
            [[x + w / 2, y + h / 2] for (x, y, w, h) in self.boxes.values()]
        )
        new_centroids = np.array(
            [[x + w / 2, y + h / 2] for (x, y, w, h) in detections]
        )

        if len(new_centroids) == 0:
            for pid in existing_ids:
                self.disappeared[pid] += 1
                if self.disappeared[pid] > self.max_disappeared:
                    self._remove(pid)
            return

        D = np.linalg.norm(existing_centroids[:, None] - new_centroids[None, :], axis=2)
        assigned = set()
        for i, pid in enumerate(existing_ids):
            j = np.argmin(D[i])
            if D[i, j] < 50:
                self.boxes[pid] = detections[j]
                self.disappeared[pid] = 0
                assigned.add(j)
            else:
                self.disappeared[pid] += 1
                if self.disappeared[pid] > self.max_disappeared:
                    self._remove(pid)

        for j, box in enumerate(detections):
            if j not in assigned:
                self._add(box)

    def _add(self, box):
        pid = self.next_id
        self.next_id += 1
        self.boxes[pid] = box
        self.start_times[pid] = time.time()
        self.disappeared[pid] = 0

    def _remove(self, pid):
        start = self.start_times.pop(pid, None)
        if start is not None:
            duration = time.time() - start
            self.finished.append((pid, duration))
        self.boxes.pop(pid, None)
        self.disappeared.pop(pid, None)

    def report(self):
        lines = [f"Pessoa {pid}: {dur:.2f} segundos" for pid, dur in self.finished]
        return "\n".join(lines)


def main():
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    cap = cv2.VideoCapture(0)
    timer = PersonTimer()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes, _ = hog.detectMultiScale(gray, winStride=(8, 8))
        timer.update(boxes)

        for pid, (x, y, w, h) in timer.boxes.items():
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {pid}",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(timer.report())


if __name__ == "__main__":
    main()
