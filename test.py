import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)                    # 定义了摄像头对象，id为o一般就是本机的摄像头
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():                      # 判断摄像头是否已经打开
    success, image = cap.read()              # success变量表示当前是否读取到照片，image存储当前一帧的照片
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # cvtColor为颜色空间转换函数，相当于对照片进行预处理。此处将照片从BGR格式转换成了RGB
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Initially set finger count to 0 for each cap
    fingerCount = 0

    if results.multi_hand_landmarks:

      for hand_landmarks in results.multi_hand_landmarks:
        # Get hand index to check label (left or right)
        handIndex = results.multi_hand_landmarks.index(hand_landmarks)
        handLabel = results.multi_handedness[handIndex].classification[0].label

        # Set variable to keep landmarks positions (x and y)
        handLandmarks = []

        # Fill list with x and y positions of each landmark
        for landmarks in hand_landmarks.landmark:
          handLandmarks.append([landmarks.x, landmarks.y])

        print(handLandmarks)

        # Test conditions for each finger: Count is increased if finger is
        #   considered raised.
        # Thumb: TIP x position must be greater or lower than IP x position,
        #   deppeding on hand label.
        if handLabel == "Left" and handLandmarks[4][0] > handLandmarks[3][0]:     # 先确定左右手，再看大拇指的特征，因为大
          fingerCount = fingerCount+1                                             # 拇指是横着伸出去的，所以通过x轴判断是否伸出去了大拇指
        elif handLabel == "Right" and handLandmarks[4][0] < handLandmarks[3][0]:
          fingerCount = fingerCount+1

        # Other fingers: TIP y position must be lower than PIP y position,
        #   as image origin is in the upper left corner.
        if handLandmarks[8][1] < handLandmarks[6][1]:       #Index finger         # 纯机械的判断伸出了几根手指
          fingerCount = fingerCount+1
        if handLandmarks[12][1] < handLandmarks[10][1]:     #Middle finger
          fingerCount = fingerCount+1
        if handLandmarks[16][1] < handLandmarks[14][1]:     #Ring finger
          fingerCount = fingerCount+1
        if handLandmarks[20][1] < handLandmarks[18][1]:     #Pinky
          fingerCount = fingerCount+1

        # Draw hand landmarks
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    # Display finger count
    cv2.putText(image, str(fingerCount), (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)

    print(fingerCount)

    # Display image
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()