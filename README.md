# capstone-project

This is my senior capstone project, as a double major in both Computer Science and Data Science at Pacific University!

I am creating a web app that takes a photo of a chessboard, and uses an AI model to identify and classify each chess piece. To do this I am using a custom YOLO model trained on my own dataset, combined with some image analysis to map those chess pieces back to specific squares on a chessboard. (Gaussian Blur, Canny Edge Detection, Hough Transform, and some custom code to single out the corners of the chessboard - needed to calculate the location of each square on the chessboard)