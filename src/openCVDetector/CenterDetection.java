package openCVDetector;

import org.opencv.core.Core;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

public class CenterDetection {

	public static final int WEIGHT_BLUR_SIDES = 5;
	public static final boolean ENABLE_WEIGHT = true;
	public static final float WEIGHT_DIVISOR = 1.0f;
	public static final double GRADIENT_THRESHOLD = 50.0;

	public static void testPossibleCenters(int x, int y, Mat weight, double gx, double gy, Mat out) {
		for (int cy = 0; cy < out.rows(); ++cy) {
			Mat Or = out.row(cy);
			Mat Wr = weight.row(cy);
			for (int cx = 0; cx < out.cols(); ++cx) {
				if (x == cx && y == cy) {
					continue;
				}
				//creez un vector pt centrele posibile asociate gradientilor
				
				double dx = x - cx;
				double dy = y - cy;
				
				// normalizez distanta
				
				double magnitude = Math.sqrt((dx * dx) + (dy * dy));
				dx = dx / magnitude;
				dy = dy / magnitude;
				double dotProduct = dx * gx + dy * gy;
				
				// multiplic in functie de weight
				
				Or.put(0, cx, Or.get(0, cx)[0] + dotProduct * dotProduct * (Wr.get(0, cx)[0] / WEIGHT_DIVISOR));
			}
		}
	}

	public static void main(String[] args) {

		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		System.out.println("\nRunning Eye Center Detector!");

		CascadeClassifier faceDetector = new CascadeClassifier("haarcascade_frontalface_alt.xml");
		CascadeClassifier eyeDetector = new CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml");

		// incarc imaginea si clasificatorii;

		Mat image = Highgui.imread("img3.jpg");

		System.out.println("\n\nImage is loaded!");

		String faces;
		String eyes;

		MatOfRect faceDetections = new MatOfRect();
		MatOfRect eyeDetections = new MatOfRect();

		Mat face;
		Mat crop = null;

		faceDetector.detectMultiScale(image, faceDetections);

		for (int i = 0; i < faceDetections.toArray().length; i++) {

			faces = "Face" + i + ".png";

			face = image.submat(faceDetections.toArray()[i]);
			crop = face.submat(4, (2 * face.width()) / 3, 0, face.height());

			Highgui.imwrite(faces, face);

			System.out.println("\nFace is detected!");

			eyeDetector.detectMultiScale(crop, eyeDetections, 1.1, 2, 0, new Size(30, 30), new Size());

			if (eyeDetections.toArray().length == 0) {

				System.out.println(" Not a face" + i);
			} else {

				for (int j = 0; j < eyeDetections.toArray().length; j++) {

					Mat eye1 = crop.submat(eyeDetections.toArray()[j]);
					Imgproc.cvtColor(eye1, eye1, Imgproc.COLOR_BGR2GRAY, 1);
					
					Mat eye=new Mat(eye1.rows(),eye1.cols(),CvType.CV_64F);
					eye1.convertTo(eye, CvType.CV_64F);
					
					eyes = "Eye" + j + ".png";
					Highgui.imwrite(eyes, eye);

					System.out.println("\nEye" + j + " is detected!");

					// determin gradientii
					
					Mat gradientX = new Mat(eye.rows(), eye.cols(), CvType.CV_64F);
					Mat gradientY = new Mat(eye.rows(), eye.cols(), CvType.CV_64F);
				
					Imgproc.Sobel(eye, gradientX, -1, 1, 0);
					Imgproc.Sobel(eye, gradientY, -1, 0, 1);
					
					// determin magnitudinea gradientilor
					
					Mat mags=new Mat(eye.rows(),eye.cols(),CvType.CV_64F);
					
					Core.magnitude(gradientX, gradientY, mags);
					
					String xGradient = "gradientX_eye" + j + ".png";
					String yGradient = "gradientY_eye" + j + ".png";

					Highgui.imwrite(xGradient, gradientX);
					Highgui.imwrite(yGradient, gradientY);

					System.out.println("\nGradient on X and Y are determined for eye" + j + " !");

					// calculez threshold-ul pentru magnitudine
					
					MatOfDouble stdMagnGrad = new MatOfDouble();
					MatOfDouble meanMagnGrad = new MatOfDouble();
					Core.meanStdDev(mags, meanMagnGrad, stdMagnGrad);
					double stdDev = stdMagnGrad.get(0, 0)[0] / Math.sqrt(mags.rows() * mags.cols()); 
					
					double gradientThresh = GRADIENT_THRESHOLD * stdDev + meanMagnGrad.get(0, 0)[0];

					// normalizez
					
					for (int y = 0; y < eye.rows(); y++) {
						Mat Xr = gradientX.row(y);
						Mat Yr = gradientY.row(y);
						Mat Mr = mags.row(y);

						for (int x = 0; x < eye.cols(); x++) {
							double gX = Xr.get(0, x)[0];
							double gY = Yr.get(0, x)[0];
							double magnitude = Mr.get(0, x)[0];

							if (magnitude > gradientThresh) {
								Xr.put(0, x, gX / magnitude);
								Yr.put(0, x, gY / magnitude);
							} else {
								Xr.put(0, x, 0.0);
								Yr.put(0, x, 0.0);
							}
						}
					}

					// aplic filtrul Gaussian

					Mat weight = new Mat();
					Imgproc.GaussianBlur(eye, weight, new Size(WEIGHT_BLUR_SIDES, WEIGHT_BLUR_SIDES), 0, 0);

					for (int y = 0; y < weight.rows(); y++) {
						Mat row = weight.row(y);
						for (int x = 0; x < weight.cols(); x++) {
							row.put(0, x, 255 - row.get(0, x)[0]);
						}
					}

					// gasesc centrele posibile
					
					Mat outSum = Mat.zeros(eye.rows(), eye.cols(), CvType.CV_64F);
					for (int y = 0; y < weight.rows(); y++) {
						Mat Xr = gradientX.row(y);
						Mat Yr = gradientY.row(y);

						for (int x = 0; x < weight.cols(); x++) {
							double gX = Xr.get(0, x)[0];
							double gY = Yr.get(0, x)[0];
							if (gX == 0.0 && gY == 0.0) {
								continue;
							}
							testPossibleCenters(x, y, weight, gX, gY, outSum);
						}
					}

					// gasesc centrul final
					
					double numGradients = (weight.rows() * weight.cols());
					Mat out = new Mat();
					outSum.convertTo(out, CvType.CV_32F, 1.0 / numGradients);

					MinMaxLocResult mmr = Core.minMaxLoc(out);
					Point center = mmr.maxLoc;

					Core.circle(eye, center, 10, new Scalar(0, 255, 0));

					String centerFinal = "Center" + j + ".png";

					Highgui.imwrite(centerFinal, eye);

					System.out.println("\nEye center for eye" + j + " is detected!");
				}

			}
		}
		System.out.println("\n\nStop Eye Center Detector!");
	}
}
