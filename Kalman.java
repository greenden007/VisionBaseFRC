package frc.robot.drivers.vision;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.video.KalmanFilter;

public class Kalman extends KalmanFilter {
	private KalmanFilter kalman;
	private Point lRes;
	private double dTime;

	public void init() {}

	public Kalman(Point pt) {
		kalman = new KalmanFilter(4, 2, 0, CvType.CV_16F);

		Mat transMatrix = new Mat(4, 4, CvType.CV_16F, new Scalar(0));
		float[] tm = {
				1, 0, 1, 0,
				0, 1, 0, 1,
				0, 0, 1, 0,
				0, 0, 0, 1
		};
		transMatrix.put(0, 0, tm);

		kalman.set_transitionMatrix(transMatrix);

		lRes = pt;
		Mat statePre = new Mat(4, 1, CvType.CV_16F, new Scalar(0));
		statePre.put(0, 0, pt.x);
		statePre.put(1, 0, pt.y);
		kalman.set_statePre(statePre);

		kalman.set_measurementMatrix(Mat.eye(2, 4, CvType.CV_16F));

		Mat processNoiseCov = Mat.eye(4, 4, CvType.CV_16F);
		processNoiseCov = processNoiseCov.mul(processNoiseCov, 1e-4);
		kalman.set_processNoiseCov(processNoiseCov);

		Mat id1 = Mat.eye(2,2, CvType.CV_16F);
		id1 = id1.mul(id1, 1e-1);
		kalman.set_measurementNoiseCov(id1);

		Mat id2 = Mat.eye(4,4, CvType.CV_16F);
		kalman.set_errorCovPost(id2);
	}

	public Kalman(Point pt, double dt, double accel_noise_mag) {
		kalman = new KalmanFilter(4, 2, 0, CvType.CV_16F);
		dTime = dt;
		Mat transitionMatrix = new Mat(4, 3, CvType.CV_16F, new Scalar(0));
		float[] tm = {
				1, 0, 1, 0,
				0, 1, 0, 1,
				0, 0, 1, 0,
				0, 0, 0, 1
		};
		transitionMatrix.put(0, 0, tm);

		kalman.set_transitionMatrix(transitionMatrix);

		lRes = pt;
		Mat statePre = new Mat(4, 1, CvType.CV_16F, new Scalar(0));
		statePre.put(0, 0, pt.x);
		statePre.put(1, 0, pt.y);
		statePre.put(2, 0, 0);
		statePre.put(3, 0, 0);
		kalman.set_statePre(statePre);

		Mat statePost = new Mat(4, 1, CvType.CV_16F, new Scalar(0));
		statePost.put(0, 0, pt.x);
		statePost.put(1, 0, pt.y);
		statePost.put(2, 0, 0);
		statePost.put(3, 0, 0);
		kalman.set_statePost(statePost);

		kalman.set_measurementMatrix(Mat.eye(2, 4, CvType.CV_16F));

		Mat processNoiseCov = new Mat(4, 4, CvType.CV_16F, new Scalar(0));
		float[] deltaT = {
				(float) (Math.pow(dTime, 4) / 4.0), 0,
				(float) (Math.pow(dTime, 3) / 2.0) , 0, 0,
				(float) (Math.pow(dTime, 4) / 4.0), 0,
				(float) (Math.pow(dTime, 3) / 2.0),
				(float) (Math.pow(dTime, 3) / 2.0), 0,
				(float) Math.pow(dTime, 2.0), 0 , 0,
				(float) (Math.pow(dTime, 3) / 2.0) , 0,
				(float) Math.pow(dTime, 2.0)
		};
		processNoiseCov = processNoiseCov.mul(processNoiseCov, accel_noise_mag);
		kalman.set_processNoiseCov(processNoiseCov);

		Mat id1 = Mat.eye(2, 2, CvType.CV_16F);
		id1.mul(id1, 1e-1);
		kalman.set_measurementMatrix(id1);

		Mat id2 = Mat.eye(4, 4, CvType.CV_16F);
		id2 = id2.mul(id2, .1);
		kalman.set_errorCovPost(id2);
	}

	public Point getPrediction() {
		Mat prediction = kalman.predict();
		lRes = new Point(prediction.get(0, 0)[0], prediction.get(1, 0)[0]);
		return lRes;
	}

	public Point update(Point p, boolean isDataCorr) {
		Mat measurement = new Mat(2, 1, CvType.CV_16F, new Scalar(0));
		if(!isDataCorr) {
			measurement.put(0, 0, lRes.x);
			measurement.put(1, 0, lRes.y);
		} else {
			measurement.put(0, 0, p.x);
			measurement.put(1, 0, p.y);
		}

		Mat estimated = kalman.correct(measurement);
		lRes.x = estimated.get(0, 0)[0];
		lRes.y = estimated.get(1, 0)[0];
		return lRes;
	}

	public Point correction(Point p) {
		Mat measurement = new Mat(2, 1, CvType.CV_16F, new Scalar(0));
		measurement.put(0, 0, p.x);
		measurement.put(1, 0, p.y);

		Mat estimated = kalman.correct(measurement);
		lRes.x = estimated.get(0,0)[0];
		lRes.y = estimated.get(1, 0)[0];
		return lRes;
	}

	/**
	 * @return Returns the deltatime
	 */
	public double getDTime() {
		return dTime;
	}

	/**
	 * @return Returns the lastResult
	 */
	public Point getLastResult() {
		return lRes;
	}

	/**
	 * @param lastRes The last result to set
	 */
	public void setLastRes(Point lastRes) {
		lRes = lastRes;
	}
}
