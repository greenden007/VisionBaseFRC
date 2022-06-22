package frc.lib;

import java.util.Vector;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

public class Tracker extends JTracker {
	int nextTrackID = 0;
	Vector<Integer> feed = new Vector<>();

	public Tracker(float dt, float accel_noise_mag, double distThresh, int max_allowed_skipped_frames, int max_trace_length) {
		tracks = new Vector<>();
		this.dt = dt;
		this.accel_noise_mag = accel_noise_mag;
		this.dist_thresh = distThresh;
		this.max_allowed_skipped_frames = max_allowed_skipped_frames;
		this.max_trace_length = max_trace_length;
		track_removed = 0;
	}

	static Scalar[] Colors = {
			new Scalar(255, 0, 0), new Scalar(0, 255, 0),
			new Scalar(0, 0, 255), new Scalar(255, 255, 0),
			new Scalar(0, 255, 255), new Scalar(255, 0, 255),
			new Scalar(255, 127, 255), new Scalar(127, 0, 255),
			new Scalar(127, 0, 127)
	};

	double euclideanDist(Point p, Point q) {
		Point diff = new Point(p.x - q.x, p.y - q.y);
		return Math.sqrt(diff.x * diff.x + diff.y * diff.y);
	}

	public void update(Vector<Rect> rectArray, Vector<Point> detections, Mat imag) {

	}
}
