package frc.lib.vision;

import java.util.Vector;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;

public abstract class JTracker {
	public float dt;
	public float accel_noise_mag;
	public double dist_thresh;
	public int max_allowed_skipped_frames;
	public int max_trace_length;
	public Vector<Track> tracks;
	public int track_removed;

	public abstract void update(Vector<Rect> rectArrays, Vector<Point> detections, Mat imag);
}
