package frc.lib.vision;

import org.opencv.core.Point;
import java.util.Vector;

public class Track {

	public Vector<Point> trace;
	public Vector<Point> history;
	public static int NextTrackID;
	public int track_id;
	public int skipped_frames;
	public int crossBorder;
	public Point prediction;
	public Kalman KF;

	/**
	 * @param pt
	 * @param dt delta time increments =0.2
	 * @param accel_noise_mag
	 * @param id
	 */
	public Track(Point pt, float dt, float accel_noise_mag, int id) {
		trace = new Vector<>();
		history = new Vector<>();
		track_id = id;
		KF = new Kalman(pt, dt, accel_noise_mag);
		prediction = pt;
		skipped_frames = 0;
		crossBorder = 0;
	}
}
