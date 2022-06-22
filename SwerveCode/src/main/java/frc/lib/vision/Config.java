package frc.lib.vision;

import java.awt.Toolkit;
import org.opencv.core.Scalar;

public class Config {
	public static final int F_WIDTH = Toolkit.getDefaultToolkit().getScreenSize().width / 2;
	public static final int F_HEIGHT = Toolkit.getDefaultToolkit().getScreenSize().height / 2;

	public static double minB = 250;
	public static double maxB = 3000;

	public static Scalar cols[] = {
			new Scalar(255, 0, 0), new Scalar(0, 255, 0),
			new Scalar(0, 0, 255), new Scalar(255, 255, 0),
			new Scalar(0, 255, 255), new Scalar(255, 0, 255),
			new Scalar(255, 127, 255), new Scalar(127, 0, 255),
			new Scalar(127, 0, 127)
	};

	public static double learning = 0.005;

	public static double dt = 0.2;
	public static double accelNoise = 0.5;
	public static double distThresh = 360;
	public static int maxSkippedFrames = 10;
	public static int maxTraceLength = 10;
}
