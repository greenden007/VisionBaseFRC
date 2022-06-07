package frc.robot.subsystems;

import edu.wpi.first.wpilibj2.command.SubsystemBase;
import frc.robot.Constants;

public class SwerveDrivetrain extends SubsystemBase {

	public void drive(double x1, double y1, double x2) {
		double r = Math.sqrt(Math.pow(Constants.Drivetrain.L, 2) + Math.pow(Constants.Drivetrain.W, 2));

		y1 *= -1;

		double a = x1 - x2 * (Constants.Drivetrain.L / r);
		double b = x1 + x2 * (Constants.Drivetrain.L / r);
		double c = y1 - x2 * (Constants.Drivetrain.W / r);
		double d = y1 + x2 * (Constants.Drivetrain.W / r);

		double bRSpd = Math.sqrt((a * a) + (d * d));
		double bLSpd = Math.sqrt((a * a) + (c * c));
		double fRSpd = Math.sqrt((b * b) + (d * d));
		double fLSpd = Math.sqrt((b * b) + (c * c));


		double bRAng = Math.atan2(a, d) / Math.PI;
		double bLAng = Math.atan2(a, c) / Math.PI;
		double fRAng = Math.atan2(b, d) / Math.PI;
		double fLAng = Math.atan2(b, c) / Math.PI;
	}

}
