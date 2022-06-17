package frc.robot.drivers;

import edu.wpi.first.math.geometry.Translation2d;
import edu.wpi.first.math.kinematics.SwerveDriveKinematics;
import edu.wpi.first.math.kinematics.SwerveDriveOdometry;
import frc.robot.Constants;

public class SwerveOdometry {
	private Translation2d frontLeft = new Translation2d(Constants.Drivetrain.fL[0], Constants.Drivetrain.fL[1]);
	private Translation2d frontRight = new Translation2d(Constants.Drivetrain.fR[0], Constants.Drivetrain.fR[1]);
	private Translation2d backLeft = new Translation2d(Constants.Drivetrain.bL[0], Constants.Drivetrain.bL[1]);
	private Translation2d backRight = new Translation2d(Constants.Drivetrain.bR[0], Constants.Drivetrain.bR[1]);

	private SwerveDriveKinematics kinematics = new SwerveDriveKinematics(
			frontLeft, frontRight, backLeft, backRight
	);

}
