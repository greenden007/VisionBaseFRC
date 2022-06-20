package frc.robot.subsystems;

import com.ctre.phoenix.motorcontrol.NeutralMode;
import com.ctre.phoenix.motorcontrol.StatusFrameEnhanced;
import com.ctre.phoenix.motorcontrol.SupplyCurrentLimitConfiguration;
import com.ctre.phoenix.motorcontrol.can.WPI_TalonSRX;
import com.ctre.phoenix.sensors.SensorVelocityMeasPeriod;
import com.revrobotics.CANSparkMax;
import com.revrobotics.CANSparkMaxLowLevel;
import com.revrobotics.RelativeEncoder;
import com.revrobotics.SparkMaxPIDController;
import edu.wpi.first.math.kinematics.DifferentialDriveWheelSpeeds;
import edu.wpi.first.wpilibj2.command.SubsystemBase;
import frc.robot.Constants;
import frc.robot.RobotMap;

public class SwerveDrivetrain extends SubsystemBase {

	private static final int MAX_AMPS = 40;
	private static final double OPEN_LOOP_RAMP_SECONDS = 0.3;

	private static final WPI_TalonSRX bRPow;
	private static final WPI_TalonSRX bLPow;
	private static final WPI_TalonSRX fRPow;
	private static final WPI_TalonSRX fLPow;

	private static final CANSparkMax bRRot = new CANSparkMax(RobotMap.Drivetrain.bRRot, CANSparkMaxLowLevel.MotorType.kBrushless);
	private static final RelativeEncoder bREncoder = bRRot.getEncoder();
	private static final SparkMaxPIDController bRControl = bRRot.getPIDController();

	private static final CANSparkMax bLRot = new CANSparkMax(RobotMap.Drivetrain.bLRot, CANSparkMaxLowLevel.MotorType.kBrushless);
	private static final RelativeEncoder bLEncoder = bLRot.getEncoder();
	private static final SparkMaxPIDController bLControl = bLRot.getPIDController();

	private static final CANSparkMax fRRot = new CANSparkMax(RobotMap.Drivetrain.fRRot, CANSparkMaxLowLevel.MotorType.kBrushless);
	private static final RelativeEncoder fREncoder = fRRot.getEncoder();
	private static final SparkMaxPIDController fRControl = fRRot.getPIDController();
	
	private static final CANSparkMax fLRot = new CANSparkMax(RobotMap.Drivetrain.fLRot, CANSparkMaxLowLevel.MotorType.kBrushless);
	private static final RelativeEncoder fLEncoder = fLRot.getEncoder();
	private static final SparkMaxPIDController fLControl = fLRot.getPIDController();

	public SwerveDrivetrain() {
		super();
		bRPow = new WPI_TalonSRX(RobotMap.Drivetrain.bRPow);
		bLPow = new WPI_TalonSRX(RobotMap.Drivetrain.bLPow);
		fRPow = new WPI_TalonSRX(RobotMap.Drivetrain.fRPow);
		fLPow = new WPI_TalonSRX(RobotMap.Drivetrain.fLPow);

		bRPow.setStatusFramePeriod(StatusFrameEnhanced.Status_4_AinTempVbat, 20);
		bLPow.setStatusFramePeriod(StatusFrameEnhanced.Status_4_AinTempVbat, 20);
		fRPow.setStatusFramePeriod(StatusFrameEnhanced.Status_4_AinTempVbat, 20);
		fLPow.setStatusFramePeriod(StatusFrameEnhanced.Status_4_AinTempVbat, 20);
		
		bRPow.setNeutralMode(NeutralMode.Brake);
		bLPow.setNeutralMode(NeutralMode.Brake);
		fRPow.setNeutralMode(NeutralMode.Brake);
		fLPow.setNeutralMode(NeutralMode.Brake);

		bRPow.setSafetyEnabled(false);
		bLPow.setSafetyEnabled(false);
		fRPow.setSafetyEnabled(false);
		fLPow.setSafetyEnabled(false);

		bRPow.configOpenloopRamp(OPEN_LOOP_RAMP_SECONDS);
		bLPow.configOpenloopRamp(OPEN_LOOP_RAMP_SECONDS);
		fRPow.configOpenloopRamp(OPEN_LOOP_RAMP_SECONDS);
		fLPow.configOpenloopRamp(OPEN_LOOP_RAMP_SECONDS);

		bRPow.configVelocityMeasurementPeriod(SensorVelocityMeasPeriod.Period_1Ms);
		bLPow.configVelocityMeasurementPeriod(SensorVelocityMeasPeriod.Period_1Ms);
		fRPow.configVelocityMeasurementPeriod(SensorVelocityMeasPeriod.Period_1Ms);
		fLPow.configVelocityMeasurementPeriod(SensorVelocityMeasPeriod.Period_1Ms);

		bRPow.configVelocityMeasurementWindow(32);
		bLPow.configVelocityMeasurementWindow(32);
		fLPow.configVelocityMeasurementWindow(32);
		fRPow.configVelocityMeasurementWindow(32);

		bRRot.restoreFactoryDefaults();
		bRRot.enableVoltageCompensation(12);
		bRRot.setIdleMode(CANSparkMax.IdleMode.kBrake);
		bRRot.setSmartCurrentLimit(MAX_AMPS, MAX_AMPS);
		bRControl.setOutputRange(-1, 1);

		bRControl.setP(0.0003);
		bRControl.setFF(0.000175); //placeholders from old code

		bLRot.restoreFactoryDefaults();
		bLRot.enableVoltageCompensation(12);
		bLRot.setIdleMode(CANSparkMax.IdleMode.kBrake);
		bLRot.setSmartCurrentLimit(MAX_AMPS, MAX_AMPS);
		bLControl.setOutputRange(-1, 1);

		bLControl.setP(0.0003);
		bLControl.setFF(0.000175);

		fRRot.restoreFactoryDefaults();
		fRRot.enableVoltageCompensation(12);
		fRRot.setIdleMode(CANSparkMax.IdleMode.kBrake);
		fRRot.setSmartCurrentLimit(MAX_AMPS, MAX_AMPS);
		fRControl.setOutputRange(-1, 1);

		fRControl.setP(0.0003);
		fRControl.setFF(0.000175);

		fLRot.restoreFactoryDefaults();
		fLRot.enableVoltageCompensation(12);
		fLRot.setIdleMode(CANSparkMax.IdleMode.kBrake);
		fLRot.setSmartCurrentLimit(MAX_AMPS, MAX_AMPS);
		fLControl.setOutputRange(-1, 1);

		fLControl.setP(0.0003);
		fLControl.setFF(0.000175);
	}

	

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

		bRRot.set(bRAng);
		bLRot.set(bLAng);
		fRRot.set(fRAng);
		fLRot.set(fLAng);
		bRPow.set(bRSpd);
		bLPow.set(bLSpd);
		fRPow.set(fRSpd);
		fLPow.set(fLSpd);
	}

	public DifferentialDriveWheelSpeeds getPowWheelSpeeds() {
		return new DifferentialDriveWheelSpeeds(bRPow.getSelectedSensorVelocity(), fLPow.getSelectedSensorVelocity());
	}

	public void resetEncoders() {
		bREncoder.setPosition(0);
		bLEncoder.setPosition(0);
		fREncoder.setPosition(0);
		fLEncoder.setPosition(0);
		bRPow.setSelectedSensorPosition(0);
		bLPow.setSelectedSensorPosition(0);
		fRPow.setSelectedSensorPosition(0);
		fLPow.setSelectedSensorPosition(0);
	}
}
