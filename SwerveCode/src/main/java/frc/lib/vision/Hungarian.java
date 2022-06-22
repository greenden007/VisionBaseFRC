package frc.lib.vision;

import java.util.Arrays;

public class Hungarian {
	private final double[][] costMatrix;
	private final int rows, cols, dim;
	private final double[] labelByWorker, labelByJob;
	private final int[] minSlackWBJ;
	private final double[] minSlackVBJ;
	private final int[] matchJBW, matchWBJ;
	private final int[] parentWByCommittedJob;
	private final boolean[] committedWorkers;

	public Hungarian(double[][] costMatrix) {
		this.dim = Math.max(costMatrix.length, costMatrix[0].length);
		this.rows = costMatrix.length;
		this.cols = costMatrix[0].length;
		this.costMatrix = new double[this.dim][this.dim];
		for (int w = 0; w < this.dim; w++) {
			if (w < costMatrix.length) {
				if (costMatrix[w].length != this.cols) {
					throw new IllegalArgumentException("Irregular Cost Matrix");
				}
			} else {
				this.costMatrix[w] = new double[this.dim];
			}
		}
		labelByWorker = new double[this.dim];
		labelByJob = new double[this.dim];
		minSlackWBJ = new int[this.dim];
		minSlackVBJ = new double[this.dim];
		committedWorkers = new boolean[this.dim];
		parentWByCommittedJob = new int[this.dim];
		matchJBW = new int[this.dim];
		Arrays.fill(matchJBW, -1);
		matchWBJ = new int[this.dim];
		Arrays.fill(matchWBJ, -1);
	}

	protected void computeInitialSol() {
		for (int j = 0; j < dim; j++) {
			labelByJob[j] = Double.POSITIVE_INFINITY;
		}
		for (int i = 0; i < dim; i++) {
			for (int j = 0; j < dim; j++) {
				if (costMatrix[i][j] < labelByJob[j]) {
					labelByJob[j] = costMatrix[i][j];
				}
			}
		}
	}

	public int[] execute() {
	}

	protected void executePhase() {
		while (true) {
			int minSlack
		}
	}
}
