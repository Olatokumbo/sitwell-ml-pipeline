import * as functions from "firebase-functions";
import * as admin from "firebase-admin";
import { PredictionServiceClient, protos } from "@google-cloud/aiplatform";
import { env } from "../config";

const client = new PredictionServiceClient({
  apiEndpoint: `${env.GCLOUD_REGION}-aiplatform.googleapis.com`,
});

/**
 * Normalize sensor data using StandardScaler parameters
 */
const normalizeData = (
  backrest: number[][],
  seat: number[][],
  mean: number[],
  scale: number[]
): number[][][] => {
  // Flatten both sensors into single array (2048 values)
  const flat: number[] = [];

  // Add backrest (1024 values) - row by row
  for (let i = 0; i < 32; i++) {
    for (let j = 0; j < 32; j++) {
      flat.push(backrest[i][j]);
    }
  }

  // Add seat (1024 values) - row by row
  for (let i = 0; i < 32; i++) {
    for (let j = 0; j < 32; j++) {
      flat.push(seat[i][j]);
    }
  }

  // Verify we have correct number of features
  if (flat.length !== mean.length || flat.length !== scale.length) {
    throw new Error(
      `Feature mismatch: data=${flat.length}, mean=${mean.length}, scale=${scale.length}`
    );
  }

  // Apply normalization: (x - mean) / scale
  const normalized = flat.map((val, idx) => (val - mean[idx]) / scale[idx]);

  // Reshape to 32x32x2 format (height, width, channels)
  const result: number[][][] = [];
  for (let i = 0; i < 32; i++) {
    const row: number[][] = [];
    for (let j = 0; j < 32; j++) {
      const backrestIdx = i * 32 + j;
      const seatIdx = 1024 + i * 32 + j;
      row.push([normalized[backrestIdx], normalized[seatIdx]]);
    }
    result.push(row);
  }

  return result;
};

/**
 * HTTPS endpoint that performs inference using the user's personalized model.
 */
export const inference = functions.https.onRequest(
  { memory: "128MiB", concurrency: 1, minInstances: 1 },
  async (req, res) => {
    if (req.method !== "POST") {
      res.status(405).send({ error: "Method not allowed" });
      return;
    }

    try {
      const { gtrace, conf, userId } = req.body as {
        userId: string;
        gtrace: { backrest: number[][]; seat: number[][] };
        conf: { backrest: number[][]; seat: number[][] };
      };

      if (!userId || !gtrace || !conf) {
        res.status(400).send({ error: "Missing required fields" });
        return;
      }

      // Validate input dimensions
      if (
        !gtrace.backrest ||
        !gtrace.seat ||
        gtrace.backrest.length !== 32 ||
        gtrace.seat.length !== 32
      ) {
        res.status(400).send({ error: "Invalid gtrace data dimensions" });
        return;
      }

      if (
        !conf.backrest ||
        !conf.seat ||
        conf.backrest.length !== 32 ||
        conf.seat.length !== 32
      ) {
        res.status(400).send({ error: "Invalid conf data dimensions" });
        return;
      }

      console.log(`🔍 Processing inference request for user: ${userId}`);

      // Get user's calibration data from Firestore
      const calibrationDoc = await admin
        .firestore()
        .collection("calibration_sessions")
        .doc(userId)
        .get();

      if (!calibrationDoc.exists) {
        res.status(404).send({ error: "Calibration record not found" });
        return;
      }

      const data = calibrationDoc.data();
      const gtraceEndpointId = data?.gtrace;
      const conformatEndpointId = data?.confmat;

      // 🔑 Get scaler metadata from Firestore
      const gtraceScaler = data?.gtrace_scaler;
      const confmatScaler = data?.confmat_scaler;

      // Validate endpoints exist
      if (!gtraceEndpointId || !conformatEndpointId) {
        res.status(400).send({
          error: "One or both models are not deployed",
          details: {
            gtraceDeployed: !!gtraceEndpointId,
            confmatDeployed: !!conformatEndpointId,
          },
        });
        return;
      }

      // Validate scalers exist
      if (!gtraceScaler?.mean || !gtraceScaler?.scale) {
        res.status(400).send({
          error: "GTrace scaler not found",
          details: "Model may not be trained yet",
        });
        return;
      }

      if (!confmatScaler?.mean || !confmatScaler?.scale) {
        res.status(400).send({
          error: "ConfMat scaler not found",
          details: "Model may not be trained yet",
        });
        return;
      }

      console.log(`✅ Loaded scalers for user ${userId}`);
      console.log(
        `   GTrace: ${gtraceScaler.n_features} features, ${gtraceScaler.n_samples_fit} samples fitted`
      );
      console.log(
        `   ConfMat: ${confmatScaler.n_features} features, ${confmatScaler.n_samples_fit} samples fitted`
      );

      // Helper to convert 1D array → protobuf list
      const makeListValue = (
        arr: number[]
      ): protos.google.protobuf.IListValue => ({
        values: arr.map((n) => ({ numberValue: n })),
      });

      // Helper to convert 2D array → protobuf nested list
      const make2DListValue = (
        arr: number[][]
      ): protos.google.protobuf.IListValue => ({
        values: arr.map((row) => ({ listValue: makeListValue(row) })),
      });

      // Helper to convert 3D array (32x32x2) → protobuf nested list
      const make3DListValue = (
        arr: number[][][]
      ): protos.google.protobuf.IListValue => ({
        values: arr.map((matrix) => ({ listValue: make2DListValue(matrix) })),
      });

      // 🔑 NORMALIZE the data using user's personal scalers
      console.log("📊 Normalizing gtrace data...");
      const gtraceData = normalizeData(
        gtrace.backrest,
        gtrace.seat,
        gtraceScaler.mean,
        gtraceScaler.scale
      );

      console.log("📊 Normalizing confmat data...");
      const confData = normalizeData(
        conf.backrest,
        conf.seat,
        confmatScaler.mean,
        confmatScaler.scale
      );

      // Optional: Verify normalization (can remove in production)
      const flatGtrace = gtraceData.flat(2);
      const gtraceMean =
        flatGtrace.reduce((a, b) => a + b, 0) / flatGtrace.length;
      const gtraceVariance =
        flatGtrace.reduce(
          (sum, val) => sum + Math.pow(val - gtraceMean, 2),
          0
        ) / flatGtrace.length;
      const gtraceStd = Math.sqrt(gtraceVariance);

      console.log(
        `✅ GTrace normalized - mean: ${gtraceMean.toFixed(
          6
        )}, std: ${gtraceStd.toFixed(6)}`
      );

      if (Math.abs(gtraceMean) > 0.1 || Math.abs(gtraceStd - 1.0) > 0.2) {
        console.warn(
          "⚠️  Warning: GTrace normalization may be incorrect (mean should be ~0, std should be ~1)"
        );
      }

      // Create instances
      const gtraceInstance: protos.google.protobuf.IValue = {
        listValue: make3DListValue(gtraceData),
      };

      const conformatInstance: protos.google.protobuf.IValue = {
        listValue: make3DListValue(confData),
      };

      const gtraceEndpoint = `projects/${env.GCLOUD_PROJECT_NUMBER}/locations/${env.GCLOUD_REGION}/endpoints/${gtraceEndpointId}`;
      const conformatEndpoint = `projects/${env.GCLOUD_PROJECT_NUMBER}/locations/${env.GCLOUD_REGION}/endpoints/${conformatEndpointId}`;

      console.log("🎯 Sending predictions to:");
      console.log(`   GTrace: ${gtraceEndpoint}`);
      console.log(`   ConfMat: ${conformatEndpoint}`);

      // 🔄 Run both predictions in parallel
      try {
        const [gtraceResponse, conformatResponse] = await Promise.all([
          client.predict({
            endpoint: gtraceEndpoint,
            instances: [gtraceInstance],
          }),
          client.predict({
            endpoint: conformatEndpoint,
            instances: [conformatInstance],
          }),
        ]);

        // Extract probabilities from the response
        const gtraceProbabilities =
          gtraceResponse[0].predictions?.[0]?.listValue?.values?.map(
            (v: any) => v.numberValue || 0
          ) || [];

        const confProbabilities =
          conformatResponse[0].predictions?.[0]?.listValue?.values?.map(
            (v: any) => v.numberValue || 0
          ) || [];

        // Get the predicted class (argmax)
        const gtracePredictedClass = gtraceProbabilities.indexOf(
          Math.max(...gtraceProbabilities)
        );
        const confPredictedClass = confProbabilities.indexOf(
          Math.max(...confProbabilities)
        );

        // Get confidence scores
        const gtraceConfidence = gtraceProbabilities[gtracePredictedClass];
        const confConfidence = confProbabilities[confPredictedClass];

        console.log("✅ Predictions successful:");
        console.log(
          `   GTrace: class ${gtracePredictedClass}, confidence ${(
            gtraceConfidence * 100
          ).toFixed(2)}%`
        );
        console.log(
          `   ConfMat: class ${confPredictedClass}, confidence ${(
            confConfidence * 100
          ).toFixed(2)}%`
        );

        res.status(200).send({
          message: "Inference successful",
          gtrace: {
            predictedClass: gtracePredictedClass,
            confidence: gtraceConfidence,
            probabilities: gtraceProbabilities,
          },
          confmat: {
            predictedClass: confPredictedClass,
            confidence: confConfidence,
            probabilities: confProbabilities,
          },
        });
      } catch (predictionError: any) {
        console.error("❌ Prediction failed:", predictionError);
        res.status(500).send({
          error: "Prediction failed",
          details: predictionError.message,
          stack: predictionError.stack,
        });
        return;
      }
    } catch (error: any) {
      console.error("❌ Inference error:", error);
      res.status(500).send({
        error: error.message,
        stack: error.stack,
      });
      return;
    }
  }
);
