import * as functions from "firebase-functions";
import * as admin from "firebase-admin";
import { PredictionServiceClient, protos } from "@google-cloud/aiplatform";
import { env } from "../config";

const client = new PredictionServiceClient({
  apiEndpoint: `${env.GCLOUD_REGION}-aiplatform.googleapis.com`,
});

/**
 * HTTPS endpoint that performs inference using the user's personalized model.
 */
export const inference = functions.https.onRequest(
  { memory: "4GiB", concurrency: 1 },
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

      const calibrationDoc = await admin
        .firestore()
        .collection("calibration_sessions")
        .doc(userId)
        .get();

      if (!calibrationDoc.exists) {
        res.status(404).send({ error: "Calibration Record not found" });
        return;
      }

      const data = calibrationDoc.data();
      const gtraceEndpointId = data?.gtrace;
      const conformatEndpointId = data?.confmat;

      if (!gtraceEndpointId || !conformatEndpointId) {
        res.status(400).send({ error: "One or both models are not deployed" });
        return;
      }

      // Transform data from {backrest: 32x32, seat: 32x32}
      // to 32x32x2 format matching training:
      // np.stack([gtrace_back, gtrace_seat], axis=-1)
      const transformTo32x32x2 = (
        backrest: number[][],
        seat: number[][]
      ): number[][][] => {
        const result: number[][][] = [];

        for (let i = 0; i < 32; i++) {
          const row: number[][] = [];
          for (let j = 0; j < 32; j++) {
            row.push([backrest[i][j], seat[i][j]]);
          }
          result.push(row);
        }

        return result;
      };

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

      // Transform both gtrace and conf to 32x32x2 format
      const gtraceData = transformTo32x32x2(gtrace.backrest, gtrace.seat);
      const confData = transformTo32x32x2(conf.backrest, conf.seat);

      // Create instances
      const gtraceInstance: protos.google.protobuf.IValue = {
        listValue: make3DListValue(gtraceData),
      };

      const conformatInstance: protos.google.protobuf.IValue = {
        listValue: make3DListValue(confData),
      };

      const gtraceEndpoint = `projects/${env.GCLOUD_PROJECT_NUMBER}/locations/${env.GCLOUD_REGION}/endpoints/${gtraceEndpointId}`;
      const conformatEndpoint = `projects/${env.GCLOUD_PROJECT_NUMBER}/locations/${env.GCLOUD_REGION}/endpoints/${conformatEndpointId}`;

      console.log("🎯 Gtrace Endpoint:", gtraceEndpoint);
      console.log("🎯 Confmat Endpoint:", conformatEndpoint);

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
        });
        return;
      }
    } catch (error: any) {
      console.error("❌ Inference error:", error);
      res.status(500).send({ error: error.message });
      return;
    }
  }
);
